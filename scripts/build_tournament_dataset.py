"""Tournament dataset builder: the "Slippi Dumps" OneDrive archives -> a parsed, skill-labelled Root.

Same layout as data/full/Root (Parsed/<md5> zlib parquet, parsed.sqlite, parsed.pkl, raw.json,
meta.json), built one archive at a time: download into --work, unpack into a fresh directory, parse
with slippi_db's own functions, label each player, then delete the archive and its replays.

Labels go in netplay.name, where the loader's name_from_metadata reads them: "GM Player" for HAL's 38
pros and SSBMRank's top players, "Master Player" for other named players, and "" (Unknown) when the
set doesn't say who played. Players who asked not to be in AI training (nametags.BANNED_NAMES) carry
that name instead, so the loader's name filter drops their side of the game, as it would upstream.
meta-pro.json and meta-misc.json split meta.json by whether a pro is named in the game's set.

Raw.json records finished archives, so a rerun resumes. Each archive's work runs in its own systemd
scope (MemoryMax) at nice 19 / ionice idle. Archives are untrusted: only 7z touches them, and every
entry is checked before extraction.

Usage:
  .venv/bin/python -I scripts/build_tournament_dataset.py                 # every archive
  .venv/bin/python -I scripts/build_tournament_dataset.py --only gx3.7z   # chosen archives
  .venv/bin/python -I scripts/build_tournament_dataset.py report          # counts so far
"""

import argparse
import base64
import collections
import concurrent.futures
import contextlib
import dataclasses
import http.client
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path("/home/kage/drive2/ShineBot/data")
SHARE_FILE = Path("/home/kage/smashbot_workspace/slippi-dumps.txt")
BADGER_APP_ID = "5cbed6ac-a083-4e14-b191-b4ba07653de2"
SKIP = {
    "bobc-6-main-stream.7z": "duplicates bobc-6/Main Stream",
    "personal-pre-slippi-online.zip": "the owner's personal games",
}
EVENT_PLAYERS = {"the-match-zain-vs-cody": ("Zain", "Cody")}
MIN_FREE_BYTES = 60e9
MEMORY_MAX = "16G"

PROS = {  # HAL MDS's 38 pros: slug -> name forms seen in brackets
    "cody": ["cody", "codyschwab", "ibdw"], "mang0": ["mang0", "mango"], "solobattle": ["solobattle"],
    "desertsnoopy": ["desertsnoopy"], "jahridin": ["jahridin"], "amsa": ["amsa"], "gosu": ["gosu"],
    "cookbook": ["cookbook"], "ginger": ["ginger"], "frenzy": ["frenzy"], "aklo": ["aklo"],
    "monotheon": ["monotheon"], "siddward": ["siddward"], "franz": ["franz"], "iliketurtles": ["iliketurtles"],
    "trif": ["trif"], "krudo": ["krudo"], "zain": ["zain"], "kodorin": ["kodorin"], "friend": ["friend"],
    "daniel": ["daniel"], "m2k": ["m2k", "mew2king"], "ycz": ["ycz"], "uhhei": ["uhhei"], "grab2win": ["grab2win"],
    "isdsar": ["isdsar"], "fknsilver": ["fknsilver", "silver"], "technospider": ["technospider"],
    "jchu": ["jchu"], "bobbybigballz": ["bobbybigballz"], "nicki": ["nicki"], "kjh": ["kjh"], "axe": ["axe"],
    "redx": ["redx"], "mof": ["mof"], "billybopeep": ["billybopeep"], "rapm": ["rapm", "rapmonster"],
    "druggedfox": ["druggedfox"],
}
RANK_2025 = """Zain|Cody Schwab|Hungrybox|Joshman|moky|Axe|Jmook|SDJ|Wizzrobe|lloD|Trif|Nicki|Soonsay|Aklo|Salt|Krudo|
Aura|Ginger|Junebug|Ossify|Magi|MOF|aMSa|Panda|Zamu|n0ne|SluG|RapMonster|Khryke|404Cray|JChu|Ben|Daytripper|Chem|
Spark|Zuppy|OkayP.|Medz|Morsecode762|Sirmeris|Epoodle|Preeminent|Maelstrom|Zanya|Smash Papi|Kevbot|Chickenman400|
Dawson|Faust|Slowking|Boyd|Graves|Kacey|Seven|BING|Rikzz|Inky|mvlvchi|Foxy Grandpa|DarkGenex|Joey Donuts|Frostbyte|
Mot$|Lowercase hero|Drephen|lexor|max|eve|Holiday|Goodie|Agent|Maher|Zasa|Jude|Grab|Freezus|Fudge|OG Kid|Gahtzu|
coffee|Faith|bonn|Juicebox|TheRealThing|Kalvar|Ober|M1sf1re|mgmg|Stiv|Pleeba|fitzy|Jah Ridin'|JoJo|Fable|Daniel|
Balloon Day|bonfire10|dz|Adwan|Louis"""
RANK_SUMMER_2026 = """Zain|Wizzrobe|Hungrybox|moky|Joshman|Salt|RapMonster|Jmook|Krudo|Aura|SluG|Soonsay|SDJ|Magi|Ginger|Aklo|
Spark|Preeminent|KoDoRiN|Axe|The Weapon|Fiction|DarkGenex|n0ne|Kacey|Zasa|404Cray|Dial M|Ultra|Ben"""
LABELS = {"pro": "GM Player", "GM": "GM Player", "Master": "Master Player"}

CHAR_WORDS = (r"(?:fox|falco|marth|sheik|puff|jigglypuff|jiggs|falcon|captain falcon|cf|peach|ics|ic|ice climbers|"
              r"popo|pikachu|pika|samus|luigi|yoshi|doc|dr\.? mario|mario|ganon|ganondorf|link|young link|yl|ylink|"
              r"ness|kirby|zelda|roy|mewtwo|m2|g&w|gnw|gw|game ?& ?watch|mr\.? game ?& ?watch|pichu|dk|donkey kong|"
              r"bowser)")
TAGGED_CHARACTER = re.compile(rf"^(\([^)]*\)\s*|[^()]{{1,4}}\s+)?{CHAR_WORDS}(\s*\([^)]*\))?$", re.I)
ROUND_WORDS = re.compile(r"winners|losers|grand|final|semi|quarter|pools|round|top \d+|bracket", re.I)


def norm(name: str) -> str:
    name = re.split(r"\||\s{2,}", name.strip())[-1]  # a sponsor prefix; folder names lose the '|'
    return re.sub(r"[^a-z0-9$]", "", name.lower())


def _character_ids() -> dict[str, frozenset[int]]:
    from melee import Character as C

    groups = {
        (C.FOX,): "fox", (C.FALCO,): "falco", (C.MARTH,): "marth", (C.SHEIK, C.ZELDA): "sheik zelda",
        (C.JIGGLYPUFF,): "puff jigglypuff jiggs jiggly", (C.CPTFALCON,): "falcon captainfalcon cf cptfalcon",
        (C.PEACH,): "peach", (C.POPO, C.NANA): "ics ic iceclimbers icies popo nana",
        (C.PIKACHU,): "pikachu pika", (C.SAMUS,): "samus", (C.LUIGI,): "luigi", (C.YOSHI,): "yoshi",
        (C.DOC,): "doc drmario", (C.MARIO,): "mario", (C.GANONDORF,): "ganon ganondorf", (C.LINK,): "link",
        (C.YLINK,): "younglink yl ylink", (C.NESS,): "ness", (C.KIRBY,): "kirby", (C.ROY,): "roy",
        (C.MEWTWO,): "mewtwo m2", (C.PICHU,): "pichu", (C.DK,): "dk donkeykong", (C.BOWSER,): "bowser",
        (C.GAMEANDWATCH,): "gw gnw gaw gameandwatch gamewatch mrgameandwatch mrgamewatch",
    }
    return {word: frozenset(c.value for c in chars) for chars, words in groups.items() for word in words.split()}


CHARACTER_IDS = _character_ids()
PRO_SLUGS = {form: slug for slug, forms in PROS.items() for form in forms}
TOP = {norm(n) for n in (RANK_2025 + "|" + RANK_SUMMER_2026).replace("\n", "").split("|")} | {"jahridin"}


def _opt_outs() -> dict[str, str]:
    from slippi_ai import nametags

    out = {norm(alias): group[0] for group in nametags.NAME_GROUPS if group[0] in nametags.BANNED_NAMES
           for alias in group}
    for slug, forms in PROS.items():
        if slug in out:
            out.update((form, out[slug]) for form in forms)
    return out


OPT_OUT = _opt_outs()


def kind(name: str) -> str:
    key = norm(name)
    if key in OPT_OUT:
        return "opt-out"
    if key in PRO_SLUGS:
        return "pro"
    if key in TOP:
        return "GM"
    return "Master"


@dataclasses.dataclass(frozen=True)
class Side:
    name: str
    characters: frozenset[int] = frozenset()


def character_ids(text: str) -> frozenset[int]:
    """Character ids named in a list like 'ICs, Fox'; words that aren't characters (colours, tags) add none."""
    ids = set()
    for word in re.split(r"[,/]", text):
        ids |= CHARACTER_IDS.get(norm(word), frozenset())
    return frozenset(ids)


def _side(text: str) -> Side:
    name = re.sub(r"\s*[\(\[].*?[\)\]]", "", text).strip(" _-")
    chars = frozenset().union(*(character_ids(g) for g in re.findall(r"\(([^()]*)\)", text)))
    return Side(name, chars)


def _is_doubles(text: str) -> bool:
    text = re.sub(r"\([^)]*\)|game\s*&\s*watch", "", text, flags=re.I)
    return bool(re.search(r"\s[+&_]\s", text))


def parse_set(part: str) -> tuple[str | None, Side, Side] | None:
    """(round, side, side) from one path component naming a set.

    Handles 'WQF - A (Fox) vs B (Ganon)', '1. A vs B' and 'A vs B 2'. A renamed replay that names
    characters and in-game tags ('(PH8) Marth vs Link (MILO)', 'BOY Falco (Default) vs Marth') names
    no players, and neither do doubles teams.
    """
    part = re.sub(r"\.(slp|zip)$", "", part, flags=re.I)
    if not re.search(r"\svs\.?\s", part, re.I):
        return None
    left, right = re.split(r"\s+vs\.?\s+", part, maxsplit=1, flags=re.I)
    round_ = None
    if m := re.match(r"^(.*)\s-\s(.*)$", left):
        round_, left = m.group(1).strip() or None, m.group(2)
    left = re.sub(r"^\d+[.)_]\s*", "", left)
    right = re.split(r"\s-\s|\s\[|\sGame_", right)[0]
    right = re.sub(r"\s+\d+$", "", right)
    if _is_doubles(left) or _is_doubles(right):
        return None
    if TAGGED_CHARACTER.match(left.strip()) and TAGGED_CHARACTER.match(right.strip()):
        return None
    a, b = _side(left), _side(right)
    if not norm(a.name) or not norm(b.name):
        return None
    return round_, a, b


def parse_game_file(base: str) -> tuple[Side, Side] | None:
    """The two players of a file named like '2 - Maybe (ICs), C. Row (Fox) - Pokémon Stadium.slp'."""
    m = re.match(r"^(?:\d+\s*-\s*)?([^(),]+\([^()]*\)),\s*([^(),]+\([^()]*\))\s+-\s+[^()]+\.slp$", base, re.I)
    if not m:
        return None
    a, b = _side(m.group(1)), _side(m.group(2))
    if not (a.characters and b.characters and norm(a.name) and norm(b.name)):
        return None
    return a, b


def find_set(event: str, path: str) -> tuple[str | None, list[Side]] | None:
    """(round, [side, side]) for the set a replay at `path` belongs to, or None when nothing names it."""
    parts = path.split("/")
    if any(re.search(r"\b(doubles|dubs|teams)\b", p, re.I) for p in parts[:-1]):
        return None
    if any(re.search(r"\svs\.?\s", p, re.I) and _is_doubles(p) for p in parts):
        return None
    for i in range(len(parts) - 1, -1, -1):
        if found := parse_set(parts[i]):
            round_, a, b = found
            for parent in reversed(parts[:i] if round_ is None else []):
                if ROUND_WORDS.search(parent) and not re.search(r"\svs\.?\s", parent, re.I):
                    round_ = re.sub(r"^\d+\s*-\s*", "", parent)
                    break
            return round_, [a, b]
    if found := parse_game_file(parts[-1]):
        return None, list(found)
    if event in EVENT_PLAYERS:
        return None, [Side(n) for n in EVENT_PLAYERS[event]]
    return None


def game_number(path: str) -> int | None:
    base = path.rsplit("/", 1)[-1]
    m = re.match(r"^(\d+)\s*-\s", base) or re.match(r"^game\s*(\d+)\b", base, re.I)
    return int(m.group(1)) if m else None


def context_ports(ctx: dict, number: int | None) -> dict[int, str] | None:
    """Port (0-indexed, as parsed) -> display name for one game of a set's context.json."""
    def ports_of(score) -> dict[int, str] | None:
        out = {}
        for slot in (score or {}).get("slots") or []:
            names, ports = slot.get("displayNames") or [], slot.get("ports") or []
            if len(names) != 1 or len(ports) != 1:
                return None
            out[int(ports[0]) - 1] = str(names[0])
        return out if len(out) == 2 else None

    scores = ctx.get("scores") or []
    if number is not None and 0 < number <= len(scores):
        return ports_of(scores[number - 1])
    maps = [ports_of(s) for s in [*scores, ctx.get("finalScore")]]
    return maps[0] if maps and maps[0] and all(m == maps[0] for m in maps) else None


def context_sides(ctx: dict) -> list[Side]:
    sides = []
    for key in ("entrant1", "entrant2"):
        members = (ctx.get("players") or {}).get(key) or []
        if len(members) != 1:
            return []
        chars = frozenset().union(*(character_ids(str(c)) for c in members[0].get("characters") or []))
        sides.append(Side(str(members[0].get("name") or ""), chars))
    return sides if all(norm(s.name) for s in sides) else []


def context_round(ctx: dict) -> str | None:
    startgg = ctx.get("startgg") or {}
    words = [(startgg.get("phase") or {}).get("name"), (startgg.get("set") or {}).get("fullRoundText")]
    return " / ".join(str(w) for w in words if w) or None


def match_characters(sides: list[Side], characters: list[int]) -> list[Side] | None:
    """The sides in port order when their characters pick exactly one assignment, else None."""
    def fits(side: Side, character: int) -> bool:
        return not side.characters or character in side.characters

    orders = [o for o in ((0, 1), (1, 0)) if all(fits(sides[s], c) for s, c in zip(o, characters))]
    return [sides[s] for s in orders[0]] if len(orders) == 1 else None


def context_info(ctx: dict, path: str) -> tuple[str | None, list[Side], dict[int, str] | None]:
    """(round, entrants, port -> name for this game) from a context.json, empty where it doesn't say."""
    try:
        return context_round(ctx), context_sides(ctx), context_ports(ctx, game_number(path))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None, [], None


def own_name(player: dict) -> str:
    """nametags.name_from_metadata for a parse result, whose netplay may be an empty dict."""
    netplay = player.get("netplay") or {}
    return (netplay.get("code") or "").replace("＃", "#") or netplay.get("name") or player.get("name_tag") or ""


def label_game(event: str, path: str, players: list[dict], ctx: dict | None) -> tuple[list[str], list, dict]:
    """Labels (netplay names) and identities per port, and the game's provenance record."""
    from slippi_ai import nametags

    set_info = find_set(event, path)
    round_ = set_info[0] if set_info else None
    named: list[Side] = []
    by_port: list[Side] | None = None
    method = None
    if ctx is not None:
        ctx_round, sides, ports = context_info(ctx, path)
        round_ = ctx_round or round_
        if ports and set(ports) == {p["port"] for p in players}:
            chars = {norm(s.name): s.characters for s in sides}
            by_port = [Side(ports[p["port"]], chars.get(norm(ports[p["port"]]), frozenset())) for p in players]
            named, method = by_port, "context"
        else:
            named = sides
    if not named and set_info:
        named = set_info[1]
    if named and by_port is None:
        by_port = match_characters(named, [p["character"] for p in players])
        method = "characters" if by_port else "unresolved"

    kinds = [kind(s.name) for s in named]
    if by_port:
        identities = [s.name for s in by_port]
        labels = [OPT_OUT[norm(s.name)] if kind(s.name) == "opt-out" else LABELS[kind(s.name)] for s in by_port]
    else:
        identities = [None, None]
        opted = [OPT_OUT[norm(s.name)] for s in named if kind(s.name) == "opt-out"]
        if opted:
            labels = [opted[0]] * 2
        elif named and LABELS[kinds[0]] == LABELS[kinds[1]]:
            labels = [LABELS[kinds[0]]] * 2
            method += " same-label"
        else:
            labels = ["", ""]

    for i, player in enumerate(players):
        own = own_name(player)
        if own and (nametags.is_banned_name(own) or norm(own) in OPT_OUT):
            labels[i] = OPT_OUT.get(norm(own)) or nametags.normalize_name(own)
            identities[i] = identities[i] or own

    provenance = {
        "event": event,
        "path": path,
        "group": "pro" if "pro" in kinds else "misc",
        "round": round_,
        "players": [s.name for s in named],
        "assignment": method,
    }
    return labels, identities, provenance


def _init_worker() -> None:
    import pyarrow

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


def file_md5(path: str) -> str:
    from slippi_db import utils

    with open(path, "rb") as f:
        return utils.md5(f.read())


LEGAL_STAGES = {"FINAL_DESTINATION", "BATTLEFIELD", "DREAMLAND", "FOUNTAIN_OF_DREAMS", "YOSHIS_STORY",
                "POKEMON_STADIUM"}


def quality_problem(game, metadata: dict) -> str | None:
    """HAL's quality filter, from the peppi frames; None when the game passes."""
    from melee import enums

    if enums.to_internal_stage(metadata["stage"]).name not in LEGAL_STAGES:
        return "HAL: not one of the six legal stages"
    if len(game.frames.id) < 1500:
        return "HAL: under 1500 frames"
    deaths = cheap = 0
    for port in game.frames.ports:
        stocks = port.leader.post.stocks.to_numpy(zero_copy_only=False).astype(np.int64)
        percent = port.leader.post.percent.to_numpy(zero_copy_only=False)
        if stocks[0] != 4:
            return "HAL: not a 4-stock start"
        lost = np.nonzero(np.diff(stocks) < 0)[0]
        deaths += int((stocks[lost] - stocks[lost + 1]).sum())
        cheap += int((percent[lost] <= 10).sum())
    if deaths < 3:
        return "HAL: under 3 deaths"
    damage = [p["damage_taken"] for p in metadata["players"]]
    if min(damage) <= 50:
        return "HAL: a player dealt 50 damage or less"
    if sum(damage) < 100:
        return "HAL: under 100 damage in total"
    if 2 * cheap > deaths:
        return "HAL: most deaths at 10% or less"
    return None


def without_metadata(slp_bytes: bytes) -> bytes:
    """The replay with an empty metadata block and its frames untouched.

    For metadata peppi can't decode: Nintendont writes its console nickname from uninitialised memory,
    cut mid-UTF-8 sequence.
    """
    header = b"{U\x03raw[$U#l"
    if not slp_bytes.startswith(header):
        raise ValueError("not a .slp with a sized raw array")
    end = len(header) + 4 + int.from_bytes(slp_bytes[len(header):len(header) + 4], "big")
    return slp_bytes[:end] + b"U\x08metadata{}}"


def parse_replay(task: tuple[str, str, str, str]) -> dict:
    """parse_local.parse_slp's steps for a replay on disk, with HAL's filter before the parquet is written."""
    from slippi_db import parse_peppi, parsing_utils, preprocessing, utils
    from slippi_db.parsing_utils import CompressionType

    path, name, raw, parsed_dir = task
    try:
        with open(path, "rb") as f:
            slp_bytes = f.read()
        md5 = utils.md5(slp_bytes)
        result = dict(name=name, raw=raw, slp_md5=md5, slp_size=len(slp_bytes))
        try:
            game = parse_peppi.read_slippi(path)
        except OSError as e:
            if "invalid UTF8" not in str(e):
                raise
            with tempfile.NamedTemporaryFile(dir=utils.get_tmp_dir(in_memory=True), suffix=".slp") as tmp:
                tmp.write(without_metadata(slp_bytes))
                tmp.flush()
                game = parse_peppi.read_slippi(tmp.name)
            result["metadata_dropped"] = True
        del slp_bytes
        metadata = preprocessing.get_metadata(game)
        is_training, reason = preprocessing.is_training_replay(metadata)
        result.update(metadata, valid=True, is_training=is_training, not_training_reason=reason)
        if not is_training:
            return result
        result["quality"] = quality_problem(game, metadata)
        if result["quality"]:
            return result
        arrays = parse_peppi.from_peppi(game)
        data_error = preprocessing.check_game_data(arrays)
        game_bytes = parsing_utils.convert_game(arrays, compression=CompressionType.ZLIB)
        result.update(data_ok=data_error is None, data_error=data_error, pq_size=len(game_bytes),
                      compression=CompressionType.ZLIB.value)
        out = os.path.join(parsed_dir, md5)
        with open(out + ".tmp", "wb") as f:
            f.write(game_bytes)
        os.replace(out + ".tmp", out)
        return result
    except KeyboardInterrupt:
        raise
    except BaseException as e:  # peppi's Rust panics are BaseExceptions, as parse_slp_safe knows
        return dict(name=name, raw=raw, valid=False, reason=repr(e))


def to_row(result: dict, labels: list[str]) -> dict:
    """A meta.json row (parsed.pkl's shape) with each player's name replaced by its label."""
    from slippi_db import parse_local

    if result.get("players"):
        result = dict(result, players=[
            dict(p, name_tag="", netplay={"name": label, "code": "", "suid": ""})
            for p, label in zip(result["players"], labels)])
    return parse_local.sqlite_row_to_dict(dict(zip(parse_local.PARSED_COLUMNS, parse_local.result_to_tuple(result))))


def check_listing(archive: Path) -> int:
    """Total unpacked bytes, after refusing any entry that is a link or could land outside the target."""
    out = subprocess.run(["7z", "l", "-slt", str(archive)], capture_output=True, text=True, check=True).stdout
    total = 0
    for block in out.partition("----------")[2].strip().split("\n\n"):
        kv = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        path = kv.get("Path")
        if path is None:
            continue
        parts = path.replace("\\", "/").split("/")
        if path.startswith(("/", "\\")) or ".." in parts or re.search(r"\bl[rwx-]{9}\b", kv.get("Attributes", "")):
            raise ValueError(f"unsafe entry in {archive.name}: {path!r} {kv.get('Attributes')!r}")
        total += int(kv.get("Size") or 0)
    return total


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    res = subprocess.run(["7z", "x", "-y", "-bd", f"-o{dest}", str(archive)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode > 2:
        raise RuntimeError(f"7z failed on {archive.name} ({res.returncode}): {res.stderr[-500:]}")
    if res.returncode:
        print(f"  7z {archive.name}: exit {res.returncode}, some files skipped: {res.stderr[-500:]}", flush=True)


def walk(root: Path, prefix: str = "") -> list[tuple[str, Path]]:
    """Regular files under root as (path relative to root, path on disk); links are never followed."""
    out = []
    with os.scandir(root) as it:
        for entry in it:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                out += walk(Path(entry.path), f"{prefix}{entry.name}/")
            elif entry.is_file(follow_symlinks=False):
                out.append((prefix + entry.name, Path(entry.path)))
    return out


def unpack(archive: Path, dest: Path) -> list[tuple[str, Path]]:
    """Every file in the archive and in the zips inside it, as (path in archive, path on disk)."""
    extract(archive, dest / "outer")
    files = []
    for i, (path, disk) in enumerate(sorted(walk(dest / "outer"))):
        if path.lower().endswith(".zip"):
            check_listing(disk)
            extract(disk, dest / "nested" / str(i))
            files += [(f"{path}/{inner}", d) for inner, d in walk(dest / "nested" / str(i))]
        else:
            files.append((path, disk))
    return sorted(files)


def load_context(path: Path) -> dict | None:
    if path.stat().st_size > 1 << 20:
        return None
    try:
        ctx = json.loads(path.read_bytes())
    except ValueError:
        return None
    return ctx if isinstance(ctx, dict) else None


def find_context(contexts: dict[str, Path], path: str) -> dict | None:
    folder = os.path.dirname(path)
    for key in (folder, re.sub(r"\.zip$", "", folder, flags=re.I)):
        if key in contexts:
            return load_context(contexts[key])
    return None


SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    raw TEXT NOT NULL, name TEXT NOT NULL, slp_md5 TEXT, outcome TEXT NOT NULL, PRIMARY KEY (raw, name));
CREATE TABLE IF NOT EXISTS games (
    slp_md5 TEXT PRIMARY KEY, raw TEXT NOT NULL, name TEXT NOT NULL, grp TEXT NOT NULL, match_key TEXT,
    row TEXT NOT NULL);
"""


def open_db(root: Path) -> sqlite3.Connection:
    from slippi_db import parse_local

    conn = sqlite3.connect(root / "parsed.sqlite")
    conn.executescript(parse_local.PARSED_SCHEMA + SCHEMA)
    return conn


def match_key(row: dict) -> str | None:
    match = row.get("match") or {}
    return f"{match['id']}|{match.get('game')}|{match.get('tiebreaker')}" if match.get("id") else None


def drop_reason(result: dict, row: dict, check_replay, match_keys: set[str]) -> str | None:
    """The first filter a parsed replay fails, or None when it counts."""
    if not result["valid"]:
        return "parse error"
    if not result["is_training"]:
        return f"pipeline: {result['not_training_reason']}"
    if result.get("quality"):
        return result["quality"]
    if reason := check_replay(row):
        return f"pipeline: {reason}"
    if (key := match_key(row)) and key in match_keys:
        return "pipeline: duplicate match ID"
    return None


def labelled_row(result: dict, event: str, ctx: dict | None) -> dict:
    """The meta.json row for a kept game: labels for names, the originals and provenance beside them."""
    labels, identities, provenance = label_game(event, result["name"], result["players"], ctx)
    row = to_row(result, labels)
    for out, orig, identity in zip(row["players"], result["players"], identities):
        out["identity"] = identity
        out["orig_name_tag"] = orig["name_tag"]
        if any((orig.get("netplay") or {}).values()):
            out["orig_netplay"] = orig["netplay"]
    if result.get("metadata_dropped"):
        provenance["metadata_dropped"] = True
    row["tournament"] = provenance
    return row


def process(archive: Path, root: Path, work: Path, existing_md5s: Path, workers: int) -> dict:
    """Unpack, parse, filter and label one archive, and record it in root. Returns its outcome counts."""
    from slippi_db import parse_local
    from slippi_db.scripts.make_local_dataset import check_replay

    raw, event = archive.name, re.sub(r"\.(7z|zip)$", "", archive.name)
    unpacked = check_listing(archive)
    free = shutil.disk_usage(work).free
    if free - unpacked < MIN_FREE_BYTES:
        print(f"PAUSE: {raw} unpacks to {unpacked / 1e9:.1f} GB with {free / 1e9:.1f} GB free", flush=True)
        sys.exit(3)
    dest = work / "x" / event
    shutil.rmtree(dest, ignore_errors=True)
    parsed_dir = root / "Parsed"
    t0 = time.time()
    try:
        files = unpack(archive, dest)
        t1 = time.time()
        contexts = {os.path.dirname(p): d for p, d in files if os.path.basename(p) == "context.json"}
        replays = [(p, d) for p, d in files if p.lower().endswith(".slp")
                   and not os.path.basename(p).startswith("._") and "__MACOSX/" not in p]

        conn = open_db(root)
        existing = set(existing_md5s.read_text().split())
        earlier = {m for (m,) in conn.execute(
            "SELECT slp_md5 FROM replays WHERE raw != ? AND slp_md5 IS NOT NULL", (raw,))}
        match_keys = {k for (k,) in conn.execute(
            "SELECT match_key FROM games WHERE raw != ? AND match_key IS NOT NULL", (raw,))}

        outcomes: dict[str, tuple[str | None, str]] = {}
        with concurrent.futures.ProcessPoolExecutor(workers, initializer=_init_worker) as pool:
            md5s = list(pool.map(file_md5, [str(d) for _, d in replays], chunksize=16))
            tasks, here = [], set()
            for (path, disk), md5 in zip(replays, md5s):
                if md5 in existing:
                    outcomes[path] = (md5, "duplicate md5 (existing dataset)")
                elif md5 in earlier:
                    outcomes[path] = (md5, "duplicate md5 (earlier archive)")
                elif md5 in here:
                    outcomes[path] = (md5, "duplicate md5 (same archive)")
                else:
                    here.add(md5)
                    tasks.append((str(disk), path, raw, str(parsed_dir)))
            results = []
            for result in pool.map(parse_replay, tasks, chunksize=2):
                results.append(result)
                if len(results) % 1000 == 0:
                    print(f"  {raw}: parsed {len(results)}/{len(tasks)} in {time.time() - t1:.0f}s", flush=True)
        t2 = time.time()

        replay_rows, games = [], []
        for result in results:
            path = result["name"]
            row = to_row(result, ["", ""]) if result["valid"] else result
            outcome = drop_reason(result, row, check_replay, match_keys) or "kept"
            if outcome == "kept":
                row = labelled_row(result, event, find_context(contexts, path))
                if key := match_key(row):
                    match_keys.add(key)
                games.append((row["slp_md5"], raw, path, row["tournament"]["group"], key, json.dumps(row)))
            elif result.get("pq_size"):
                Path(parsed_dir, result["slp_md5"]).unlink(missing_ok=True)
            outcomes[path] = (result.get("slp_md5"), outcome)
            replay_rows.append(parse_local.result_to_tuple(row))

        with conn:
            for table in ("replays", "outcomes", "games"):
                conn.execute(f"DELETE FROM {table} WHERE raw = ?", (raw,))
            conn.executemany(parse_local.PARSED_INSERT_SQL, replay_rows)
            conn.executemany("INSERT INTO outcomes VALUES (?, ?, ?, ?)",
                             [(raw, p, m, o) for p, (m, o) in outcomes.items()])
            conn.executemany("INSERT INTO games VALUES (?, ?, ?, ?, ?, ?)", games)
        mark_processed(root, raw)
        write_meta(root, conn)
        conn.close()
    finally:
        shutil.rmtree(dest, ignore_errors=True)

    counts = collections.Counter(o for _, o in outcomes.values())
    summary = {"archive": raw, "files": len(replays), "kept": counts["kept"],
               "pro": sum(g[3] == "pro" for g in games), "context.json": len(contexts),
               "unpack_s": round(t1 - t0), "parse_s": round(t2 - t1), "outcomes": dict(counts.most_common())}
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


def mark_processed(root: Path, raw: str) -> None:
    path = root / "raw.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    if raw not in {r["name"] for r in rows}:
        rows.append({"processed": True, "name": raw})
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2))
    os.replace(tmp, path)


def processed(root: Path) -> set[str]:
    path = root / "raw.json"
    return {r["name"] for r in json.loads(path.read_text()) if r.get("processed")} if path.exists() else set()


def write_meta(root: Path, conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT grp, row FROM games ORDER BY raw, name").fetchall()
    for name, group in (("meta.json", None), ("meta-pro.json", "pro"), ("meta-misc.json", "misc")):
        tmp = root / f"{name}.tmp"
        tmp.write_text("[" + ", ".join(row for grp, row in rows if group in (None, grp)) + "]")
        os.replace(tmp, root / name)


def finalize(root: Path) -> None:
    """parsed.pkl from parsed.sqlite, and drop parquet files no kept game points at (left by a killed run)."""
    from slippi_db import parse_local

    parse_local.convert_sqlite_to_pickle(str(root / "parsed.sqlite"), str(root / "parsed.pkl"))
    conn = open_db(root)
    kept = {m for (m,) in conn.execute("SELECT slp_md5 FROM games")}
    conn.close()
    stray = [p for p in (root / "Parsed").iterdir() if p.name not in kept]
    for p in stray:
        p.unlink()
    print(f"finalize: parsed.pkl written; removed {len(stray)} stray parquet files", flush=True)


def share_url() -> str:
    return SHARE_FILE.read_text().split()[0]


def list_archives() -> dict[str, dict]:
    """name -> {size, url} for every archive in the share, via OneDrive's anonymous "badger" token."""
    req = urllib.request.Request("https://api-badgerp.svc.ms/v1.0/token",
                                 data=json.dumps({"appId": BADGER_APP_ID}).encode(),
                                 headers={"Content-Type": "application/json"})
    token = json.load(urllib.request.urlopen(req, timeout=60))["token"]
    share = "u!" + base64.urlsafe_b64encode(share_url().encode()).decode().rstrip("=")
    url = f"https://my.microsoftpersonalcontent.com/_api/v2.0/shares/{share}/driveitem?$expand=children"
    headers = {"Authorization": f"Badger {token}", "Prefer": "autoredeem"}
    item = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120))
    children = item["children"]
    next_link = item.get("children@odata.nextLink")
    while next_link:
        page = json.load(urllib.request.urlopen(urllib.request.Request(next_link, headers=headers), timeout=120))
        children += page["value"]
        next_link = page.get("@odata.nextLink")
    return {c["name"]: {"size": c["size"], "url": c["@content.downloadUrl"]} for c in children
            if c["name"].lower().endswith((".7z", ".zip"))}


DOWNLOAD_CHUNK = 64 << 20


def download(name: str, size: int, dest: Path, connections: int = 4, attempts: int = 8) -> Path:
    """Fetch one archive as ranged chunks over a few connections (OneDrive throttles a long-lived one).

    <name>.chunks lists the chunks already written, so an interrupted download refetches only the rest.
    """
    out, part, chunks = dest / name, dest / (name + ".part"), dest / (name + ".chunks")
    if chunks.exists():
        done = {int(i) for i in chunks.read_text().split()}
    else:
        have = part.stat().st_size if part.exists() else 0
        done = set(range(min(have, size) // DOWNLOAD_CHUNK))
    with open(part, "ab") as f:
        f.truncate(size)
    url = [list_archives()[name]["url"]]
    lock = threading.Lock()

    def fetch(i: int) -> None:
        start, end = i * DOWNLOAD_CHUNK, min(size, (i + 1) * DOWNLOAD_CHUNK) - 1
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(url[0], headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = resp.read() if resp.status == 206 else b""
                if len(data) != end - start + 1:
                    raise ValueError(f"chunk {i}: {len(data)} bytes, status {resp.status}")
                fd = os.open(part, os.O_WRONLY)
                try:
                    os.pwrite(fd, data, start)
                finally:
                    os.close(fd)
                with lock, open(chunks, "a") as f:
                    f.write(f"{i}\n")
                return
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
                print(f"  download {name}: chunk {i} attempt {attempt + 1} failed: {e!r}", flush=True)
                time.sleep(15 * (attempt + 1))
                with lock, contextlib.suppress(urllib.error.URLError, http.client.HTTPException, OSError):
                    url[0] = list_archives()[name]["url"]
        raise RuntimeError(f"download {name}: chunk {i} failed {attempts} times")

    todo = [i for i in range(-(-size // DOWNLOAD_CHUNK)) if i not in done]
    with concurrent.futures.ThreadPoolExecutor(connections) as pool:
        list(pool.map(fetch, todo))
    part.rename(out)
    chunks.unlink(missing_ok=True)
    return out


def write_existing_md5s(meta: Path, out: Path) -> None:
    """The slp_md5 of every row in an existing meta.json, scanned without loading the whole file."""
    pattern = re.compile(rb'"slp_md5": "([0-9a-f]{32})"')
    found, tail = set(), b""
    with open(meta, "rb") as f:
        while chunk := f.read(1 << 26):
            buf = tail + chunk
            found.update(m.group(1).decode() for m in pattern.finditer(buf))
            tail = buf[-64:]
    out.write_text("\n".join(sorted(found)) + "\n")
    print(f"{len(found)} md5s from {meta}", flush=True)


def scoped(args: list[str]) -> int:
    """Run this script's subcommand in its own memory-capped systemd scope at the lowest priority."""
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={MEMORY_MAX}", "--",
           "env", "OMP_NUM_THREADS=1", "RAYON_NUM_THREADS=1",
           "nice", "-n", "19", "ionice", "-c3", sys.executable, "-I", str(Path(__file__).resolve()), *args]
    proc = subprocess.Popen(cmd)
    print(f"  pid {proc.pid}: {' '.join(args[:3])}", flush=True)
    return proc.wait()


def run(args: argparse.Namespace) -> int:
    root, work = args.root, args.work
    (root / "Parsed").mkdir(parents=True, exist_ok=True)
    (work / "dl").mkdir(parents=True, exist_ok=True)
    existing = work / "existing-md5.txt"
    if not existing.exists():
        write_existing_md5s(args.existing_meta, existing)

    archives = list_archives()
    done = processed(root)
    todo = [n for n in sorted(archives) if n not in done and n not in SKIP and (not args.only or n in args.only)]
    print(f"{len(archives)} archives; {len(done)} done; {len(todo)} to go; skipping {sorted(SKIP)}", flush=True)
    failed = []
    start = time.time()
    for i, name in enumerate(todo):
        size = archives[name]["size"]
        free = shutil.disk_usage(work).free
        if free - 9 * size < MIN_FREE_BYTES:
            print(f"PAUSE: {free / 1e9:.0f} GB free, {name} needs ~{9 * size / 1e9:.0f} GB", flush=True)
            return 3
        t = time.time()
        print(f"[{i + 1}/{len(todo)}] {time.strftime('%H:%M')} {name} ({size / 1e9:.2f} GB)", flush=True)
        path = download(name, size, work / "dl")
        print(f"  downloaded in {time.time() - t:.0f}s", flush=True)
        rc = scoped(["process", "--archive", str(path), "--root", str(root), "--work", str(work),
                     "--workers", str(args.workers)])
        path.unlink()
        if rc == 3:
            return 3
        if rc:
            failed.append(name)
            print(f"  FAILED {name} (exit {rc}); a rerun retries it", flush=True)
        print(f"  {name} done in {time.time() - t:.0f}s; {time.time() - start:.0f}s total; "
              f"{shutil.disk_usage(work).free / 1e9:.0f} GB free", flush=True)
    if not args.only:
        scoped(["finalize", "--root", str(root)])
    print(f"DONE in {time.time() - start:.0f}s; failed: {failed}", flush=True)
    return 1 if failed else 0


def report(root: Path) -> dict:
    """Counts over everything recorded so far: outcomes, groups, label slots, pros, sizes."""
    conn = open_db(root)
    outcomes = collections.Counter(o for (o,) in conn.execute("SELECT outcome FROM outcomes"))
    rows = [json.loads(r) for (r,) in conn.execute("SELECT row FROM games")]
    conn.close()
    groups, slots, pros, events, methods = (collections.Counter() for _ in range(5))
    for row in rows:
        prov = row["tournament"]
        groups[prov["group"]] += 1
        events[prov["event"]] += 1
        methods[prov["assignment"]] += 1
        for i, player in enumerate(row["players"]):
            label = player["netplay"]["name"]
            if label == "":
                slots["Unknown"] += 1
            elif label not in ("GM Player", "Master Player"):
                slots["opt-out"] += 1
            else:
                slots[kind(player["identity"] or prov["players"][i])] += 1
        for slug in {PRO_SLUGS[norm(n)] for n in prov["players"] if kind(n) == "pro"}:
            pros[slug] += 1
    sizes = np.array([row["pq_size"] for row in rows]) if rows else np.zeros(1)
    disk = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return {
        "games": len(rows), "groups": dict(groups), "label_slots": dict(slots),
        "games_per_pro": dict(pros.most_common()), "assignment": dict(methods),
        "outcomes": dict(outcomes.most_common()), "games_per_event": dict(events.most_common()),
        "pq_size": {"mean": int(sizes.mean()), "median": int(np.median(sizes)), "max": int(sizes.max())},
        "disk_gb": round(disk / 1e9, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="run", choices=["run", "process", "finalize", "report"])
    ap.add_argument("--root", type=Path, default=DATA / "tournament" / "Root")
    ap.add_argument("--work", type=Path, default=DATA / "tournament" / "work")
    ap.add_argument("--existing_meta", type=Path, default=DATA / "full" / "Root" / "meta.json")
    ap.add_argument("--only", nargs="*", help="archive names to build (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--archive", type=Path, help="process: the downloaded archive")
    args = ap.parse_args()

    if args.command == "run":
        sys.exit(run(args))
    if args.command == "process":
        process(args.archive, args.root, args.work, args.work / "existing-md5.txt", args.workers)
    elif args.command == "finalize":
        finalize(args.root)
    else:
        print(json.dumps(report(args.root), indent=1))


if __name__ == "__main__":
    main()
