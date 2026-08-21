"""The spawned env process: one Dolphin, one libmelee parser, numpy frame
encoding. Torch-free by construction (see smashbot.rl.config / encode) —
every env would otherwise carry torch's ~0.26 GB private RSS.
"""

from __future__ import annotations

import typing as tp

from smashbot.rl.config import (
    CPU_CHARS, OFF_ROSTER, OPPONENT_CHARS, EnvSpec, RolloutConfig,
    student_whitelist,
)


def next_opponent_char(
    cur_kind: str,
    char_lock: str | None,
    redraw_chars: bool,
    armed_char: str | None,
    draw: tp.Callable[[], str],
) -> str | None:
    """Opponent-seat character for the NEXT game (game boundary or Dolphin
    recycle). Returns the character to arm, or None to keep the sitting one.

    Char-locked league members (imports) pin their lock: while locked, NO
    rng draw is consumed (the redraw stream simply pauses) and the seat pins
    the lock; when the lock clears, normal redraws resume. CPU serving
    ignores the lock — locks belong to policy imports only, and an env that
    lazily adopted cpu must draw from the cpu roster. Default path (lock
    None) is byte-identical to the pre-import redraw behavior."""
    if char_lock is not None and cur_kind != "cpu":
        return char_lock if armed_char != char_lock else None
    if redraw_chars:
        return draw()
    return None


def _env_process_main(
    idx: int, cfg: RolloutConfig, spec: EnvSpec, conn, encoder_spec: tuple,
) -> None:
    """One Dolphin per PROCESS (multiple libmelee Consoles cannot share a
    process — the vendor's envs.py reaches the same conclusion). Speaks over
    a Pipe: sends per-frame payloads, receives {port: Controller} commands
    (None = shut down)."""
    import os
    import sys

    # Dolphin banners/spam would hit the parent terminal on every boot and
    # recycle; redirect this process (and its Dolphin child, via fd
    # inheritance) to a per-env log where real errors remain findable.
    _tag = f"-{cfg.log_tag}" if cfg.log_tag else ""
    _log = open(f"/tmp/smashbot-env{_tag}-{idx}.log", "a", buffering=1)
    os.dup2(_log.fileno(), sys.stdout.fileno())
    os.dup2(_log.fileno(), sys.stderr.fileno())

    import faulthandler
    import signal as _sig

    # On SIGUSR1 (sent by the worker watchdog before it gives up), dump this
    # process's exact python stack to the env log — no more guessing where a
    # silent env is stuck.
    faulthandler.register(_sig.SIGUSR1, file=_log, all_threads=True)

    import melee
    import numpy as np
    import tree as tree_lib

    from slippi_ai import controller_lib
    from slippi_ai import dolphin as dolphin_lib
    from slippi_ai.dolphin import WrongCharacterSelected
    from slippi_db.parse_libmelee import Parser

    from smashbot import encode
    from smashbot.eval.dolphin_setup import make_dolphin

    # Encode (from_state: pure numpy typing/bucketing, no NN) worker-side so
    # the 32 env processes parallelize it instead of the main loop. Must match
    # the policy's embed schema — both use the default EmbedConfig (verified
    # by test_worker_side_encode_matches_policy_encode).
    embed_game = encode.build(encoder_spec)  # torch-free numpy encoder

    import random as random_lib

    opp_port = 3 - spec.student_port
    # Per-env deterministic RNG for opponent-character redraws at each
    # Dolphin recycle: matchups rotate over the run instead of being frozen
    # by the boot-time draw (the first Dolphin still uses the partition's
    # stratified char, preserving guaranteed full-roster coverage at boot).
    # Per-env replay subdir: parallel Dolphins start games in the same
    # second, and Slippi names files Game_<timestamp>.slp — a shared dir
    # would collide/overwrite across envs.
    _replay_dir = f"{cfg.replay_dir}/env-{idx}" if cfg.replay_dir else ""

    char_rng = random_lib.Random((cfg.partition_seed << 16) ^ idx)
    # Student-seat whitelist draws use their OWN rng stream so a multi-char
    # whitelist never perturbs the opponent redraw sequence (and vice versa).
    whitelist = student_whitelist(cfg.char_whitelist, cfg.bot_char)
    student_rng = random_lib.Random(((cfg.partition_seed + 0x51EC7) << 16) ^ idx)

    # Kind ACTUALLY serving on the opponent seat. Normally fixed (= spec.kind
    # for the whole run); under league_cpu a snapshot-slot env can be asked
    # (via the "opp_kind" command key) to flip policy<->cpu — Dolphin players
    # are only built at (re)boot and a CPU port cannot hot-swap mid-game, so
    # the flip is adopted LAZILY at the next recycle boundary. Until then the
    # env keeps serving its previous kind and reports what it serves, so
    # attribution follows reality, not the desired assignment.
    cur_kind = spec.kind
    desired_kind = None  # latest "policy"/"cpu" wish from the worker
    # Character lock piggybacked by the worker while this env's slot serves
    # a char-locked import ("opp_char_lock" command key; None = unlocked).
    # While locked, the opponent seat pins this char instead of redrawing
    # per game — see next_opponent_char. Never sent outside league_imports.
    char_lock = None

    def _draw_char() -> str:
        if cur_kind == "cpu":
            # CPU opponents draw 60/40 main12/off-roster; CPU-Sheik is
            # impossible (engine ignores the transform on CPU ports), hence
            # CPU_CHARS
            pool = (CPU_CHARS if char_rng.random() < cfg.main12_prob
                    else OFF_ROSTER)
            return char_rng.choice(pool)
        if cur_kind == "self":  # second student seat: whitelist only
            return char_rng.choice(whitelist)
        return char_rng.choice(OPPONENT_CHARS)

    def _draw_student_char() -> str:
        # len==1: exactly the fixed-character behavior, zero rng draws.
        if len(whitelist) == 1:
            return whitelist[0]
        return student_rng.choice(whitelist)

    def _build_players(opp_char: str, student_char: str) -> dict:
        if cur_kind == "cpu":
            opponent_player = dolphin_lib.CPU(
                character=melee.Character[opp_char.upper()],
                level=spec.cpu_level,
            )
        else:
            opponent_player = dolphin_lib.AI(
                character=melee.Character[opp_char.upper()]
            )
        return {
            spec.student_port: dolphin_lib.AI(
                character=melee.Character[student_char.upper()]
            ),
            opp_port: opponent_player,
        }

    players = _build_players(spec.opponent_char, _draw_student_char())
    # Character actually playing the CURRENT game on the opponent seat.
    # Per-game redraws mutate `players` BETWEEN games and take effect at the
    # NEXT rematch CSS pass, so the char is "armed" one boundary before it
    # plays: cur <- armed at each boundary, then armed <- fresh draw.
    cur_opp_char = spec.opponent_char
    armed_opp_char = spec.opponent_char
    # Carried across Dolphin recycles: the new instance's first frame must
    # still announce the game boundary (fresh recurrent state, zeroed reward)
    # and deliver the final game's result — otherwise two different games
    # would silently splice into one stream.
    pending_reset = False
    pending_result = None
    # serving label ("policy"/"cpu") of the game that produced
    # pending_result: a result carried across a recycle must attribute to
    # the kind that PLAYED it, even if the recycle just adopted a new kind
    pending_result_kind = None
    consecutive_misselects = 0

    # Double buffering: during a Dolphin's LAST game, boot its replacement in
    # a background thread. The spare is only CONSTRUCTED (process up, ISO
    # loaded, idle at intro menus — menus don't advance without inputs, so
    # nothing progresses unattended); menu navigation still happens at swap,
    # via the normal iter_gamestates path with its misselect guard. Hides the
    # 10-15s boot that otherwise stalls the whole worker barrier per recycle.
    import signal
    import threading

    spare = {"dolphin": None, "thread": None}
    old_stops: list = []  # (thread, dolphin_pid) pairs

    def _dolphin_pid(d) -> int | None:
        try:
            return d.console._process.pid
        except AttributeError:
            return None

    def _boot_spare() -> None:
        try:
            spare["dolphin"] = make_dolphin(
                players, headless=cfg.headless, stage=cfg.stage,
                save_replays=cfg.save_replays, replay_dir=_replay_dir,
            )
        except Exception as e:  # fall back to a cold boot at swap time
            print(f"spare boot failed (cold boot at swap): {e}", flush=True)
            spare["dolphin"] = None

    def _start_spare() -> None:
        if cfg.double_buffer and spare["thread"] is None:
            spare["thread"] = threading.Thread(target=_boot_spare, daemon=True)
            spare["thread"].start()

    def _take_spare():
        if spare["thread"] is None:
            return None
        spare["thread"].join(timeout=120)
        if spare["thread"].is_alive():
            # wedged spare boot: abandon it (daemon thread) and cold-boot
            print("WARNING: spare boot wedged; abandoning it", flush=True)
            spare["thread"] = None
            spare["dolphin"] = None
            return None
        d, spare["dolphin"], spare["thread"] = spare["dolphin"], None, None
        if d is not None:
            print("recycle: swapped to pre-booted spare", flush=True)
        return d

    def _drain_old_stops(timeout: float = 20.0) -> None:
        """Cold boots must not overlap a dying Dolphin: the previous instance
        can still hold ports (seen live: retry Dolphin failed its spectator-
        server bind and wedged mid-boot, hanging the whole worker barrier).
        Wait for pending stops; SIGKILL any Dolphin whose stop() is stuck."""
        for t, pid in old_stops:
            t.join(timeout)
            if t.is_alive() and pid is not None:
                print(f"stop() stuck; SIGKILL dolphin pid {pid}", flush=True)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                t.join(timeout=5)
        old_stops[:] = [(t, p) for t, p in old_stops if t.is_alive()]

    class AlarmTimeout(Exception):
        pass

    def _alarm_handler(*_):
        raise AlarmTimeout()

    signal.signal(signal.SIGALRM, _alarm_handler)

    def _cold_boot():
        """Boot with a hard deadline: a Dolphin that wedges during startup
        (port collision, dead handshake) must become a bounded retry, not an
        infinite hang. SIGALRM is safe here: env-process main thread."""
        _drain_old_stops()
        signal.alarm(180)
        try:
            return make_dolphin(
                players, headless=cfg.headless, stage=cfg.stage,
                save_replays=cfg.save_replays, replay_dir=_replay_dir,
            )
        finally:
            signal.alarm(0)

    consecutive_boot_failures = 0
    consecutive_wedges = 0
    first_boot = True
    try:
        while True:
            # Recycle boundary = the ONLY place a desired policy<->cpu flip
            # is adopted: players are built fresh right below, exactly like a
            # cold boot. A kind change forces a redraw even with redraw_chars
            # off — the sitting char may be illegal for the new kind
            # (CPU-Sheik) and the player object type (AI vs CPU) changes.
            kind_changed = False
            if desired_kind is not None:
                want = "cpu" if desired_kind == "cpu" else spec.kind
                if want != cur_kind:
                    cur_kind = want
                    kind_changed = True
                    print(f"recycle: opponent kind adopted -> {cur_kind}",
                          flush=True)
            if not first_boot and (cfg.redraw_chars or kind_changed):
                # armed_char=None forces the lock through even when already
                # armed (players are rebuilt from scratch below)
                new_char = next_opponent_char(
                    cur_kind, char_lock, True, None, _draw_char
                )
                players.clear()
                players.update(_build_players(new_char, _draw_student_char()))
                # fresh Dolphin: its first CSS pass uses the new draw directly
                cur_opp_char = armed_opp_char = new_char
                print(f"recycle: opponent redrawn -> {new_char}", flush=True)
            first_boot = False
            try:
                dolphin = _take_spare() or _cold_boot()
                consecutive_boot_failures = 0
            except (AlarmTimeout, dolphin_lib.ConnectFailed) as e:
                # transient boot flakes (slow boot, console connect refusal
                # during a 128-wide boot storm) are retriable, not fatal
                consecutive_boot_failures += 1
                print(f"BOOT FAILURE ({consecutive_boot_failures}/3): {e}",
                      flush=True)
                if consecutive_boot_failures >= 3:
                    raise
                continue
            parser = Parser(ports=[1, 2])
            games = 0
            last_frame = None
            last_stocks = None
            # New-game gate: a pre-booted spare idles at the title screen,
            # where Melee's ATTRACT-MODE DEMO auto-plays after a timeout —
            # demo frames are "in-game" frames with garbage ports/fields
            # (live-caught: NaN states -> multinomial assert at the first
            # 128-env swap wave). Drop frames until a REAL game start
            # (frame counter resets to INITIAL_FRAME=-123, i.e. < 0).
            # Cold boots' first frame IS -123, so this is a no-op for them.
            game_started = False
            try:
              try:
                # Dolphin can freeze silently mid-game (live-caught: env 17,
                # zero log output, console.step never returned). Guard every
                # frame fetch with an alarm: legit silent stretches (boot +
                # menus + rematch) stay under ~60s, so 120s = wedged. On
                # trip: SIGKILL this Dolphin, mark the game aborted (reset,
                # no result), and boot a fresh one. Bounded: 3 wedges with
                # no completed game in between = something systemic, die
                # loudly (the worker watchdog is the outer backstop).
                gs_iter = iter(dolphin.iter_gamestates(skip_menu_frames=True))
                while True:
                    signal.alarm(120)
                    try:
                        gs = next(gs_iter)
                    except AlarmTimeout:
                        consecutive_wedges += 1
                        print(f"DOLPHIN WEDGED mid-stream "
                              f"({consecutive_wedges}/3); killing it",
                              flush=True)
                        if consecutive_wedges >= 3:
                            raise RuntimeError(
                                "dolphin wedged 3x without a completed game"
                            )
                        pid = _dolphin_pid(dolphin)
                        if pid is not None:
                            # forensics before the kill: what was Dolphin
                            # stuck on? (kernel wait-channel per thread)
                            try:
                                for tid in os.listdir(f"/proc/{pid}/task"):
                                    base = f"/proc/{pid}/task/{tid}"
                                    with open(f"{base}/wchan") as fh:
                                        wchan = fh.read().strip()
                                    with open(f"{base}/stat") as fh:
                                        state = fh.read().split()[2]
                                    print(f"  wedged tid {tid}: "
                                          f"state={state} wchan={wchan}",
                                          flush=True)
                            except OSError:
                                pass
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except OSError:
                                pass
                        pending_reset, pending_result = True, None
                        pending_result_kind = None
                        break
                    except StopIteration:
                        break
                    finally:
                        signal.alarm(0)
                    if not game_started:
                        if gs.frame > 0:
                            continue  # attract-mode demo frame: discard
                        game_started = True
                    serving = "cpu" if cur_kind == "cpu" else "policy"
                    boundary = last_frame is not None and gs.frame < last_frame
                    resetting = boundary or pending_reset
                    result = pending_result if pending_reset else None
                    result_kind = pending_result_kind if pending_reset else None
                    pending_reset, pending_result = False, None
                    pending_result_kind = None
                    if boundary:
                        games += 1
                        consecutive_wedges = 0
                        result = last_stocks  # ended game's final (bot, opp)
                        result_kind = serving  # kind is fixed within a dolphin
                        # the game starting NOW plays the previously armed char
                        cur_opp_char = armed_opp_char
                        if games >= cfg.games_per_dolphin:
                            pending_reset, pending_result = True, result
                            pending_result_kind = serving
                            break
                        # per-GAME character rotation: the vendor's menu
                        # helper and misselect guard both read
                        # player.character LIVE each menu pass, so
                        # mutating it between games retargets the next
                        # rematch CSS pick — no recycle needed. A char lock
                        # (import serving) pins the seat instead — no rng
                        # draw consumed; unlock resumes normal redraws.
                        nc = next_opponent_char(
                            cur_kind, char_lock, cfg.redraw_chars,
                            armed_opp_char, _draw_char,
                        )
                        if nc is not None:
                            players[opp_port].character = (
                                melee.Character[nc.upper()]
                            )
                            armed_opp_char = nc
                            verb = (
                                "pinned (char lock)"
                                if char_lock is not None and cur_kind != "cpu"
                                else "redrawn"
                            )
                            print(f"game end: opponent {verb} -> {nc}",
                                  flush=True)
                        if cfg.redraw_chars and len(whitelist) > 1:
                            # student-seat whitelist draws are unaffected by
                            # any opponent char lock (own rng stream)
                            sc = _draw_student_char()
                            players[spec.student_port].character = (
                                melee.Character[sc.upper()]
                            )
                            print(f"game end: student redrawn -> {sc}",
                                  flush=True)
                        parser = Parser(ports=[1, 2])
                    if games >= cfg.games_per_dolphin - 1:
                        _start_spare()  # entering this Dolphin's final game
                    last_frame = gs.frame
                    raw = tree_lib.map_structure(
                        np.asarray, parser.get_game(gs)
                    )
                    game = encode.flatten_typed(embed_game.from_state(raw))
                    # armor at the source: never ship a nonfinite frame
                    finite = all(
                        np.all(np.isfinite(leaf))
                        for leaf in tree_lib.flatten(game)
                        if np.issubdtype(np.asarray(leaf).dtype, np.floating)
                    )
                    if not finite:
                        print(f"nonfinite frame dropped (frame {gs.frame})",
                              flush=True)
                        continue
                    p1, p2 = gs.players[1], gs.players[2]
                    last_stocks = (int(p1.stock), int(p2.stock))
                    conn.send(
                        dict(
                            game=game,
                            resetting=resetting,
                            final_stocks=result,  # ended game's (port1, port2); None mid-game
                            stocks=last_stocks,
                            percent=(float(p1.percent), float(p2.percent)),
                            # opponent seat's char in the CURRENT game — the
                            # worker's imitation-harvest whitelist gate
                            opp_char=cur_opp_char,
                            # what the opponent seat ACTUALLY serves right
                            # now / served in the game `result` came from
                            # (league_cpu lazy adoption: may differ from the
                            # worker's desired kind until the next recycle)
                            opp_serving=serving,
                            result_serving=result_kind,
                        )
                    )
                    controllers = conn.recv()
                    if controllers is None:
                        return
                    # league_cpu: the worker piggybacks its desired serving
                    # kind on the command dict; stash it for the next recycle
                    desired = controllers.pop("opp_kind", None)
                    if desired is not None:
                        desired_kind = desired
                    # league_imports: piggybacked char lock (None = unlock);
                    # absent key (imports not configured) keeps the current
                    # value — i.e. stays None forever, today's behavior
                    char_lock = controllers.pop("opp_char_lock", char_lock)
                    for port, controller_state in controllers.items():
                        if cur_kind == "cpu" and port == opp_port:
                            continue  # engine AI drives this port; no inputs
                        controller_lib.send_controller(
                            dolphin.controllers[port], controller_state
                        )
              except WrongCharacterSelected as e:
                # menu cursor race under fast-forward (notably the Sheik/
                # Zelda slot): scrap this Dolphin and retry with a fresh one.
                # BOUNDED: persistent misselection means the character is
                # mechanically unpickable — die loudly, not loop forever
                # (learned via 362 consecutive CPU-Sheik retries).
                consecutive_misselects += 1
                if consecutive_misselects >= 3:
                    raise
                print(f"menu misselection, retrying: {e}", flush=True)
              else:
                consecutive_misselects = 0
            finally:
                # Recycle path: stop the old Dolphin off-thread so a SPARE
                # swap isn't gated on teardown. Cold boots drain these first
                # (_drain_old_stops), so teardown/boot never overlap except
                # in the validated healthy-spare case. Non-daemon: shutdown
                # waits for the kills (no zombie Dolphins).
                pid = _dolphin_pid(dolphin)
                t = threading.Thread(target=dolphin.stop)
                t.start()
                old_stops.append((t, pid))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        if spare["thread"] is not None:
            spare["thread"].join(timeout=60)
            if spare["dolphin"] is not None:
                spare["dolphin"].stop()
        _drain_old_stops()
