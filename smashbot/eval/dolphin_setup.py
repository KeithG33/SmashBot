"""Torch-free Dolphin construction (importable by env processes)."""

from __future__ import annotations

import melee
from slippi_ai import dolphin as dolphin_lib

from smashbot.paths import EXIAI_APPIMAGE, MELEE_ISO, NETPLAY_APPIMAGE


def make_dolphin(
    players: dict,
    headless: bool,
    stage: str = "FINAL_DESTINATION",
    fullscreen: bool = False,
    gfx_backend: str = "OGL",
    online_delay: int = 0,
    mute: bool = False,
    save_replays: bool = False,
    replay_dir: str = "",
) -> dolphin_lib.Dolphin:
    """One Dolphin. Headless uses the ExiAI build (Null video, fast-forward);
    visible play uses the standard netplay build. save_replays writes .slp
    files (headless included — run fast, watch later in Slippi at 60fps)."""
    console_kwargs: dict = {"stage": melee.Stage[stage.upper()]}
    if save_replays:
        console_kwargs["save_replays"] = True
        if replay_dir:
            console_kwargs["replay_dir"] = replay_dir
    # Slippi builds open slippi.gg/online/enable in a browser when their
    # (fresh temp) user dir has no linked account. All our games are local
    # direct-mode (headless fleets AND rendered watch sessions), so make
    # URL-opening a no-op for every Dolphin we spawn. $BROWSER alone is NOT
    # enough: on KDE, xdg-open delegates to kde-open and ignores it — so we
    # also shadow the openers via PATH (shim dir of exit-0 scripts).
    import os

    os.environ["BROWSER"] = "true"
    _noopen = "/home/kage/drive2/ShineBot/noopen"
    if not os.environ.get("PATH", "").startswith(_noopen):
        os.environ["PATH"] = _noopen + os.pathsep + os.environ.get("PATH", "")
    if headless:
        path = EXIAI_APPIMAGE
    else:
        path = NETPLAY_APPIMAGE
        console_kwargs["fullscreen"] = fullscreen
        if gfx_backend:
            console_kwargs["gfx_backend"] = gfx_backend
        if mute:
            # Pulse underruns ("Dropping OutputStream") disturb Dolphin's
            # frame pacing; muting removes the audio path entirely.
            console_kwargs["disable_audio"] = True
    return dolphin_lib.Dolphin(
        path=str(path),
        iso=str(MELEE_ISO),
        players=players,
        headless=headless,
        online_delay=online_delay,
        emulation_speed=0 if headless else 1,
        **console_kwargs,
    )
