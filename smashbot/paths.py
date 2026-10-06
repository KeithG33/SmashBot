"""Canonical filesystem locations for SmashBot.

Everything heavy (datasets, checkpoints, caches) lives on drive2; the repo
holds only code. Override any location with the corresponding env var.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# melee-sim-light: vendored source (not an installed package) and the game data
# it loads, extracted from the ISO
MELEE_SIM_DIR = REPO_ROOT / "vendor" / "melee-sim-light"

DRIVE2 = Path(os.environ.get("SMASHBOT_DRIVE2", "/home/kage/drive2/ShineBot"))
MODELS_DIR = DRIVE2 / "models"
MSL_DATA_DIR = Path(os.environ.get("MSL_DATA_DIR", str(DRIVE2 / "msl-data")))

MELEE_ISO = Path(
    os.environ.get(
        "SMASHBOT_ISO",
        "/home/kage/slippi/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso",
    )
)
# Mainline ExiAI "NoLeak" build: headless + fast-forward support
EXIAI_APPIMAGE = Path(
    os.environ.get(
        "SMASHBOT_DOLPHIN",
        str(DRIVE2 / "dolphin" / "Slippi_Netplay_Mainline_ExiAI_NoLeak-x86_64.AppImage"),
    )
)
# standard Slippi netplay Dolphin, which renders (ExiAI builds are null-video):
# the Launcher-maintained build
NETPLAY_APPIMAGE = Path(
    os.environ.get(
        "SMASHBOT_NETPLAY_DOLPHIN",
        "/home/kage/.config/Slippi Launcher/netplay/Slippi_Online-x86_64.AppImage",
    )
)
