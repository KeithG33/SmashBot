"""Canonical filesystem locations for ShineBot.

Everything heavy (datasets, checkpoints, caches) lives on drive2; the repo
holds only code. Override any location with the corresponding env var.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_SLIPPI_AI = REPO_ROOT / "vendor" / "slippi-ai"

DRIVE2 = Path(os.environ.get("SHINEBOT_DRIVE2", "/home/kage/drive2/ShineBot"))
DATA_DIR = DRIVE2 / "data"
RUNS_DIR = DRIVE2 / "runs"
MODELS_DIR = DRIVE2 / "models"

MELEE_ISO = Path(
    os.environ.get(
        "SHINEBOT_ISO",
        "/home/kage/slippi/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso",
    )
)
# Mainline ExiAI "NoLeak" build (2026-03): headless + fast-forward support.
EXIAI_APPIMAGE = Path(
    os.environ.get(
        "SHINEBOT_DOLPHIN",
        str(DRIVE2 / "dolphin" / "Slippi_Netplay_Mainline_ExiAI_NoLeak-x86_64.AppImage"),
    )
)
# Ishiiruka ExiAI build (Slippi 3.5.1) — fallback / playback duties.
EXIAI_ISHIIRUKA_APPIMAGE = DRIVE2 / "dolphin" / "Slippi_Online-x86_64-ExiAI.AppImage"
