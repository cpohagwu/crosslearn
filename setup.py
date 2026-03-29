from __future__ import annotations

import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
INIT_FILE = ROOT / "crosslearn" / "__init__.py"
README_FILE = ROOT / "README.md"


def read_version() -> str:
    text = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*\"([^\"]+)\"", text)
    if not match:
        raise RuntimeError(f"Unable to find __version__ in {INIT_FILE}")
    return match.group(1)


def read_readme() -> str:
    return README_FILE.read_text(encoding="utf-8")

setup(
    name="crosslearn",
    version=read_version(),
    description=(
        "Extractor-first RL utilities for Gymnasium, SB3, and Chronos-backed time-series features"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(include=["crosslearn", "crosslearn.*"]),
    include_package_data=True,
    keywords=[
        "reinforcement-learning",
        "stable-baselines3",
        "gymnasium",
        "chronos",
        "atari",
    ],
)
