import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "ImagiNav" / "LTX-Video-Trainer"


def test_open_source_project_files_exist():
    for relative_path in (
        "LICENSE",
        "README.md",
        "ImagiNav/README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
    ):
        assert (ROOT / relative_path).is_file(), relative_path


def test_supported_python_range_includes_python_312_patches():
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r"python_requires=['\"]([^'\"]+)", setup_text)
    assert match is not None

    supported = SpecifierSet(match.group(1))
    assert "3.9.18" not in supported
    assert "3.12.10" in supported
    assert "3.13.0" not in supported


def test_package_discovery_is_scoped_to_internnav():
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "find_packages(include=['internnav', 'internnav.*'])" in setup_text
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include requirements *.txt" in manifest
    assert "prune FastWAM" in manifest


def test_imaginav_requirements_are_machine_readable():
    requirements = (ROOT / "ImagiNav" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    for line in requirements:
        line = line.strip()
        if line and not line.startswith("#"):
            Requirement(line)


def test_imaginav_readme_uses_tools_and_paths_from_frozen_environment():
    guide = (ROOT / "ImagiNav" / "README.md").read_text(encoding="utf-8")
    assert ".venv/bin/hf" not in guide
    assert "hf auth login" not in guide
    assert 'export IMAGINAV_HF="$IMAGINAV_ROOT/LTX-Video-Trainer/.venv/bin/huggingface-cli"' in guide
    assert 'mkdir -p "$IMAGINAV_ROOT/video_quality_eval/datasets"' in guide


def test_only_imaginav_training_configs_are_published():
    expected = {
        "ltxv_13b_lora-imaginav.yaml",
        "ltxv_2b_full-imaginav.yaml",
        "ltxv_2b_lora-imaginav-left.yaml",
        "ltxv_2b_lora-imaginav-right.yaml",
    }
    actual = {path.name for path in (TRAINER / "configs").glob("*.yaml")}
    assert actual == expected

