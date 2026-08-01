from pathlib import Path


def test_console_script_name_is_a_literal_dotted_key():
    metadata = (
        Path(__file__).parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert '"pw.py" = "qepy_pw.cli:main"' in metadata
