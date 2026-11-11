import tomllib
from pathlib import Path

import review_agents


def test_reports_the_version_declared_in_pyproject() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert review_agents.__version__ == pyproject["project"]["version"]
