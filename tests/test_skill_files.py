import re
from importlib.resources import files
from importlib.resources.abc import Traversable

import pytest

SKILLS = files("review_agents") / "skills"
DIMENSIONS = ["spec_drift", "edge_cases", "security", "maintainability", "verifiability"]

# Words from the evaluation cases, plurals included. A skill that names them teaches the answer
# instead of the method, and the evaluation stops measuring anything.
BANNED = [
    "NaN",
    "None",
    "ZeroDivision",
    "division by zero",
    "0 / 0",
    "prefix",
    "case-sensitive",
    "mean",
    "word",
    "inbox",
    "template",
]
# Generic injection classes a security reviewer must know; removed before the scan.
ALLOWED_PHRASES = ["template injection"]


def frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "missing YAML frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        assert sep, f"not a key: value line: {line!r}"
        fields[key.strip()] = value.strip()
    return fields


def skill_files() -> list[Traversable]:
    found = [SKILLS / "_shared.md", SKILLS / "_walkthrough.md"]
    found += [SKILLS / name / "SKILL.md" for name in DIMENSIONS]
    return found


@pytest.mark.parametrize("name", DIMENSIONS)
def test_each_dimension_has_a_skill_with_matching_frontmatter(name: str) -> None:
    text = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")

    fields = frontmatter(text)

    assert fields["name"] == name
    assert fields["description"]
    assert "_shared.md" in text


def test_shared_rules_define_every_severity_and_the_evidence_rule() -> None:
    text = (SKILLS / "_shared.md").read_text(encoding="utf-8")

    for level in ["🔴 blocking", "🟡 important", "🔵 robustness", "⚪ nit"]:
        assert level in text
    assert "Every 🔴 and 🟡 needs at least one basis" in text
    for kind in ("`spec`", "`contract`", "`convention`", "`path`"):  # decision D-036
        assert kind in text
    assert "`evidence`" not in text  # replaced by `basis`


def test_every_skill_directory_is_a_known_dimension() -> None:
    folders = {entry.name for entry in SKILLS.iterdir() if entry.is_dir()}
    assert folders - {"__pycache__"} == set(DIMENSIONS)


@pytest.mark.parametrize("skill", skill_files(), ids=lambda skill: str(skill).split("skills/")[-1])
def test_skills_do_not_name_evaluation_answers(skill: Traversable) -> None:
    text = skill.read_text(encoding="utf-8")
    for phrase in ALLOWED_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)

    leaked = [
        term
        for term in BANNED
        if re.search(rf"(?<!\w){re.escape(term)}(?:e?s)?(?!\w)", text, flags=re.IGNORECASE)
    ]

    assert leaked == []


def test_the_walkthrough_skill_names_every_field_it_asks_for() -> None:
    text = (SKILLS / "_walkthrough.md").read_text(encoding="utf-8")

    for name in ("summary", "why", "what", "impact", "diagram", "problems", "changes"):
        assert f"`{name}`" in text
    for name in ("impacts", "deploy_notes", "tradeoffs", "submit_walkthrough"):
        assert f"`{name}`" in text
    assert "at most three" in text  # Mermaid diagrams in the whole walkthrough
