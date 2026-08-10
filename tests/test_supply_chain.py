# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
ACTION_REFERENCE = re.compile(r"^\s*-\s+uses:\s+([^\s#]+)", re.MULTILINE)
IMMUTABLE_SHA = re.compile(r"[0-9a-f]{40}")


def test_github_actions_are_pinned_to_immutable_commits() -> None:
    unpinned: list[str] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        for action in ACTION_REFERENCE.findall(workflow.read_text(encoding="utf-8")):
            if action.startswith("./"):
                continue
            _repository, separator, revision = action.rpartition("@")
            if not separator or IMMUTABLE_SHA.fullmatch(revision) is None:
                unpinned.append(f"{workflow.name}: {action}")

    assert unpinned == []
