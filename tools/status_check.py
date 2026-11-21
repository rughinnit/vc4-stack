#!/usr/bin/env python3
"""Fail when docs/STATUS.md stops matching the repository.

The status page is prose, but four of its claims are mechanical:

* every state is one of ``done``, ``limit``, ``open``, ``in progress``;
* every evidence path written in backticks exists in this tree;
* every ``done`` row cites at least one evidence path;
* the ONNX completeness numbers equal ``docs/completeness.json``.

It also checks that the stated state counts match the table and that the
workstreams the page is required to cover are still present.  Standard
library only, because it has to run on the board.
"""

import collections
import json
import os
import re
import sys

STATES = ("done", "limit", "open", "in progress")

# A token is only treated as a path when it has a directory and an extension.
# This deliberately skips branch names such as feat/litert-delegate.
PATH_SPAN = re.compile(r"`([^`\n]+)`")
PATH_SHAPE = re.compile(r"\.[A-Za-z0-9]+$")
STATE_COUNTS = re.compile(
    r"States:\s*(\d+)\s+done,\s*(\d+)\s+limit,\s*(\d+)\s+open,"
    r"\s*(\d+)\s+in progress")
EXPRESSIBLE = re.compile(
    r"expressible denominator:\s*(\d+)/(\d+)\s+complete,\s*"
    r"(\d+)\s+refused,\s*(\d+)\s+folded")
OPSET = re.compile(
    r"whole opset 22:\s*(\d+)/(\d+)\s+complete,\s*"
    r"(\d+)\s+structurally excluded")
ONNX_ROW = re.compile(
    r"(\d+)/(\d+)\s+expressible,\s*(\d+)/(\d+)\s+opset 22,\s*"
    r"(\d+)\s+refused,\s*(\d+)\s+folded,\s*"
    r"(\d+)\s+structurally excluded")

# The page must keep covering these workstreams.  Match on a lowercase token.
REQUIRED = (
    "engine", "onnx", "daemon", "training", "ggml", "pytorch", "jax",
    "litert", "tensorflow", "tvm", "packaging", "documentation",
    "hygiene", "performance", "arena", "thermal",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS = os.path.join(ROOT, "docs", "STATUS.md")
COMPLETENESS = os.path.join(ROOT, "docs", "completeness.json")


def fail(failures, where, what, expected, actual):
    failures.append((where, what, expected, actual))


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def workstream_rows(text):
    """Return the 4-column rows of the table under '## Workstreams'."""
    if "## Workstreams" not in text:
        return []
    section = text.split("## Workstreams", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        if cells[0].lower() == "workstream":
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append(cells)
    return rows


def evidence_paths(text):
    """Yield path-shaped backtick spans from the whole page."""
    seen = set()
    for span in PATH_SPAN.findall(text):
        token = span.strip()
        if "://" in token:
            continue
        token = token.split("#", 1)[0].strip()
        token = re.split(r"\s*§", token, maxsplit=1)[0].strip()
        token = re.sub(r":\d+(?:-\d+)?$", "", token).strip()
        if not token or "/" not in token or any(ch.isspace() for ch in token):
            continue
        if not PATH_SHAPE.search(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        yield token


def normalise_state(cell):
    return re.sub(r"[`*_]", "", cell).strip()


def main():
    failures = []
    try:
        text = read(STATUS)
    except OSError as error:
        print("FAIL [docs/STATUS.md] cannot read: %s" % error)
        return 1

    rows = workstream_rows(text)
    if not rows:
        print("FAIL [docs/STATUS.md] no 4-column table under '## Workstreams'")
        return 1

    # 1. States and the done-evidence rule.
    counts = collections.Counter()
    for row in rows:
        name, state, evidence, _change = row
        state = normalise_state(state)
        counts[state] += 1
        if state not in STATES:
            fail(failures, name, "state", "one of: " + ", ".join(STATES),
                 repr(state) or "(empty)")
            continue
        paths = list(evidence_paths(evidence))
        for path in paths:
            if not os.path.exists(os.path.join(ROOT, path)):
                fail(failures, name, "evidence path", "exists", path)
        if state == "done" and not paths:
            fail(failures, name, "done evidence",
                 "at least one path in backticks", "none")

    # 2. Required coverage.
    names = " | ".join(row[0].lower() for row in rows)
    for token in REQUIRED:
        if token not in names:
            fail(failures, "coverage", "workstream", token, "missing")

    # 3. The declared counts must equal the table.
    match = STATE_COUNTS.search(text)
    if not match:
        fail(failures, "docs/STATUS.md", "state counts",
             "a line 'States: N done, N limit, N open, N in progress'",
             "not found")
    else:
        expected = dict(zip(("done", "limit", "open", "in progress"),
                            (int(value) for value in match.groups())))
        for state in STATES:
            if counts.get(state, 0) != expected[state]:
                fail(failures, "docs/STATUS.md",
                     "count of %s rows" % state,
                     expected[state], counts.get(state, 0))

    # 4. Completeness numbers against the machine-readable sweep.
    try:
        summary = json.loads(read(COMPLETENESS))["summary"]
        actual = {
            "expressible complete": summary["expressible"]["complete"],
            "expressible total": summary["expressible"]["total"],
            "expressible refused": summary["expressible"]["refused"],
            "expressible folded": summary["expressible"]["folded"],
            "opset complete": summary["all"]["complete"],
            "opset total": summary["all"]["total"],
            "opset excluded": summary["all"]["excluded"],
        }
    except (OSError, KeyError, ValueError) as error:
        print("FAIL [docs/completeness.json] cannot read the summary: %s"
              % error)
        return 1

    expressible = EXPRESSIBLE.search(text)
    if not expressible:
        fail(failures, "Completeness", "expressible line",
             "'expressible denominator: N/N complete, N refused, N folded'",
             "not found")
    else:
        got = {
            "expressible complete": int(expressible.group(1)),
            "expressible total": int(expressible.group(2)),
            "expressible refused": int(expressible.group(3)),
            "expressible folded": int(expressible.group(4)),
        }
        for key, value in got.items():
            if value != actual[key]:
                fail(failures, "Completeness", key, actual[key], value)

    opset = OPSET.search(text)
    if not opset:
        fail(failures, "Completeness", "opset line",
             "'whole opset 22: N/N complete, N structurally excluded'",
             "not found")
    else:
        got = {
            "opset complete": int(opset.group(1)),
            "opset total": int(opset.group(2)),
            "opset excluded": int(opset.group(3)),
        }
        for key, value in got.items():
            if value != actual[key]:
                fail(failures, "Completeness", key, actual[key], value)

    # 4b. The ONNX row must carry the same numbers, so a reader who stops at
    # the table is not reading a stale figure.
    onnx_rows = [row for row in rows if "onnx" in row[0].lower()]
    if not onnx_rows:
        fail(failures, "coverage", "workstream", "onnx", "missing")
    else:
        row_with_numbers = ONNX_ROW.search(onnx_rows[0][2])
        if not row_with_numbers:
            fail(failures, onnx_rows[0][0], "completeness in the row",
                 "'N/N expressible, N/N opset 22, N refused, N folded, "
                 "N structurally excluded'", "not found")
        else:
            got = {
                "expressible complete": int(row_with_numbers.group(1)),
                "expressible total": int(row_with_numbers.group(2)),
                "opset complete": int(row_with_numbers.group(3)),
                "opset total": int(row_with_numbers.group(4)),
                "expressible refused": int(row_with_numbers.group(5)),
                "expressible folded": int(row_with_numbers.group(6)),
                "opset excluded": int(row_with_numbers.group(7)),
            }
            for key, value in got.items():
                if value != actual[key]:
                    fail(failures, onnx_rows[0][0], key, actual[key], value)

    # 5. Every path anywhere on the page exists.
    all_paths = set(evidence_paths(text))
    for path in sorted(all_paths):
        if not os.path.exists(os.path.join(ROOT, path)):
            fail(failures, "docs/STATUS.md", "cited path", "exists", path)

    if failures:
        for where, what, expected, got in failures:
            print("FAIL [%s] %s: expected %s, actual %s"
                  % (where, what, expected, got))
        print("status_check: %d failure(s)" % len(failures))
        return 1

    print("status_check: OK — %d workstreams (%d done, %d limit, %d open, "
          "%d in progress), %d evidence paths, completeness matches "
          "docs/completeness.json"
          % (len(rows), counts.get("done", 0), counts.get("limit", 0),
             counts.get("open", 0), counts.get("in progress", 0),
             len(all_paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
