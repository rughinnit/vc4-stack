#!/usr/bin/env python3
"""test_serve_page.py - the page tools/serve.py embeds must be valid JavaScript.

The page is a Python string, so an edit can leave a literal newline inside a
quoted JavaScript string (an escaped \\n in the template is enough) and the
browser refuses the whole block: the page loads, stays on "loading model list",
and the console shows "Invalid or unexpected token".  That happened once, so the
check runs in the suite.

`node --check` when node is installed, which is the real parser; a quote and
bracket balance check otherwise.

Usage: python3 infer/tests/test_serve_page.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SERVE = os.path.join(REPO, "tools", "serve.py")


def extract_pages(src):
    pages = {}
    for name in ("LOGIN", "PAGE"):
        m = re.search(r'^%s = """.*?"""\.encode\(\)' % name, src, re.S | re.M)
        if not m:
            raise SystemExit("cannot find %s in %s" % (name, SERVE))
        block = m.group(0)
        body = re.search(r'"""(.*)"""\.encode\(\)', block, re.S).group(1)
        pages[name] = body.encode().decode("unicode_escape", "replace") \
            if "\\n" in body or "\\t" in body else body
    return pages


def script_of(html):
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    return m.group(1) if m else ""


def node_check(js, where):
    node = shutil.which("node")
    if not node:
        return None
    fd, path = tempfile.mkstemp(suffix=".js", dir=os.environ.get("TMPDIR"))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(js)
        r = subprocess.run([node, "--check", path], capture_output=True,
                           text=True)
        if r.returncode != 0:
            msg = (r.stderr or "").strip().splitlines()
            return ["%s: %s" % (where, ln) for ln in msg[:3]]
        return []
    finally:
        os.unlink(path)


def check_fallback(js, where):
    """Quotes must not span a line; brackets must balance."""
    problems, quote = [], None
    for lineno, line in enumerate(js.splitlines(), 1):
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif ch == "/" and line[i:i + 2] == "//":
                break
            i += 1
        if quote is not None:
            problems.append("%s:%d unclosed %s string" % (where, lineno, quote))
            quote = None
    pairs = {")": "(", "]": "[", "}": "{"}
    stack, quote = [], None
    for lineno, line in enumerate(js.splitlines(), 1):
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif ch in "([{":
                stack.append((ch, lineno))
            elif ch in ")]}":
                if not stack or stack[-1][0] != pairs[ch]:
                    return problems + ["%s:%d unbalanced %s" % (where, lineno, ch)]
                stack.pop()
            i += 1
    if stack:
        problems.append("%s:%d unclosed %s" % (where, stack[-1][1], stack[-1][0]))
    return problems


def main():
    with open(SERVE, encoding="utf-8") as fh:
        src = fh.read()
    pages = extract_pages(src)

    problems, parsed_by = [], "fallback"
    for name, html in pages.items():
        js = script_of(html)
        if not js:
            problems.append("%s has no <script> block" % name)
            continue
        res = node_check(js, name)
        if res is not None:
            parsed_by = "node --check"
            problems += res
        else:
            problems += check_fallback(js, name)

    page = pages["PAGE"]
    for needle in ("/models", "/select", "/detect", "/health",
                   "refresh", "load", "show"):
        if needle not in page:
            problems.append("PAGE no longer mentions %r" % needle)
    for needle in ("--token", "--host", "--port"):
        if needle not in src:
            problems.append("serve.py no longer accepts %r" % needle)

    if problems:
        for p in problems:
            print("FAIL", p)
        return 1
    print("serve page: %d script(s) valid (%s), endpoints present"
          % (len(pages), parsed_by))
    return 0


if __name__ == "__main__":
    sys.exit(main())
