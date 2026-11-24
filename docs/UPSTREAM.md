# Upstream and our work

Three of the projects this stack is built on are forks, and the other two are
ours. Keeping that boundary explicit is what makes "pull the upstream product"
a safe operation instead of a merge of unknown provenance.

| repository | who owns the code upstream | our work |
|---|---|---|
| `VC4CL` | `doe300/VC4CL` | the OpenCL 1.2 runtime on top of the QPU, plus the patches that make it build and pass on Debian 13 |
| `VC4C` | `doe300/VC4C` | the compiler back end: register allocation, VPM staging, the pipeline work on `perf/*` branches |
| `VC4CLStdLib` | `doe300/VC4CLStdLib` | intrinsics for later OpenCL C versions |
| `vc4-stack` | us | the engine, the kernels, the framework adapters and this documentation |
| `vc4top` | us | the utilisation and thermal viewer |

## The layout, per fork

```
origin    our fork on GitHub          (rughinnit/<name>)
upstream  the project we forked       (doe300/<name>)
upstream-master   a local, read-only mirror of upstream/master
master / feature branches   our work
```

`upstream-master` is never committed to and never pushed. It exists so a
divergence can be read with one command and so a merge has a named source. Our
branches are merged **with** upstream, never written over it, and no upstream
commit is ever rewritten: a fork that rewrites its parent's history cannot be
compared with it again.

```sh
git -C ~/vc4cl/VC4CL fetch upstream
git -C ~/vc4cl/VC4CL log --oneline upstream/master..master      # our work
git -C ~/vc4cl/VC4CL log --oneline master..upstream/master      # upstream's
```

## Checking, and the release rule

[`tools/upstream-status.sh`](../tools/upstream-status.sh) prints, per fork, how
many upstream commits are missing, how many commits are ours, the branch, and
whether the tree is dirty:

```sh
tools/upstream-status.sh                  # from what is already fetched
tools/upstream-status.sh --fetch           # fetch first, needs network
```

It exits non-zero when any fork is behind, so it can gate a release:

- **behind = 0, ahead = 0**: the fork is the upstream tree. Nothing to do.
- **behind = 0, ahead > 0**: our patches sit on top of current upstream. Normal.
- **behind > 0**: upstream has moved. Merge `upstream/master` into the branch
  before claiming the fork is current, and say in the merge commit what it
  brought.

## Syncing a fork with its upstream

```sh
cd ~/vc4cl/VC4CL
git fetch upstream
git switch master
git merge upstream/master          # or: git rebase upstream/master, on unpublished work only
git push origin master
```

If the merge conflicts, the conflict is a real upstream change and is resolved
in favour of upstream unless our patch exists for a reason that still holds;
then the patch is rewritten on top, in its own commit, with the reason.

Rebasing is only for commits that exist nowhere but the local clone. A branch
that has been pushed is merged, because a force-push to a fork destroys the
comparison with upstream that this document is about.

## What is vendored instead of forked

`torch-vc4` was folded into `vc4-stack/torch/` with its history rather than
kept as a fork: its only upstream is PyTorch itself, and the adapter is a
plugin, not a patch. Its old repository still exists and was left untouched,
which is why the local copies of it were removed during housekeeping.

## Current state, measured

`tools/upstream-status.sh --fetch` on the reference board, 2026-11-24:

| repository | branch | behind | ahead | note |
|---|---|---:|---:|---|
| `VC4CL` | `master` | 66 | 104 | upstream moved; the fork carries the runtime patches |
| `VC4C` | `feat/lane-block-stores` | 902 | 25 | the compiler work sits on an older base |
| `VC4CLStdLib` | `master` | 0 | 0 | in sync |

VC4C's 902 is the number to read carefully: the branch carries 25 commits of
compiler work on a base that upstream has moved far away from. Merging current
upstream into it is a real project, not a housekeeping step, and
[`FUTURE-WORK.md`](FUTURE-WORK.md) records it as one. The fork's own `master`
carries a single patch and is otherwise the same old base.
