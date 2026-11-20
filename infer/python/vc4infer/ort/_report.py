"""The honesty table: which node ran where, how much of the work that is.

The percentages are the point.  A provider that claims 90% of the nodes but
1% of the MACs is not "accelerating" a model, it is decorating it, and this
report says so in those words.  Same rule as ``vc4infer util``: the numbers
come from the static plan plus measured wall time, never from an estimate
presented as a measurement.
"""
import sys

# Rows are aggregated by these labels; "vc4-folded" is still a VC4-side node
# (Constant/Shape resolved by the strict executor at plan time, zero launches).
BACKENDS = ("vc4", "vc4-folded", "ort-cpu")


def _human(n):
    n = float(n)
    for unit in ("", "k", "M", "G"):
        if abs(n) < 1000.0:
            return "%d" % n if not unit else "%.1f%s" % (n, unit)
        n /= 1000.0
    return "%.1fT" % n


def _pct(part, whole):
    if not whole:
        return "-"
    return "%.1f%%" % (100.0 * part / whole)


def rows(session):
    """Per-node attribution.

    ``backend`` is ``vc4``, ``vc4-folded`` or ``ort-cpu``.  ``wall_ms`` is
    filled for VC4 nodes from the engine's own per-node profile when the last
    run recorded one; ORT nodes only have a partition-level measurement, so
    their ``wall_ms`` stays ``None`` and the partition total is reported
    instead of a fabricated per-node split.
    """
    rep = session.coverage
    out = []
    prof = getattr(session, "_last_vc4_profile", {}) or {}
    for part in session.partitions:
        for idx in part.indices:
            e = rep["nodes"][idx]
            if part.backend == "vc4":
                backend = "vc4-folded" if e["tier"] == "host-meta" else "vc4"
            else:
                backend = "ort-cpu"
            row = {
                "index": idx,
                "name": e["name"],
                "op": e["op"],
                "backend": backend,
                "reason": e["tier_reason"] if backend != "ort-cpu"
                          else e["reason"],
                "macs": int(e.get("macs") or 0),
                "wall_ms": None,
                "gpu_ms": None,
                "launches": None,
                "partition": part.index,
            }
            p = prof.get(idx)
            if p is not None:
                row["wall_ms"] = p.get("wall_ms")
                row["gpu_ms"] = p.get("gpu_ms")
                row["launches"] = p.get("launches")
            out.append(row)
    return out


def summary(session):
    rep = session.coverage
    counts = dict((b, 0) for b in BACKENDS)
    macs = dict((b, 0) for b in BACKENDS)
    for r in rows(session):
        counts[r["backend"]] += 1
        macs[r["backend"]] += r["macs"]
    total = sum(counts.values())
    macs_total = sum(macs.values())
    return {
        "nodes": total,
        "macs_total": macs_total,
        "counts": counts,
        "macs": macs,
        "pct_nodes": dict((b, (100.0 * counts[b] / total) if total else 0.0)
                          for b in BACKENDS),
        "pct_macs": dict((b, (100.0 * macs[b] / macs_total) if macs_total
                          else None) for b in BACKENDS),
        "parts_vc4": sum(1 for p in session.partitions if p.backend == "vc4"),
        "parts_ort": sum(1 for p in session.partitions if p.backend == "ort-cpu"),
        "claimed_all": counts["ort-cpu"] == 0,
    }


def render(session, per_node=False, out=None):
    """Print the split table and the last run's per-backend timing."""
    out = out or sys.stdout
    s = summary(session)
    rep = session.coverage
    np = len(session.partitions)
    print("%s %s -- %s (%d nodes, %d partition%s)" % (
        "vc4infer.ort", session.version, session.model_name, s["nodes"],
        np, "" if np == 1 else "s"), file=out)
    print("  providers requested: %s" % ", ".join(session.providers),
          file=out)
    print("  %-11s %7s %8s %11s %8s" % (
        "backend", "nodes", "%nodes", "MACs", "%MACs"), file=out)
    for b in BACKENDS:
        pm = s["pct_macs"][b]
        print("  %-11s %7d %8s %11s %8s" % (
            b, s["counts"][b], _pct(s["counts"][b], s["nodes"]),
            _human(s["macs"][b]), "n/a" if pm is None else "%.1f%%" % pm),
            file=out)
    print("  %-11s %7d %8s %11s" % ("total", s["nodes"], "100.0%",
                                    _human(s["macs_total"])), file=out)

    t = session.last_timing
    if t:
        print("  last run:  vc4 %.2f ms | ort-cpu %.2f ms | transfer %.2f ms"
              % (t["vc4_ms"], t["ort_ms"], t["transfer_ms"]), file=out)
        print("  work share: vc4 %s of nodes, %s of MACs" % (
            _pct(s["counts"]["vc4"] + s["counts"]["vc4-folded"], s["nodes"]),
            "n/a" if s["pct_macs"]["vc4"] is None
            else "%.1f%%" % s["pct_macs"]["vc4"]), file=out)

    if s["claimed_all"]:
        print("  fully claimed: every node runs on the QPUs", file=out)
    else:
        print("  NOT fully claimed: %d node(s) run on the ORT CPU EP"
              % s["counts"]["ort-cpu"], file=out)

    if per_node:
        print("\n  %-6s %-34s %-14s %-10s %9s %8s" % (
            "index", "node", "op", "backend", "MACs", "wall ms"), file=out)
        for r in rows(session):
            wall = "-" if r["wall_ms"] is None else "%.2f" % r["wall_ms"]
            print("  %-6d %-34s %-14s %-10s %9s %8s" % (
                r["index"], r["name"][:34], r["op"][:14], r["backend"],
                _human(r["macs"]), wall), file=out)
    return s
