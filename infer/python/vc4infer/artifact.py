"""A serialized engine artifact: the model after the optimiser (``.vc4x``).

An artifact is the equivalent of a TensorRT engine: the graph the optimiser
produced, the weights prepacked into the arena layout the kernels read, the
arena plan with fixed offsets, the kernel blobs the plan will launch, and the
keys that make the file refuse to run anywhere it was not built for.

    vc4infer build model.onnx -o model.vc4x
    eng = vc4infer.Engine("model.vc4x")      # load: no passes, no packing
    eng = vc4infer.Engine("model.onnx", passes="all")   # plan as before

What loading skips, and what it does not:

* the pass pipeline (the graph in the artifact is already optimised, with the
  provenance records intact, so the executor's rewrite tables are rebuilt from
  those records);
* shape refinement and the arena estimate (both are recorded);
* weight packing (the conv panels, expand tables and GEMM operand copies are
  in the artifact already packed, and are uploaded at their recorded offsets);
* kernel generation, when the blobs are embedded (they are written into the
  kernel cache, which the existing cache path then reuses unchanged).

What loading still does: parse the ONNX graph, upload constants and panels,
and execute.  The artifact is a plan, not an interpreter.

Hardware and generator specificity: the file carries the kernel generator
digest, the kernel ABI tag, the artifact layout version, a digest of the engine
Python package and the board model string.  A mismatch is refused with the key
that differs, before the arena is touched.  Unlike the ONNX file, an artifact
is not portable between boards, generators or engine versions -- rebuild it.

Format (version 1), little-endian:

    offset 0   8    magic "VC4XART1"
    8          4    format version (u32)
    12         4    manifest length (u32, bytes of UTF-8 JSON)
    16         8    payload length (u64)
    24         32   sha256 of the manifest bytes
    56         N    manifest JSON (sorted keys, ASCII)
    56+N       M    payload: the sections the manifest's table names

Each manifest section is ``{"name", "offset", "bytes", "sha256"}`` with the
offset relative to the start of the payload.  The graph, the prepacked weights
and the kernel blobs are sections; a section is hashed when it is written and
can be verified on load with ``Artifact.verify()``.  The header and the
manifest are always verified, so a truncated or edited manifest cannot reach
the executor.
"""
import hashlib
import json
import os
import platform
import struct
import sys
import time

from . import _plan
from ._errors import Vc4InferError

MAGIC = b"VC4XART1"
FORMAT = "vc4x"
LAYOUT_VERSION = 1
HEADER = struct.Struct("<8sIIQ32s")

# Only used when the library predates the accessor; the library is the
# authority and the generator digest includes this string either way.
ABI_TAG_FALLBACK = "vc4infer-kernel-abi-2"


class ArtifactError(Vc4InferError):
    """An artifact could not be read, or does not match this engine."""


class ArtifactMismatch(ArtifactError):
    """A key in the artifact disagrees with this engine. Named, never silent."""

    def __init__(self, key, artifact, engine, hint=""):
        self.key = key
        self.artifact = artifact
        self.engine = engine
        self.hint = hint
        msg = ("artifact rejected: %s mismatch (artifact %r, engine %r)"
               % (key, artifact, engine))
        if hint:
            msg += ". " + hint
        super().__init__(msg)


class ArtifactPlanError(ArtifactError):
    """The recorded arena plan no longer matches what the graph allocates."""


# --------------------------------------------------------------- environment
def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def board_model():
    """The board model string this plan is valid for.

    ``/proc/device-tree/model`` on a Pi ("Raspberry Pi 3 Model B Rev 1.2");
    ``Model`` from /proc/cpuinfo as a fallback on kernels without a DT model.
    """
    for path in ("/proc/device-tree/model",):
        try:
            with open(path, "rb") as f:
                s = f.read().strip(b"\0").decode("utf-8", "replace").strip()
            if s:
                return s
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine() or "unknown"


def engine_digest(package_dir=None):
    """sha256 over this package's Python sources, by relative path.

    The arena plan and the rewrite tables are decided by the executor, not by
    the kernel generator, so a plan must not survive an executor change.  The
    digest covers every ``.py`` in the package (the optimiser and kgen
    included), which is cheap: ~2 MB once per process.
    """
    if package_dir is None:
        package_dir = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    paths = []
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                full = os.path.join(root, name)
                paths.append((os.path.relpath(full, package_dir), full))
    for rel, full in sorted(paths):
        h.update(b"engine:")
        h.update(rel.encode())
        h.update(b"\0")
        with open(full, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def generator_digest(kgen=None):
    """The blob cache's generator digest, from the library that will launch.

    Falls back to hashing the generator sources exactly as ``vc4_kern.c`` does
    when the installed library has no ``vc4_generator_digest``.
    """
    from . import _binding
    # The C library resolves the generator from $VC4INFER_KGEN when it is set,
    # otherwise from the path baked at build time.  The engine sets the former
    # when it opens; set the same environment here so a build and a load in
    # different trees agree on which sources the digest describes.
    try:
        from . import _runtime
        _runtime.prepare_env()
    except Exception:                          # pragma: no cover
        pass
    got = _binding.generator_digest()
    if got:
        return got
    tag = _binding.kernel_abi_tag() or ABI_TAG_FALLBACK
    if kgen is None:
        from ._runtime import default_kgen
        kgen = default_kgen()
    if not kgen:
        raise ArtifactError(
            "cannot compute the kernel generator digest: libvc4infer has no "
            "vc4_generator_digest and no kgen.py was found")
    h = hashlib.sha256()
    for prefix, path in ((b"kgen:", kgen),
                         (b"kops:", os.path.join(os.path.dirname(kgen),
                                                 "kops.py"))):
        h.update(prefix)
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            h.update(b"absent")
    h.update(tag.encode())
    return h.hexdigest()


def kernel_abi_tag():
    from . import _binding
    return _binding.kernel_abi_tag() or ABI_TAG_FALLBACK


def is_artifact(path):
    """True when the file is a vc4x container (magic, not extension)."""
    try:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


# ------------------------------------------------------------------ panels
def panel_key(key):
    """A stable string for the executor's tuple panel keys.

    ``_panels`` is keyed by tuples such as
    ``("panel", "/model.0/conv/Conv", 16, 16, 3, 3, 1, None)``; the artifact
    stores them as JSON so two runs of the same graph agree even though the
    tuples contain ``None`` and nested tuples.
    """
    return json.dumps(key, default=str, separators=(",", ":"))


class Panel:
    """One prepacked buffer the executor keeps for the life of the Runner."""

    __slots__ = ("key", "panel", "node", "op", "kind", "layout", "format",
                 "shape", "pitch", "alloc_bytes", "bytes", "offset", "data",
                 "sha256")

    def __init__(self, key, data, alloc_bytes=None, panel=None, node="",
                 op="", kind="", layout="", format="buf", shape=None,
                 pitch=None, offset=0, sha256=None):
        self.key = key                    # "panel:0" (allocation key)
        self.panel = panel or ""          # canonical executor panel key
        self.node = node
        self.op = op
        self.kind = kind
        self.layout = layout
        self.format = format              # "buf" or "tensor"
        self.shape = tuple(shape) if shape else None
        self.pitch = pitch
        self.alloc_bytes = int(alloc_bytes if alloc_bytes else len(data))
        self.bytes = int(len(data))
        self.offset = int(offset)         # within the weights section
        self.data = data                  # bytes, or None when file-backed
        self.sha256 = sha256 or _sha256(data)

    def as_dict(self):
        return {
            "key": self.key, "panel": self.panel, "node": self.node,
            "op": self.op, "kind": self.kind, "layout": self.layout,
            "format": self.format,
            "shape": list(self.shape) if self.shape else None,
            "pitch": self.pitch, "offset": self.offset, "bytes": self.bytes,
            "alloc_bytes": self.alloc_bytes, "sha256": self.sha256,
        }


# ------------------------------------------------------------------ artifact
class Artifact:
    """A parsed ``.vc4x`` file (or, during a build, the same thing in memory).

    ``Artifact.load(path)`` reads and validates the header and manifest;
    ``Artifact.open(path)`` additionally checks the keys against this engine
    and board.  ``Engine`` calls the latter.
    """

    def __init__(self, manifest, sections=None, path=None, needed=None,
                 verify=False):
        self.manifest = manifest
        self.path = path
        self._sections = sections or {}       # name -> bytes
        self._needed = needed or {}
        self._graph = None
        self._panel_list = None
        self._alloc_map = None
        # Payload hashes are verified by ``verify()`` (the ``vc4x verify``
        # command) and by ``Artifact.load(verify=True)``.  A load that is about
        # to read 25 MB hashes the same 25 MB otherwise, and the graph is a
        # protobuf that fails its own parse on structural corruption.
        self._verify_sections = bool(verify)

    # -- reading --------------------------------------------------------
    @classmethod
    def load(cls, path, verify=False):
        try:
            with open(path, "rb") as f:
                head = f.read(HEADER.size)
                if len(head) < HEADER.size:
                    raise ArtifactError(
                        "%s: not a vc4x artifact (file is %d B, shorter than "
                        "the %d B header)" % (path, len(head), HEADER.size))
                magic, version, mlen, plen, digest = HEADER.unpack(head)
                if magic != MAGIC:
                    raise ArtifactError(
                        "%s: not a vc4x artifact (magic %r, expected %r)"
                        % (path, magic, MAGIC))
                if version != LAYOUT_VERSION:
                    raise ArtifactMismatch(
                        "format_version", version, LAYOUT_VERSION,
                        "this engine reads vc4x layout %d only"
                        % LAYOUT_VERSION)
                hdr = f.read(mlen)
                if len(hdr) != mlen:
                    raise ArtifactError(
                        "%s: truncated manifest (%d of %d B)" % (path,
                                                                 len(hdr), mlen))
                if _sha256(hdr) != digest.hex():
                    raise ArtifactError(
                        "%s: manifest checksum mismatch: the header is corrupt"
                        % path)
                base = HEADER.size + mlen
                total = os.fstat(f.fileno()).st_size
                if base + plen != total:
                    raise ArtifactError(
                        "%s: payload is %d B, the header says %d B"
                        % (path, total - base, plen))
                manifest = json.loads(hdr.decode("utf-8"))
                # The header carries the layout version the reader dispatches
                # on; a manifest that claims another one describes a different
                # layout and cannot be read as this one.
                for key in ("format_version", "layout_version"):
                    if key in manifest and int(manifest[key]) != version:
                        raise ArtifactMismatch(
                            key, manifest[key], version,
                            "the manifest disagrees with the container header")
                needed = {}
                for s in manifest.get("sections", []):
                    needed[s["name"]] = (base + int(s["offset"]),
                                         int(s["bytes"]), s.get("sha256"))
        except OSError as e:
            raise ArtifactError("%s: %s" % (path, e))
        art = cls(manifest, {}, path=path, needed=needed, verify=verify)
        if verify:
            art.verify()
        return art

    def _section(self, name):
        if name in self._sections:
            return self._sections[name]
        info = self._needed.get(name)
        if info is None:
            raise ArtifactError("%s: no section %r in the artifact"
                                % (self.path, name))
        off, n, digest = info
        try:
            with open(self.path, "rb") as f:
                f.seek(off)
                data = f.read(n)
        except OSError as e:
            raise ArtifactError("%s: %s" % (self.path, e))
        if len(data) != n:
            raise ArtifactError("%s: section %r is truncated (%d of %d B)"
                                % (self.path, name, len(data), n))
        if digest and self._verify_sections and _sha256(data) != digest:
            raise ArtifactError(
                "%s: section %r fails its sha256: the artifact is corrupt"
                % (self.path, name))
        self._sections[name] = data
        return data

    def verify(self):
        """Read and hash every section. Returns a list of (name, bytes)."""
        self._verify_sections = True
        self._sections.clear()
        out = []
        for s in self.manifest.get("sections", []):
            data = self._section(s["name"])
            out.append((s["name"], len(data)))
        return out

    # -- keys -----------------------------------------------------------
    @property
    def keys(self):
        return self.manifest.get("keys", {})

    def validate(self, *, board=None, arena=None, strict_board=True):
        """Refuse the artifact unless every key matches this engine.

        ``arena`` is the client arena size the load will use; a plan whose
        pool plus scratch does not fit is refused here, with the numbers,
        rather than half-way through the first run.
        """
        k = self.keys
        want = {
            "format_version": LAYOUT_VERSION,
            "layout_version": LAYOUT_VERSION,
            "abi_tag": kernel_abi_tag(),
            "generator_digest": generator_digest(),
            "engine_digest": engine_digest(),
            "board": board if board is not None else board_model(),
        }
        hints = {
            "format_version": "rebuild the artifact with this engine",
            "layout_version": "the artifact layout changed; rebuild it",
            "abi_tag": "the kernel ABI changed; rebuild the artifact",
            "generator_digest":
                "the kernel generator changed, so the packed weights and the "
                "blob list may describe different kernels; rebuild the "
                "artifact",
            "engine_digest":
                "the engine's plan/rewrite code changed; rebuild the artifact",
            "board":
                "the artifact is hardware-specific: its arena plan and packed "
                "weights were measured on another board",
        }
        order = ("format_version", "layout_version", "abi_tag",
                 "generator_digest", "engine_digest", "board")
        for key in order:
            got = k.get(key)
            if key in ("format_version", "layout_version"):
                got = self.manifest.get(key)
            if got is None:
                raise ArtifactMismatch(key, "<absent>", want[key], hints[key])
            if strict_board or key != "board":
                if got != want[key]:
                    raise ArtifactMismatch(key, got, want[key], hints[key])
        plan = self.manifest.get("plan", {})
        if not plan.get("allocations"):
            raise ArtifactError(
                "%s: the plan carries no allocations, so it cannot pin a run; "
                "rebuild the artifact" % (self.path or "<memory>"))
        need = int(plan.get("pool_bytes", 0)) + int(plan.get("scratch_bytes", 0))
        if arena is not None and need > int(arena):
            raise ArtifactMismatch(
                "arena", need, int(arena),
                "the plan needs %d B of pool and scratch; the client arena is "
                "%d B (start vc4d with a larger --arena-mb)"
                % (need, int(arena)))
        return True

    def check_arena(self, arena):
        """Refuse a plan that cannot fit the arena this load was given."""
        plan = self.manifest.get("plan", {})
        need = (int(plan.get("pool_bytes", 0)) +
                int(plan.get("scratch_bytes", 0)))
        if need > int(arena):
            raise ArtifactMismatch(
                "arena", need, int(arena),
                "the plan needs %d B of pool and scratch; the client arena is "
                "%d B (start vc4d with a larger --arena-mb)"
                % (need, int(arena)))
        return True

    # -- graph ----------------------------------------------------------
    @property
    def graph_inline(self):
        return bool(self.manifest.get("model", {}).get("graph_inline", True))

    def graph_bytes(self):
        m = self.manifest.get("model", {})
        if m.get("graph_inline", True):
            data = self._section(m.get("graph_section", "graph"))
            want = m.get("graph_sha256")
            if want and _sha256(data) != want:
                raise ArtifactError(
                    "%s: the embedded graph fails its sha256: the artifact is "
                    "corrupt" % self.path)
            return data
        path = m.get("graph_path")
        if not path:
            raise ArtifactError("%s: the artifact references a graph but names "
                                "no path" % self.path)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            raise ArtifactError(
                "%s: the referenced ONNX graph %s is not readable (%s); the "
                "artifact holds a reference, not a copy"
                % (self.path, path, e))
        want = m.get("graph_sha256")
        if want and _sha256(data) != want:
            raise ArtifactError(
                "%s: the referenced graph %s has changed since the artifact "
                "was built (sha256 mismatch)" % (self.path, path))
        return data

    def model(self):
        """The optimised graph as an ``onnx.ModelProto`` (parsed once)."""
        if self._graph is None:
            import onnx
            self._graph = onnx.load_from_string(self.graph_bytes())
        return self._graph

    # -- plan -----------------------------------------------------------
    @property
    def pool_bytes(self):
        return int(self.manifest["plan"]["pool_bytes"])

    @property
    def segment_bytes(self):
        return [int(v) for v in self.manifest["plan"].get("segment_bytes", [])]

    @property
    def scratch_bytes(self):
        return int(self.manifest["plan"].get("scratch_bytes", 0))

    @property
    def pool_plan(self):
        p = dict(self.manifest.get("plan", {}))
        out = {
            "pool_bytes": int(p.get("pool_bytes", 0)),
            "estimate_bytes": int(p.get("estimate_bytes",
                                        p.get("pool_bytes", 0))),
            "plan_bytes": int(p.get("plan_bytes",
                                    p.get("pool_bytes", 0))),
            "headroom_bytes": int(p.get("headroom_bytes", 0)),
            "floor_bytes": int(p.get("floor_bytes", 0)),
            "scratch_bytes": int(p.get("scratch_bytes", 0)),
            "arena_bytes": int(p.get("arena_bytes", 0)),
            "clamped": bool(p.get("clamped", False)),
            "permanent_bytes": int(p.get("permanent_bytes", 0)),
            "temp_bytes": int(p.get("temp_bytes", 0)),
            "activation_bytes": int(p.get("activation_bytes", 0)),
            "reused_bytes": int(p.get("reused_bytes", 0)),
            "margin_bytes": int(p.get("margin_bytes", 0)),
            "memory": p.get("memory") or {},
            "artifact": True,
        }
        return out

    @property
    def memory_plan(self):
        """The plan-time liveness table recorded at build, as a dict.

        Empty for an artifact built before the liveness pass; the loader then
        uses the recorded allocation table and the recorded pool size.
        """
        return dict(self.manifest.get("plan", {}).get("memory") or {})

    @property
    def shapes(self):
        return {name: tuple(dims) for name, dims in
                self.manifest.get("shapes", {}).items()}

    @property
    def allocations(self):
        """key -> (segment index, relative offset, aligned bytes)."""
        if self._alloc_map is None:
            self._alloc_map = {
                e["key"]: (int(e["seg"]), int(e["offset"]), int(e["bytes"]))
                for e in self.manifest.get("plan", {}).get("allocations", [])}
        return self._alloc_map

    def frozen_plan(self):
        """(manifest block, op bytes) for the frozen execution plan, or None.

        The block describes records over segment-relative offsets; the loader
        rebases them onto the segments it received.  No plan is an old
        artifact, not an error: the Python path is what it ran before.
        """
        fr = self.manifest.get("plan", {}).get("frozen")
        if not fr:
            return None
        return fr, self._section(fr.get("section", "plan_ops"))

    @property
    def optimise_report(self):
        return _OptimiseReport(self.manifest.get("model", {}))

    @property
    def plan_folds(self):
        """ConstantOfShape nodes folded into initializers before planning."""
        return list(self.manifest.get("model", {}).get("plan_folds", []))

    @property
    def exec_state(self):
        """The executor's rewrite tables as the build recorded them."""
        st = self.manifest.get("exec")
        if not st:
            return None
        out = {}
        for key in ("silu_skip", "skip"):
            out[key] = set(int(v) for v in st.get(key, []))
        for key in ("silu_src", "fused", "fused_skip", "alias"):
            d = {}
            for k, v in (st.get(key) or {}).items():
                d[int(k)] = (tuple(v) if key == "alias" else v)
            out[key] = d
        out["skip_reason"] = {int(k): v
                              for k, v in (st.get("skip_reason") or {}).items()}
        return out

    @property
    def dropped(self):
        """Initializers whose bytes the artifact carries only as panels."""
        return {e["name"]: tuple(e["shape"]) for e in
                self.manifest.get("weights", {}).get("dropped", [])}

    # -- panels ---------------------------------------------------------
    @property
    def panels(self):
        if self._panel_list is None:
            entries = self.manifest.get("weights", {}).get("entries", [])
            blob = None
            sec = self.manifest.get("weights", {}).get("section")
            if entries and sec:
                blob = self._section(sec)
            out = []
            for e in entries:
                data = None
                if blob is not None:
                    off = int(e["offset"])
                    data = memoryview(blob)[off:off + int(e["bytes"])]
                out.append(Panel(
                    e["key"], data, alloc_bytes=e.get("alloc_bytes"),
                    panel=e.get("panel"), node=e.get("node", ""),
                    op=e.get("op", ""), kind=e.get("kind", ""),
                    layout=e.get("layout", ""), format=e.get("format", "buf"),
                    shape=e.get("shape"), pitch=e.get("pitch"),
                    offset=e.get("offset", 0), sha256=e.get("sha256")))
            self._panel_list = out
        return self._panel_list

    # -- kernels --------------------------------------------------------
    @property
    def kernels(self):
        return self.manifest.get("kernels", {})

    def kernel_blobs(self):
        """``{name: bytes}`` when the blobs were embedded, else ``{}``."""
        k = self.kernels
        entries = k.get("entries", [])
        if not entries or not k.get("embedded"):
            return {}
        blob = self._section(k.get("section", "kernels"))
        return {e["name"]: blob[int(e["offset"]):int(e["offset"]) +
                                int(e["bytes"])] for e in entries}

    @property
    def tuning(self):
        return self.manifest.get("tuning", {})

    # -- summary --------------------------------------------------------
    def summary(self):
        m = self.manifest
        plan = m.get("plan", {})
        return {
            "path": self.path,
            "format": m.get("format"),
            "layout_version": m.get("layout_version"),
            "created": m.get("created"),
            "board": self.keys.get("board"),
            "abi_tag": self.keys.get("abi_tag"),
            "generator_digest": self.keys.get("generator_digest"),
            "engine_digest": self.keys.get("engine_digest"),
            "model": m.get("model", {}).get("name"),
            "opt": m.get("model", {}).get("opt"),
            "passes": [p.get("key") for p in
                       m.get("model", {}).get("passes", [])],
            "nodes": m.get("model", {}).get("nodes"),
            "weights": len(m.get("weights", {}).get("entries", [])),
            "weight_bytes": int(m.get("weights", {}).get("bytes", 0)),
            "kernels": len(m.get("kernels", {}).get("entries", [])),
            "kernel_bytes": int(m.get("kernels", {}).get("bytes", 0)),
            "frozen_plan": (None if not plan.get("frozen") else {
                "version": plan["frozen"].get("version"),
                "c_ops": plan["frozen"].get("c_ops"),
                "c_nodes": plan["frozen"].get("c_nodes"),
                "py_nodes": len(plan["frozen"].get("py_nodes") or []),
                "launches": plan["frozen"].get("launches"),
                "bytes": plan["frozen"].get("bytes"),
            }),
            "pool_bytes": int(plan.get("pool_bytes", 0)),
            "plan_bytes": int(plan.get("plan_bytes",
                                       plan.get("pool_bytes", 0))),
            "headroom_bytes": int(plan.get("headroom_bytes", 0)),
            "permanent_bytes": int(plan.get("permanent_bytes", 0)),
            "temp_bytes": int(plan.get("temp_bytes", 0)),
            "activation_bytes": int(plan.get("activation_bytes", 0)),
            "reused_bytes": int(plan.get("reused_bytes", 0)),
            "margin_bytes": int(plan.get("margin_bytes", 0)),
            "scratch_bytes": int(plan.get("scratch_bytes", 0)),
            "file_bytes": (os.path.getsize(self.path)
                           if self.path and os.path.exists(self.path) else None),
        }

    def __repr__(self):
        m = self.manifest
        return ("Artifact(%r, %s nodes, %d panels, %d kernels)"
                % (self.path, m.get("model", {}).get("nodes"),
                   len(m.get("weights", {}).get("entries", [])),
                   len(m.get("kernels", {}).get("entries", []))))


class _OptimiseReport:
    """The optimiser report as recorded, with the attributes callers use."""

    def __init__(self, model):
        self.enabled = list(model.get("opt_keys", []))
        self.results = list(model.get("passes", []))
        self.artifact = True

    @property
    def changed(self):
        return any(r.get("before") != r.get("after") for r in self.results)

    def summary(self):
        return {"enabled": list(self.enabled),
                "passes": [{"key": r.get("key"), "before": r.get("before"),
                            "after": r.get("after")} for r in self.results]}


# --------------------------------------------------------------- writing
def _sections_from(parts):
    """parts: [(name, bytes)] -> (payload, table)."""
    payload = bytearray()
    table = []
    for name, blob in parts:
        off = len(payload)
        payload += blob
        table.append({"name": name, "offset": off, "bytes": len(blob),
                      "sha256": _sha256(blob)})
    return bytes(payload), table


def write(path, manifest, parts):
    """Write the container atomically; returns (bytes, sha256)."""
    payload, table = _sections_from(parts)
    manifest = dict(manifest)
    manifest["format"] = FORMAT
    manifest["format_version"] = LAYOUT_VERSION
    manifest["layout_version"] = LAYOUT_VERSION
    manifest["sections"] = table
    hdr = json.dumps(manifest, sort_keys=True).encode("ascii")
    head = HEADER.pack(MAGIC, LAYOUT_VERSION, len(hdr), len(payload),
                       hashlib.sha256(hdr).digest())
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(head)
        f.write(hdr)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return os.path.getsize(path), _sha256(head + hdr + payload)


def seed_cache(artifact, cache_dir, *, force=False):
    """Write the artifact's embedded kernel blobs into ``cache_dir``.

    Only files that are absent, or that fail the recorded sha256, are written;
    the existing blob cache (which stamps its own generator digest) then loads
    them without generating anything.  Returns ``(written, present)``.
    """
    blobs = artifact.kernel_blobs()
    if not blobs:
        return 0, 0
    os.makedirs(cache_dir, exist_ok=True)
    stamps = {e["name"]: e.get("stamp") for e in
              artifact.kernels.get("entries", [])}
    written = present = 0
    for name, data in blobs.items():
        dst = os.path.join(cache_dir, name + ".bin")
        stamp = stamps.get(name)
        if not force and os.path.exists(dst) and stamp:
            try:
                with open(dst + ".stamp", encoding="ascii") as f:
                    have = f.read().strip()
                if have == stamp and os.path.getsize(dst) == len(data):
                    present += 1          # the cache will accept it as is
                    continue
            except OSError:
                pass
        with open(dst, "wb") as f:
            f.write(data)
        if stamp:
            # The per-kernel digest the cache compares.  Without it the blob
            # is present but the cache regenerates it, which is exactly the
            # work an embedded blob exists to avoid.
            with open(dst + ".stamp", "w", encoding="ascii") as f:
                f.write(stamp + "\n")
        written += 1
    return written, present


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------- build
class _Recipe:
    """What a build knows before it has a plan: shapes and packed panels.

    The loader replays the recorded plan; build pass B runs the same code path
    with the panels already packed and *records* the layout instead of pinning
    it, so the numbers a build writes are the numbers a load will allocate.
    """

    def __init__(self, shapes, panels, optimise_report=None, dropped=None,
                 exec_state=None, plan_folds=None):
        self.shapes = shapes
        self.panels = panels
        self.optimise_report = optimise_report
        self.dropped = dropped or {}
        self.exec_state = exec_state
        self.plan_folds = plan_folds or []
        self.allocations = None          # record, do not pin
        self.pool_plan = None
        self.pool_bytes = 0
        self.segment_bytes = []


def export_pt(pt_path, imgsz=None, opset=17):
    """Export a .pt through Ultralytics and return (onnx_path, record)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ArtifactError(
            "cannot export %s: Ultralytics (and torch) are not importable "
            "(%s).  Export it elsewhere with "
            "`yolo export model=%s format=onnx imgsz=%s opset=%s` and pass the "
            ".onnx to build." % (pt_path, e, pt_path, imgsz or 640, opset))
    size = int(imgsz or 640)
    model = YOLO(pt_path)
    out = model.export(format="onnx", imgsz=size, opset=int(opset),
                       simplify=False, dynamic=False, batch=1,
                       device="cpu")
    if not out or not os.path.isfile(out):
        raise ArtifactError("Ultralytics did not produce an ONNX file for %s"
                            % pt_path)
    import ultralytics
    return os.path.abspath(out), {
        "from_pt": True, "imgsz": size, "opset": int(opset),
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "pt": os.path.abspath(pt_path),
    }


def _panel_kind(key):
    kind = key[0] if isinstance(key, tuple) and key else ""
    return {
        "panel": "conv-weight-panel",
        "panelg": "grouped-conv-weight-panel",
        "expand": "broadcast-constant",
        "gemmT": "gemm-operand", "gemmB": "gemm-operand",
        "gemmC": "gemm-operand",
    }.get(kind, "table")


def _panel_layout(kind):
    if kind in ("conv-weight-panel", "grouped-conv-weight-panel"):
        return "k-major, tap-major (k = tap*Cin + ci), 64 B rows"
    if kind == "broadcast-constant":
        return "plan-time broadcast of a graph constant"
    if kind == "gemm-operand":
        return "GEMM operand copy"
    return "constant table"


def _collect_panels(runner, eng):
    import numpy as np
    from ._onnxrun import Tensor
    from ._binding import Buf
    from .artifact import panel_key
    out = []
    for i, (key, val) in enumerate(runner._panels.items()):
        if isinstance(val, Tensor):
            buf, fmt = val.buf, "tensor"
        elif isinstance(val, Buf):
            buf, fmt = val, "buf"
        else:
            continue          # a host-side copy of a constant, not in the arena
        raw = eng.download(buf, int(buf.bytes))
        data = np.ascontiguousarray(raw, dtype=np.float32).tobytes()
        kind = _panel_kind(key)
        node = key[1] if isinstance(key, tuple) and len(key) > 1 else ""
        op = ""
        if isinstance(node, str):
            for n in runner.graph.node:
                if n.name == node:
                    op = n.op_type
                    break
        out.append(Panel("panel:%d" % i, data, alloc_bytes=int(buf.bytes),
                         panel=panel_key(key), node=str(node), op=op,
                         kind=kind, layout=_panel_layout(kind), format=fmt,
                         shape=list(val.shape) if isinstance(val, Tensor) else None,
                         pitch=val.pitch if isinstance(val, Tensor) else None))
    return out


def _exec_state(runner):
    """The rewrite tables ``_find_silu``/``_collect_fused`` would rebuild."""
    return {
        "silu_skip": sorted(int(v) for v in runner._silu_skip),
        "silu_src": {str(k): v for k, v in runner._silu_src.items()},
        "skip": sorted(int(v) for v in runner._skip),
        "skip_reason": {str(k): v for k, v in runner._skip_reason.items()},
        "fused": {str(k): v for k, v in runner._fused.items()},
        "fused_skip": {str(k): v for k, v in runner._fused_skip.items()},
        "alias": {str(k): list(v) for k, v in runner._alias.items()},
    }


def _dropped_initializers(model, panels, host):
    """Conv initializers whose only use is a panel that is in the artifact.

    A conv weight is read on the host to *build* its panel and never again:
    the NHWC path passes the packed panel to the kernel, and the host copy is
    what ``_conv_impl`` reads.  Uploading the raw tensor as well copies the
    model a second time into the arena (12.4 MiB of YOLOv8n's 12.4 MiB of
    constants), so an initializer that feeds only panel-backed convs is
    listed here and is not uploaded on load.
    """
    import numpy as np
    from onnx import numpy_helper
    panel_names = set()
    for key in panels:
        if isinstance(key, tuple) and len(key) > 1 and isinstance(key[1], str):
            if key[0] in ("panel", "panelg"):
                panel_names.add(key[1])
    outputs = {o.name for o in model.graph.output}
    inputs = {i.name for i in model.graph.input}
    users = {}
    for node in model.graph.node:
        for pos, name in enumerate(node.input):
            if name:
                users.setdefault(name, []).append((node, pos))
    out = []
    for init in model.graph.initializer:
        if init.data_type != 1 or init.name not in host:     # fp32 only
            continue
        if init.name in outputs or init.name in inputs:
            continue          # the value itself escapes; keep its bytes
        use = users.get(init.name) or []
        if not use:
            continue
        ok = True
        for node, pos in use:
            if node.op_type != "Conv" or node.name not in panel_names:
                ok = False
                break
            if pos not in (1, 2):        # weight or bias only
                ok = False
                break
        if not ok:
            continue
        arr = np.asarray(host[init.name])
        if arr.size == 0:
            continue
        out.append({"name": init.name, "shape": list(arr.shape),
                    "op": "Conv"})
    return out


def _kernel_record(eng, cache_dir):
    """Names, bytes and cache stamps of every blob the build loaded.

    The ``.stamp`` sidecar is the per-kernel digest the cache compares before
    it runs a blob (sha256 of the generator hash and the kernel's argument
    vector).  An embedded blob has to carry it: the cache does not accept a
    file without one, and a wrong stamp makes it regenerate the blob instead
    of reading the artifact's copy.
    """
    names = [n for n in eng.loaded_kernels() if n]
    entries, blobs = [], {}
    for name in names:
        path = os.path.join(cache_dir, name + ".bin")
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            entries.append({"name": name, "bytes": 0, "sha256": None,
                            "stamp": None})
            continue
        blobs[name] = data
        stamp = None
        try:
            with open(path + ".stamp", encoding="ascii") as f:
                text = f.read().strip()
            if len(text) == 64 and all(c in "0123456789abcdef" for c in text):
                stamp = text
        except OSError:
            pass
        entries.append({"name": name, "bytes": len(data),
                        "sha256": _sha256(data), "stamp": stamp})
    return entries, blobs


def build(model_path, out_path, *, opt="all", imgsz=None, opset=17,
          embed_graph=True, embed_kernels=True, cache_dir=None,
          arena_mb=None, verify_output=True, log=None):
    """Build ``out_path`` (a ``.vc4x``) from a fixed-shape ONNX or .pt model.

    Returns a dict with the artifact summary and the build timings.  ``opt`` is
    the pass spec the graph is built with (``"all"`` by default, so the
    artifact is the model the optimiser produces); its provenance records are
    kept in the embedded graph and its plan is what the loader skips redoing.

    The build runs the model twice on the device: once to produce the packed
    panels, once in the load code path to record the arena plan.  Both runs are
    needed -- the first discovers the panels, the second records offsets in the
    layout a fresh load will allocate.
    """
    import numpy as np
    import onnx
    from . import _runtime, optimizer
    from ._onnxrun import Runner

    t_start = time.perf_counter()
    say = log or (lambda *a: None)
    src = os.path.abspath(model_path)
    export = {"from_pt": False}
    if src.lower().endswith(".pt"):
        say("exporting %s through Ultralytics" % src)
        src, export = export_pt(src, imgsz=imgsz, opset=opset)
        say("exported %s (imgsz %d, opset %d)" % (src, export["imgsz"],
                                                  export["opset"]))
    if not os.path.isfile(src):
        raise ArtifactError("no such model: %s" % src)
    source_sha = _sha256_file(src)
    model = onnx.load(src)
    t0 = time.perf_counter()
    res = optimizer.optimise(model, passes=opt)
    opt_s = time.perf_counter() - t0
    if res.changed:
        model = res.model
    say("optimise: %.0f ms, %d -> %d nodes"
        % (opt_s * 1e3, res.results[0].before["nodes"] if res.results else
           len(model.graph.node), len(model.graph.node)))

    # --- pass A: run once with the normal planner, collect the panels -------
    cache = cache_dir or _runtime.default_cache_dir()
    t0 = time.perf_counter()
    eng = _runtime.open_engine(cache_dir=cache, arena_mb=arena_mb)
    try:
        # The executor folds a plan-time ConstantOfShape into an initialiser
        # before it sizes the pool (optimizer.passes.fold_plan_constants).  Do
        # it here, before the graph is written and the plan recorded, so the
        # artifact carries the graph that will actually execute; the fold can
        # also drop inputs that only declared an initialiser, so the feed set
        # is derived afterwards.
        from .optimizer import fold_plan_constants
        model, plan_folds = fold_plan_constants(
            model, budget=int(eng.arena_bytes), limit=int(eng.arena_bytes))
        if plan_folds:
            say("plan-folded %d constant fill(s)" % len(plan_folds))
        graph_bytes = model.SerializeToString()

        float_inputs = []
        for inp in model.graph.input:
            tt = inp.type.tensor_type
            if tt.elem_type != onnx.TensorProto.FLOAT:
                continue
            dims = [d.dim_value if d.HasField("dim_value") else -1
                    for d in tt.shape.dim]
            if any(int(d) <= 0 for d in dims):
                raise ArtifactError(
                    "build needs a fixed-shape graph; input %r is %r"
                    % (inp.name, dims))
            float_inputs.append((inp.name, dims))
        rng = np.random.default_rng(0)
        feeds = {name: rng.standard_normal(dims).astype(np.float32)
                 for name, dims in float_inputs}
        outs = [{"name": o.name,
                 "shape": [d.dim_value if d.HasField("dim_value") else -1
                           for d in o.type.tensor_type.shape.dim]}
                for o in model.graph.output]

        r = Runner(src, eng, passes="none", model=model)
        r.run(feeds)
        panels, shapes = _collect_panels(r, eng), dict(r.shapes)
        exec_state = _exec_state(r)
        dropped = _dropped_initializers(model, r._panels, r.host)
        names = eng.loaded_kernels()
        reports = _pass_records(res)
    finally:
        eng.close()
    t_a = time.perf_counter() - t0
    say("pass A: %.1f s, %d panels, %d kernels, %d shapes"
        % (t_a, len(panels), len(names), len(shapes)))

    # --- pass B: the load code path, with the panels already packed ---------
    recip = _Recipe(shapes, panels, optimise_report=res,
                    dropped={e["name"]: tuple(e["shape"]) for e in dropped},
                    exec_state=exec_state, plan_folds=plan_folds)
    rec = []
    t0 = time.perf_counter()
    eng = _runtime.open_engine(cache_dir=cache, arena_mb=arena_mb)
    try:
        r2 = Runner(src, eng, passes="none", model=model, artifact=recip,
                    record=rec)
        mark0 = len(rec)                 # constants and preseeded panels
        r2.run(feeds)
        mark1 = len(rec)
        s0 = eng.stats
        # The frozen plan is recorded from the frame whose allocation offsets
        # the artifact pins: the second run, not the first (the first frame's
        # offsets come from a clean free list and are not in the plan below).
        r2.begin_plan_capture()
        out_b = r2.run(feeds)
        s1 = eng.stats
        try:
            frozen, frozen_segs = r2.end_plan_capture()
        except _plan.FrozenPlanError as e:
            raise ArtifactError(
                "the frozen execution plan cannot be recorded from this "
                "build: %s" % e)
        # Init-time entries plus the steady-state run: the first run's
        # activation offsets are reachable only from a clean free list and are
        # not part of the plan.
        entries = rec[:mark0] + rec[mark1:]
        run1_keys = {e["key"] for e in rec[mark0:mark1]}
        launches_b = int(s1.launches - s0.launches)
        kernels, blobs = _kernel_record(eng, cache)
        frozen_block = frozen_ops = None
        if frozen is not None:
            frozen_block, frozen_ops = _plan.frozen_plan(
                frozen, frozen_segs, _plan.op_abi(eng),
                _plan.kernel_digest(kernels), launches_b)
        segment_bytes = [int(seg[1]) for seg in r2.pool.segs]
        pool_bytes = int(r2.pool.size)
        pool_plan = dict(r2.pool_plan)
        high = int(r2.pool.high)
    finally:
        eng.close()
    t_b = time.perf_counter() - t0

    plan = _plan_from(entries)
    missing = run1_keys - {e["key"] for e in plan}
    recorded_end = int(max((e["offset"] + e["bytes"] for e in plan),
                           default=0))
    # The artifact does not upload the raw weights a packed panel replaces, so
    # its recorded plan is computed over the graph without them.  The build's
    # own run was sized with them (a safe first segment); with one segment the
    # pool can now shrink to the furthest byte the run actually used.
    from .optimizer.memory import plan_memory
    mem_report = plan_memory(model, shapes,
                             dropped=[e["name"] for e in dropped])
    plan_peak = int(mem_report.peak_bytes)
    if len(segment_bytes) == 1:
        pooled = max((recorded_end + 63) // 64 * 64, plan_peak)
        if pooled != pool_bytes:
            pool_bytes = pooled
            segment_bytes = [pool_bytes]
    say("pass B: %.1f s, %d allocations, pool %.1f MiB"
        % (t_b, len(plan), pool_bytes / 2 ** 20))
    say("memory plan: peak %.2f MiB, recorded %.2f MiB, %+.2f MiB margin%s"
        % (plan_peak / 2 ** 20, recorded_end / 2 ** 20,
           (recorded_end - plan_peak) / 2 ** 20,
           "" if len(segment_bytes) == 1 else " (the pool grew a segment)"))

    # --- assemble the artifact ---------------------------------------------
    entries = []
    payload = bytearray()
    for p in panels:
        p.offset = len(payload)
        payload += p.data
        entries.append(p.as_dict())
    parts = []
    if embed_graph:
        parts.append(("graph", graph_bytes))
    else:
        ref = out_path + ".onnx"
        with open(ref, "wb") as f:
            f.write(graph_bytes)
    if entries:
        parts.append(("weights", bytes(payload)))
    if frozen_ops:
        parts.append(("plan_ops", frozen_ops))
    kern_entries, kern_payload = [], bytearray()
    if embed_kernels and kernels:
        for e in kernels:
            if e["sha256"] is None:
                raise ArtifactError(
                    "kernel blob %s is not in %s, so it cannot be embedded; "
                    "run with --no-embed-kernels to record names only"
                    % (e["name"], cache))
            blob = blobs[e["name"]]
            e = dict(e, offset=len(kern_payload))
            kern_payload += blob
            kern_entries.append(e)
        parts.append(("kernels", bytes(kern_payload)))
    else:
        kern_entries = [dict(e, offset=0) for e in kernels]

    manifest = {
        "format": FORMAT,
        "format_version": LAYOUT_VERSION,
        "layout_version": LAYOUT_VERSION,
        "created": _now(),
        "keys": {
            "generator_digest": generator_digest(),
            "abi_tag": kernel_abi_tag(),
            "engine_digest": engine_digest(),
            "board": board_model(),
            "arch": platform.machine(),
            "engine_version": _engine_version(),
        },
        "model": {
            "name": os.path.basename(src),
            "source": src,
            "source_sha256": source_sha,
            "graph_sha256": _sha256(graph_bytes),
            "graph_inline": bool(embed_graph),
            "graph_section": "graph",
            "graph_path": None if embed_graph else out_path + ".onnx",
            "export": export,
            "opt": str(opt),
            "opt_keys": [p.key for p in res.enabled],
            "plan_folds": list(plan_folds),
            "passes": reports,
            "nodes": len(model.graph.node),
            "inputs": [{"name": n, "shape": d, "dtype": "float32"}
                       for n, d in float_inputs],
            "outputs": outs,
        },
        "shapes": {k: list(v) for k, v in shapes.items()},
        "plan": {
            "arena_bytes": int(pool_plan.get("arena_bytes", 0)),
            "pool_bytes": pool_bytes,
            "segment_bytes": segment_bytes,
            "estimate_bytes": int(pool_plan.get("estimate_bytes", pool_bytes)),
            "plan_bytes": plan_peak,
            "headroom_bytes": int(mem_report.headroom_bytes),
            "floor_bytes": int(pool_plan.get("floor_bytes", 0)),
            "scratch_bytes": int(pool_plan.get("scratch_bytes", 0)),
            "clamped": bool(pool_plan.get("clamped", False)),
            "permanent_bytes": int(mem_report.permanent_bytes),
            "temp_bytes": int(mem_report.temp_bytes),
            "activation_bytes": int(mem_report.activation_bytes),
            "reused_bytes": int(mem_report.reused_bytes),
            # the recorded run's furthest byte, against the plan's peak: a
            # positive margin is the per-node scratch the graph could not name
            "margin_bytes": int(recorded_end - plan_peak),
            "high_water_bytes": high,
            "memory": mem_report.as_dict(),
            "allocations": plan,
            # The frozen execution plan, recorded from the same frame: the
            # device calls, their buffers, the Python steps and the output
            # descriptors.  Optional: an artifact built before it, or by a
            # graph with no representable node, carries none.
            "frozen": frozen_block,
        },
        "weights": {
            "section": "weights" if entries else None,
            "bytes": len(payload),
            "entries": entries,
            "dropped": dropped,
        },
        "exec": exec_state,
        "kernels": {
            "generator_digest": generator_digest(),
            "embedded": bool(embed_kernels),
            "section": "kernels" if embed_kernels and kernels else None,
            "bytes": len(kern_payload),
            "entries": kern_entries,
        },
        "tuning": {
            "autotune": "off",
            "records": [],
            "note": ("autotune is off by default; a record here would be the "
                     "per-shape QPU counts vc4_autotune chose, and the loader "
                     "would apply them instead of the default launch"),
        },
    }
    size, digest = write(out_path, manifest, parts)
    out_art = Artifact.load(out_path)
    out_art.validate(arena=None)
    if verify_output:
        out_art.verify()
    # --- pass C: load the file and compare with the recorded run ------------
    # The plan is a recording, and a recording is only worth shipping if a
    # load actually reproduces it: this opens the artifact exactly as a caller
    # would and checks the outputs and the steady-state launch count.
    verify = {"bit_exact": None, "max_abs": None, "launches": None}
    if verify_output:
        from .engine import Engine
        t0 = time.perf_counter()
        try:
            eng3 = Engine(out_path, cache_dir=cache, arena_mb=arena_mb)
            try:
                s0 = eng3.stats
                got = eng3.run(feeds)
                launches_c = int(eng3.stats["launches"])
            finally:
                eng3.close()
            verify["load_s"] = time.perf_counter() - t0
        except Exception:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            raise
        names = [o["name"] for o in outs]
        max_abs = 0.0
        exact = True
        for name, want in zip(names, out_b):
            have = np.asarray(got[name], dtype=np.float32)
            want = np.asarray(want, dtype=np.float32)
            if have.shape != want.shape:
                exact = False
                max_abs = float("inf")
                break
            if not np.array_equal(have, want):
                exact = False
                max_abs = max(max_abs, float(np.max(np.abs(have - want))))
        verify = {"bit_exact": bool(exact), "max_abs": max_abs,
                  "launches": launches_c, "launches_recorded": launches_b,
                  "load_s": verify.get("load_s")}
        if not exact or launches_c != launches_b:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            raise ArtifactError(
                "the artifact does not reproduce the run it recorded: "
                "%s, max abs %g, %d launches vs %d"
                % ("bit-exact" if exact else "outputs differ", max_abs,
                   launches_c, launches_b))
        say("pass C: loaded the artifact, outputs bit-exact, %d launches"
            % launches_c)
    summary = out_art.summary()
    summary["artifact_sha256"] = digest
    summary["verified"] = verify
    summary["missing_first_run_keys"] = len(missing)
    summary["timings"] = {
        "total_s": time.perf_counter() - t_start,
        "optimise_s": opt_s,
        "panels_s": t_a,
        "plan_s": t_b,
    }
    summary["arena_bytes"] = int(pool_plan.get("arena_bytes", 0))
    return summary


def _engine_version():
    try:
        from . import __version__
        return __version__
    except Exception:                          # pragma: no cover
        return "unknown"


def _pass_records(res):
    out = []
    for r in res.results:
        out.append({
            "key": r.key, "title": r.title,
            "before": r.before, "after": r.after,
            "rules": [s.as_dict() for s in r.rules],
            "error": None if r.error is None else str(r.error),
        })
    return out


def _plan_from(entries):
    """One keyed allocation table from the steady-state run.

    The first run in a process allocates from a clean pool; every later run
    starts with the previous run's holes in the free list, so the *offsets* of
    the two runs differ even though the keys and lifetimes are the same.  The
    plan is the steady-state run's table: a load pins those offsets from its
    first run onward, so it never depends on the free list at all.
    """
    plan = {}
    for e in entries:
        k = e["key"]
        if k in plan:
            if (plan[k]["seg"], plan[k]["offset"], plan[k]["bytes"]) != \
                    (e["seg"], e["offset"], e["bytes"]):
                raise ArtifactError(
                    "the arena plan is ambiguous: allocation %r appears twice "
                    "at different offsets in one run (%d+%d vs %d+%d)"
                    % (k, plan[k]["seg"], plan[k]["offset"], e["seg"],
                       e["offset"]))
            continue
        plan[k] = {"key": k, "seg": int(e["seg"]),
                   "offset": int(e["offset"]), "bytes": int(e["bytes"]),
                   "requested": int(e["requested"])}
    return [plan[k] for k in sorted(plan)]

