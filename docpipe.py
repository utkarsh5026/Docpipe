# -*- coding: utf-8 -*-
"""
docpipe -- a single-file document intelligence substrate.
=========================================================

``docpipe`` is the domain-independent 70% of every scanned-document extraction
pipeline: ingest, page-quality measurement, adaptive preprocessing, pluggable
OCR/VLM reading backends, schema-driven extraction with provenance, calibrated
confidence fusion, and an evaluation harness.

Everything lives in this one module on purpose.  Copy ``docpipe.py`` into a
project, or ``pip install docpipe-core`` -- both work, and neither requires a
build step.  (The distribution is ``docpipe-core`` because ``docpipe`` on PyPI
is an unrelated project; the import name here is unaffected.)

Design in one paragraph
-----------------------
A scanned page is a *fact that survived a noisy channel*: printed, photocopied,
faxed, scanned at whatever DPI the clerk's flatbed defaulted to, JPEG'd, and
wrapped in a PDF.  Extraction is channel inversion.  Three things follow.
Preprocessing is *equalisation*, so it must be a function of the page's measured
degradation rather than a fixed sequence.  Confidence is a property of the
channel, not of how fluent the model sounds, so it must fuse independent
evidence.  Provenance is recoverable at read time and unrecoverable afterwards,
so it is threaded through every layer from the start.

Layers (each usable standalone)
-------------------------------
==========  ==================================================================
Layer 5     :class:`EvalSuite` -- datasets, metrics, pipeline A/B, regressions
Layer 4     :func:`extract` -- schema in, typed result with evidence out
Layer 3     :func:`read` -- pluggable, cost-aware OCR / VLM backends
Layer 2     :func:`preprocess` -- composable ``Page -> Page`` ops
Layer 1     :class:`Document` / :class:`Page` / :class:`TextSpan` / :class:`BBox`
Layer 0     :func:`Ingest.ingest` -- PDF, TIFF, image, email attachment
==========  ==================================================================

Cross-cutting: caching, cost accounting, retries, tracing, provenance.

Namespaces
----------
One file does not have to mean one flat heap of names.  Groups of free functions
that share a subject are gathered onto a namespace class as staticmethods, and
only the class is exported:

==================  ======================================================
:class:`Caps`       optional-dependency probing, the OpenCV kill switch
:class:`Util`       hashing, clamping, string distance, JSON coercion
:class:`Image`      raster primitives (arrays in, arrays out)
:class:`Quality`    page-quality measures and estimators
:class:`Ingest`     bytes of unknown provenance to a :class:`Document`
:class:`Ops`        the preprocessing ops, as :class:`Op` factories
:class:`Policies`   measured page to the ops it needs
:class:`Pricing`    token accounting and the optional price table
:class:`Text`       script detection, normalisation, value parsing
:class:`Confidence` signal fusion and calibration metrics
:class:`Validators` domain-independent validator factories
==================  ======================================================

Every one of those staticmethods is *also* a module attribute under its bare
name -- ``dp.to_gray`` and ``dp.Image.to_gray`` are the same object.  The flat
names are not decoration: they are how the staticmethods call one another, and
they keep every existing call site working.  ``Image.to_gray`` is the documented
path; ``to_gray`` is the one that already works.  Classes that carry state or
are meant for subclassing (:class:`Page`, :class:`BaseBackend`,
:class:`EvalSuite`) and the layer entry points (:func:`read`,
:func:`preprocess`, :func:`extract`, :func:`process`) stay where they are.

Quickstart
----------
::

    import docpipe as dp

    doc = dp.Ingest.ingest("claim_47812.pdf")
    doc = dp.preprocess(doc, policy=dp.Policies.default_policy)
    doc = dp.read(doc, router=dp.default_router)

    print(doc.text()[:500])
    print(doc.pages[0].quality)

With a schema::

    from pydantic import BaseModel

    class Bill(BaseModel):
        hospital_name: str
        total: float

    res = dp.extract(doc, schema=Bill, client=dp.AnthropicClient())
    res.fields["total"].value        # 48250.0
    res.fields["total"].confidence   # 0.94  (fused, calibratable)
    res.fields["total"].evidence     # [BBox(page=3, ...)]

Dependencies
------------
The core (IR, normalisation, confidence, extraction plumbing, eval harness) is
pure standard library.  Everything heavier is optional and imported lazily:

==========================  ================================================
``numpy``                   all raster work
``opencv-python``           fast/high-quality image ops (NumPy fallbacks too)
``pymupdf``                 PDF ingest and native text layers
``pillow``                  image and multi-page TIFF ingest
``pytesseract``             Tesseract backend (or the ``tesseract`` binary)
``paddleocr``               PaddleOCR backend
``rapidocr-onnxruntime``    RapidOCR backend (PP-OCR weights, ONNX runtime)
``easyocr``                 EasyOCR backend
``python-doctr``            docTR backend
``surya-ocr``               Surya backend (90+ scripts)
``anthropic`` / ``openai``  vision-language backends and extraction clients
``pydantic``                schema-driven extraction (v1 and v2 supported)
==========================  ================================================

Call :func:`Caps.capabilities` to see what is actually importable right now.

Compatibility
-------------
Python 3.8+.  No ``X | Y`` annotations, no ``match``, no walrus-in-comprehension
cleverness -- this file is meant to be dropped into old codebases.

Annotations are written inline and are never evaluated at runtime, because
``from __future__ import annotations`` (PEP 563) is in force below.  That is
what lets a 3.8-compatible file use modern annotation syntax and forward
references without quoting them.  Aliases such as :data:`ImageArray` and
:data:`Source` name the conventions that the optional-dependency types cannot
express -- see the type-alias block after the imports.

License: MIT.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Utkarsh"
__license__ = "MIT"

import base64
import dataclasses
import datetime as _dt
import decimal
import difflib
import enum
import functools
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

#: ``typing.Protocol`` landed in 3.8, which is this file's floor, so it is
#: imported outright.  The old ``except ImportError: Protocol = object`` fallback
#: was dead code that also cost every structural type its meaning -- a checker
#: reading it cannot tell whether a Protocol base is a class or an instance.
from typing import Protocol

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
# Annotations are written inline, but ``from __future__ import annotations`` (at
# the top of this file) means they are never evaluated at runtime -- so a name
# used only in a signature costs nothing and forward references need no quotes.
#
# Several aliases below resolve to ``Any``.  That is deliberate, not laziness:
# NumPy, PyMuPDF and the OCR/LLM SDKs are all *optional* dependencies, so their
# real types cannot be imported at module import time without defeating the
# point.  The alias carries the contract in its name where the checker cannot.

T = TypeVar("T")
Number = Union[int, float]

#: A raster image: a ``uint8`` NumPy array, ``(H, W)`` grey or ``(H, W, 3)`` RGB.
ImageArray = Any
#: A single-channel ``uint8`` NumPy array, ``(H, W)``.  Ink is dark, paper light.
GrayImage = Any
#: A floating-point NumPy array -- an intermediate result of image maths
#: (projections, integral images, gradient magnitudes) rather than a picture.
FloatArray = Any
#: A lazily imported optional dependency, obtained via :func:`Caps.require`.
Module = Any
#: Anything a document can be ingested from: a filesystem path, raw bytes, or an
#: open binary file object.
Source = Union[str, "os.PathLike", bytes, bytearray, IO[bytes]]
#: A JSON-serialisable mapping, as produced by every ``to_dict`` in this file.
JSONDict = Dict[str, Any]
#: An extraction schema: a pydantic model, a dataclass, or a ``{field: type}``
#: dict spec.  See :class:`SchemaAdapter` for what each form supports.
SchemaLike = Any

logger = logging.getLogger("docpipe")
logger.addHandler(logging.NullHandler())


# =============================================================================
# 1. Optional dependencies
# =============================================================================
#
# Rule: nothing heavy is imported at module import time.  Every optional import
# goes through _try_import, which caches both successes and failures, so a
# missing dependency costs one failed import for the life of the process and
# produces an actionable error message at the point of use -- not a traceback
# from three frames deep inside a backend.


class DocpipeError(Exception):
    """Base class for every error this library raises deliberately."""


class MissingDependency(DocpipeError, ImportError):
    """An optional dependency is required for the requested operation."""

    def __init__(self, module: str, purpose: str = "", extra: str = "") -> None:
        """Build the error, including the ``pip install`` line that fixes it.

        :param module: the import name that failed
        :param purpose: what it was needed for, e.g. "PDF handling"
        :param extra: the pip name, when it differs from the import name
        """
        self.module = module
        self.purpose = purpose
        hint = extra or module
        msg = "docpipe needs '%s'" % module
        if purpose:
            msg += " for %s" % purpose
        msg += ".  Install it with:  pip install %s" % hint
        super().__init__(msg)


class IngestError(DocpipeError):
    """A source document could not be opened, decoded or repaired."""


class BackendError(DocpipeError):
    """A reading backend failed."""


class ExtractionError(DocpipeError):
    """Structured extraction failed (bad model output, bad schema, ...)."""


class ConfigError(DocpipeError):
    """A pipeline was configured with values that cannot work."""


class BudgetExceeded(DocpipeError):
    """A cost budget was exhausted before the document finished."""


_IMPORT_CACHE: Dict[str, Any] = {}
_IMPORT_LOCK = threading.Lock()

#: pip names for modules whose import name differs from their distribution name.
_PIP_NAMES = {
    "cv2": "opencv-python",
    "fitz": "pymupdf",
    "PIL": "pillow",
    "pytesseract": "pytesseract",
    "paddleocr": "paddleocr",
    "rapidocr_onnxruntime": "rapidocr-onnxruntime",
    "easyocr": "easyocr",
    "doctr": "python-doctr[torch]",
    "surya": "surya-ocr",
    "pydantic": "pydantic",
    "numpy": "numpy",
    "anthropic": "anthropic",
    "openai": "openai",
    # The unified Google SDK, not the deprecated `google-generativeai`.  Probed
    # by its dotted name because `google` alone is a namespace package that many
    # unrelated Google libraries populate.
    "google.genai": "google-genai",
    "pypdfium2": "pypdfium2",
}


def _try_import(name: str) -> Optional[Module]:
    """Import ``name``, returning ``None`` if it is unavailable.

    Results (including failures) are cached, so a missing optional dependency is
    only paid for once.
    """
    if name in _IMPORT_CACHE:
        return _IMPORT_CACHE[name]
    with _IMPORT_LOCK:
        if name in _IMPORT_CACHE:
            return _IMPORT_CACHE[name]
        try:
            mod = __import__(name)
            for part in name.split(".")[1:]:
                mod = getattr(mod, part)
        except Exception as exc:  # ImportError, but also broken native libs
            logger.debug("optional import %r failed: %s", name, exc)
            mod = None
        _IMPORT_CACHE[name] = mod
        return mod


class Caps(object):
    """Optional-dependency probing and the OpenCV kill switch.

    Grouped so that a caller can ask one object what is importable right
    now.  :func:`Caps.require` is the only sanctioned way to reach an
    optional import: it raises :class:`MissingDependency` with a pip hint
    at the point of the call, rather than letting an ``ImportError``
    surface three frames deep inside a backend.
    """

    @staticmethod
    def require(name: str, purpose: str = "") -> Module:
        """Import ``name`` or raise :class:`MissingDependency` with a pip hint.

        The gate every optional import goes through, so a missing dependency
        fails where the user called it rather than three frames deep inside a
        backend.

        :param name: the import name, e.g. ``"cv2"``, ``"fitz"``, ``"PIL.Image"``.
            Note that this is the import name, not the pip name -- the mapping
            between them is what produces a correct install hint.
        :param purpose: what it was needed for, e.g. ``"Tesseract input
            conversion"``.  It goes into the error message, so make it name the
            feature the user was reaching for.
        :returns: the imported module
        :raises MissingDependency: not importable; the message carries the pip
            install command
        """
        mod = _try_import(name)
        if mod is None:
            raise MissingDependency(name, purpose, _PIP_NAMES.get(name, name))
        return mod

    @staticmethod
    def have(name: str) -> bool:
        """True when optional module ``name`` can be imported.

        The non-raising counterpart to :func:`Caps.require`, for choosing a code
        path rather than demanding one.  Results are cached, so probing in a hot
        loop is cheap.

        :param name: the import name, e.g. ``"cv2"`` or ``"pytesseract"``
        :returns: whether the import succeeds.  ``False`` also covers a module
            that is installed but broken -- a native library failing to load
            counts as unavailable, which is the useful answer.
        """
        return _try_import(name) is not None

    @staticmethod
    def set_opencv_enabled(enabled: bool) -> bool:
        """Enable/disable OpenCV acceleration globally.  Returns the previous value.

        Process-wide and not thread-safe, so set it during start-up or in a test
        fixture.  For a scoped change prefer ``with Caps.without_opencv():``.
        The ``DOCPIPE_DISABLE_OPENCV=1`` environment variable does the same at
        import time, which is what CI uses to exercise the NumPy fallbacks.

        :param enabled: ``False`` forces the pure-NumPy path even when OpenCV is
            installed.  Results differ slightly between the two backends -- the
            fallbacks are equivalent, not bit-identical.
        :returns: the previous setting, so a caller can restore it
        """
        global _USE_OPENCV
        prev = _USE_OPENCV
        _USE_OPENCV = bool(enabled)
        return prev

    @staticmethod
    def without_opencv() -> _OpenCVDisabled:
        """``with docpipe.without_opencv(): ...`` -- force pure-NumPy image ops."""
        return _OpenCVDisabled()

    @staticmethod
    def capabilities() -> Dict[str, bool]:
        """Report which optional integrations are usable in this interpreter.

        Cheap to call and safe at import time; useful in a health check.
        """
        names = [
            "numpy", "cv2", "PIL", "pypdfium2", "pytesseract", "paddleocr",
            "rapidocr_onnxruntime", "easyocr", "doctr", "surya", "pydantic",
            "anthropic", "openai",
        ]
        caps: Dict[str, bool] = {}
        for n in names:
            caps[n] = have(n)
        caps["pymupdf"] = _fitz() is not None
        caps["tesseract_binary"] = shutil.which("tesseract") is not None
        caps["opencv_enabled"] = bool(caps.get("cv2")) and _USE_OPENCV
        return caps


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Caps from the inside.
# ``Caps.x`` is the documented path; ``x`` is the one that already works.
require            = Caps.require
have               = Caps.have
set_opencv_enabled = Caps.set_opencv_enabled
without_opencv     = Caps.without_opencv
capabilities       = Caps.capabilities


def _np() -> Module:
    """NumPy, or :class:`MissingDependency`.  Used by every raster code path."""
    return require("numpy", "raster image operations")


def _fitz(required: bool = False) -> Optional[Module]:
    """PyMuPDF under either of its import names.

    The project renamed ``fitz`` to ``pymupdf`` and the old name now emits a
    deprecation warning on import, so try the new one first and fall back.
    """
    mod = _try_import("pymupdf") or _try_import("fitz")
    if mod is None and required:
        raise MissingDependency("pymupdf", "PDF handling", "pymupdf")
    return mod


#: Global switch.  Set to False to force the pure-NumPy fallbacks even when
#: OpenCV is installed -- used by the test suite to exercise both paths, and
#: occasionally useful when a wheel's OpenCV misbehaves in a container.
#: ``DOCPIPE_DISABLE_OPENCV=1`` in the environment sets it at import time.
_USE_OPENCV = os.environ.get("DOCPIPE_DISABLE_OPENCV", "").strip().lower() \
    not in ("1", "true", "yes", "on")


def _cv2() -> Optional[Module]:
    """OpenCV if available *and* enabled, else ``None``.

    Image ops call this and branch: ``cv = _cv2(); if cv is not None: ...``.
    """
    if not _USE_OPENCV:
        return None
    return _try_import("cv2")


class _OpenCVDisabled(object):
    """Context manager forcing NumPy fallbacks inside a block."""

    def __enter__(self) -> _OpenCVDisabled:
        """Turn OpenCV off, remembering the previous setting."""
        self._prev = set_opencv_enabled(False)
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        """Restore the previous setting.  Never swallows an exception."""
        set_opencv_enabled(self._prev)
        return False


# =============================================================================
# 2. Small utilities
# =============================================================================


class Util(object):
    """Small, dependency-free helpers used across every layer.

    Nothing here knows about documents.  They live together because they
    are the handful of primitives -- stable hashing, clamping, string
    distance, JSON coercion -- that the rest of the file assumes exist.
    """

    @staticmethod
    def stable_hash(*parts: Any) -> str:
        """A short, stable, cross-process content hash.

        ``hash()`` is salted per process and useless for caching; this is not.
        """
        h = hashlib.sha256()
        for p in parts:
            if p is None:
                h.update(b"\x00none")
            elif isinstance(p, bytes):
                h.update(b"\x01"); h.update(p)
            elif isinstance(p, str):
                h.update(b"\x02"); h.update(p.encode("utf-8", "replace"))
            elif isinstance(p, (int, float, bool)):
                h.update(b"\x03"); h.update(repr(p).encode("ascii"))
            else:
                h.update(b"\x04")
                try:
                    h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
                except Exception:
                    h.update(repr(p).encode("utf-8", "replace"))
        return h.hexdigest()[:32]

    @staticmethod
    def array_hash(arr: ImageArray) -> str:
        """Content hash of a NumPy array, for caching backend reads.

        Hashes the pixels, shape and dtype rather than a filename, which is what
        makes :class:`CachingBackend` correct across preprocessing changes: alter
        an op and the cache misses, as it should.

        :param arr: the array to hash; a non-array falls back to hashing its
            ``repr``
        :returns: a stable hex digest, identical across processes and runs
        """
        try:
            buf = arr.tobytes()
        except AttributeError:
            return stable_hash(repr(arr))
        return stable_hash(buf, str(getattr(arr, "shape", "")), str(getattr(arr, "dtype", "")))

    @staticmethod
    def clamp(value: float, lo: float, hi: float) -> float:
        """Clamp ``value`` into ``[lo, hi]``.

        :param value: the number to bound
        :param lo: lower bound, returned when ``value`` is below it
        :param hi: upper bound, returned when ``value`` is above it
        :returns: the bounded value.  No check that ``lo <= hi``; inverted bounds
            return ``lo``.
        """
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    @staticmethod
    def percentile(values: Sequence[float], q: float) -> float:
        """Linear-interpolated percentile without pulling in NumPy.

        ``q`` is in ``[0, 100]``.  Returns ``0.0`` for an empty sequence, which is
        the convention the metrics code below relies on.

        :param values: the sample, in any order; it is sorted internally
        :param q: the percentile in ``0.0..100.0``, clamped.  ``50`` is the
            median, ``95`` the usual tail metric.
        :returns: the interpolated value, or ``0.0`` for an empty sequence
        """
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * clamp(q, 0.0, 100.0) / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ordered[lo])
        frac = pos - lo
        return float(ordered[lo]) * (1.0 - frac) + float(ordered[hi]) * frac

    @staticmethod
    def levenshtein(a: str, b: str, max_distance: Optional[int] = None) -> int:
        """Edit distance with an optional early-exit bound.

        Iterative two-row DP: O(len(a) * len(b)) time, O(min) space.  When
        ``max_distance`` is given and every cell in a row exceeds it, we bail out
        early and return ``max_distance + 1``.

        :param a: the first string
        :param b: the second string
        :param max_distance: give up once the distance provably exceeds this.
            Turns a "are these nearly equal?" test into an early exit rather than
            a full matrix -- worth passing when scanning many candidates and only
            close ones matter.
        :returns: the edit distance, or ``max_distance + 1`` when the bound was
            hit.  The sentinel means "further than you asked about", not an exact
            distance.
        """
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        if len(a) < len(b):
            a, b = b, a
        if max_distance is not None and abs(len(a) - len(b)) > max_distance:
            return max_distance + 1

        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current = [i]
            best = current[0]
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
                if current[-1] < best:
                    best = current[-1]
            previous = current
            if max_distance is not None and best > max_distance:
                return max_distance + 1
        return previous[-1]

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Normalised similarity in ``[0, 1]``: ``1 - edit_distance / max_len``.

        The comparison behind evidence matching and cross-read agreement.
        Normalising by the longer string keeps a one-character error in a long
        value from scoring as badly as one in a short value.

        :param a: the first string
        :param b: the second string
        :returns: ``1.0`` for identical strings, ``0.0`` when either is empty
            (unless both are), graded in between.  Normalise the inputs first if
            case or punctuation should not count.
        """
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        return 1.0 - levenshtein(a, b) / float(max(len(a), len(b)))

    @staticmethod
    def retry_call(fn: Callable[[], T], attempts: int = 3, base_delay: float = 0.5,
                   max_delay: float = 8.0, jitter: float = 0.25,
                   retry_on: Tuple[Type[BaseException], ...] = (Exception,),
                   give_up_on: Tuple[Type[BaseException], ...] = (),
                   on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
                   sleep: Callable[[float], None] = time.sleep) -> T:
        """Call ``fn`` with exponential backoff and full jitter.

        ``give_up_on`` wins over ``retry_on`` so that non-retryable failures (bad
        API key, malformed request) fail immediately instead of being hammered
        three times and billed three times.  ``on_retry(attempt, exc, delay)`` is
        the hook the cost tracker uses to record failed-but-charged attempts.

        :param fn: the zero-argument callable to attempt
        :param attempts: total tries including the first; must be at least ``1``
        :param base_delay: seconds before the first retry, doubling each time
        :param max_delay: ceiling on the backoff, before jitter
        :param jitter: random fraction of the delay added on top, in ``0.0..1.0``.
            ``0.25`` is enough to break up the thundering herd when many pages
            fail against the same provider at once.
        :param retry_on: exception types worth retrying.  ``(Exception,)`` is
            broad because provider SDKs raise their own hierarchies.
        :param give_up_on: exception types that never become successes.  These
            win over ``retry_on``, so a bad API key fails once, not three times.
        :param on_retry: called as ``on_retry(attempt, exc, delay)`` before each
            sleep -- the hook for counting billed-but-failed attempts.
        :param sleep: the sleep function, injectable so tests do not wait
        :returns: whatever ``fn`` returned on its first success
        :raises ConfigError: ``attempts`` is less than 1
        :raises Exception: the last failure, once every attempt is spent
        """
        if attempts < 1:
            raise ConfigError("attempts must be >= 1")
        last: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except give_up_on:
                raise
            except retry_on as exc:
                last = exc
                if attempt >= attempts:
                    break
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay += random.uniform(0.0, jitter * delay)
                if on_retry is not None:
                    on_retry(attempt, exc, delay)
                logger.warning("attempt %d/%d failed (%s); retrying in %.2fs",
                               attempt, attempts, exc, delay)
                sleep(delay)
        assert last is not None
        raise last

    @staticmethod
    def to_json(obj: Any, indent: Optional[int] = 2, sort_keys: bool = False) -> str:
        """Serialise any docpipe object graph to JSON.

        Handles the types ``json`` will not: dataclasses, enums, ``Decimal``,
        dates and NumPy scalars, by way of each object's ``to_dict``.

        :param obj: any docpipe object, or a container of them
        :param indent: spaces per level; ``None`` produces compact one-line JSON
        :param sort_keys: sort object keys, which makes output diffable -- worth
            setting for anything committed as a baseline
        :returns: the JSON text, with non-ASCII characters left as themselves so
            Devanagari and CJK stay readable in a report
        """
        return json.dumps(_as_jsonable(obj), indent=indent, sort_keys=sort_keys, ensure_ascii=False)


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Util from the inside.
# ``Util.x`` is the documented path; ``x`` is the one that already works.
stable_hash = Util.stable_hash
array_hash  = Util.array_hash
clamp       = Util.clamp
percentile  = Util.percentile
levenshtein = Util.levenshtein
similarity  = Util.similarity
retry_call  = Util.retry_call
to_json     = Util.to_json


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """``a / b``, or ``default`` when ``b`` is zero."""
    return a / b if b else default


class Timer(object):
    """Wall-clock stopwatch.  ``with Timer() as t: ...`` then ``t.ms``."""

    __slots__ = ("start", "end")

    def __init__(self) -> None:
        """Start at zero; the clock only runs between ``__enter__`` and ``__exit__``."""
        self.start = 0.0
        self.end = 0.0

    def __enter__(self) -> Timer:
        """Start the clock."""
        self.start = time.time()
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        """Stop the clock.  Never swallows an exception."""
        self.end = time.time()
        return False

    @property
    def ms(self) -> float:
        """Elapsed milliseconds -- so far, if the block has not exited yet."""
        end = self.end or time.time()
        return (end - self.start) * 1000.0


def chunked(seq: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield ``seq`` in lists of at most ``size``.

    :param seq: the sequence to split
    :param size: maximum items per chunk; the final chunk may be shorter
    :returns: an iterator of lists
    :raises ConfigError: ``size`` is less than 1
    """
    if size < 1:
        raise ConfigError("chunk size must be >= 1")
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def _map_maybe_parallel(fn: Callable[[Any], Any], items: Sequence[Any],
                        max_workers: int = 0, label: str = "work") -> List[Any]:
    """``map`` that goes through a thread pool when ``max_workers > 1``.

    Threads (not processes) because the expensive work is either native code
    that releases the GIL (OpenCV, ONNX runtime) or network I/O (VLM calls),
    and because a Page holding a lazily-rendered raster is not picklable.
    """
    items = list(items)
    if max_workers and max_workers > 1 and len(items) > 1:
        try:
            from concurrent.futures import ThreadPoolExecutor
        except ImportError:  # pragma: no cover
            pass
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(fn, items))
    return [fn(x) for x in items]


def _utcnow() -> str:
    """Current UTC time as a second-resolution ISO-8601 string.

    Reads the clock as an aware UTC datetime and then drops the offset, because
    ``datetime.utcnow()`` is deprecated from 3.12 on.  The ``Z`` suffix is
    appended by hand, so the output is unchanged: ``2026-08-12T09:15:00Z``.
    """
    return _dt.datetime.now(_dt.timezone.utc).replace(
        microsecond=0, tzinfo=None).isoformat() + "Z"


def _as_jsonable(obj: Any) -> Any:
    """Best-effort conversion of nested docpipe/py objects to JSON primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return dict((str(k), _as_jsonable(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return [_as_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _as_jsonable(obj.to_dict())
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            if f.name.startswith("_"):
                continue
            out[f.name] = _as_jsonable(getattr(obj, f.name))
        return out
    # Deliberate duck typing: these three shapes come from optional dependencies
    # that may not be installed, so they are probed rather than imported.
    duck = cast(Any, obj)
    if hasattr(obj, "model_dump"):  # pydantic v2
        try:
            return _as_jsonable(duck.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):  # pydantic v1
        try:
            return _as_jsonable(duck.dict())
        except Exception:
            pass
    if hasattr(obj, "tolist"):  # numpy
        try:
            return duck.tolist()
        except Exception:
            pass
    return str(obj)


# =============================================================================
# 3. Layer 1 -- the intermediate representation
# =============================================================================
#
# This is the actual product.  Backends are swappable because they all speak
# TextSpan; ops compose because they are Page -> Page; two whole pipelines can
# be A/B'd because they share input and output types.  Everything else in this
# file is replaceable around these five types.
#
# Coordinate convention, stated once and obeyed everywhere:
#   * BBox lives in PDF *points* (1/72 inch), top-left origin, y growing down,
#     with page rotation already applied.
#   * Raster arrays are uint8 NumPy, either (H, W) grayscale or (H, W, 3) RGB.
#   * pixels = points * dpi / 72.  Use BBox.to_pixels()/from_pixels().


class PageKind(str, enum.Enum):
    """How the text on a page is stored -- which decides how to read it."""

    DIGITAL_NATIVE = "digital_native"  #: trustworthy embedded text layer
    SCANNED = "scanned"                #: raster only, OCR required
    HYBRID = "hybrid"                  #: native text plus scanned inserts
    BLANK = "blank"                    #: nothing worth reading

    def __str__(self) -> str:
        """The bare value, so formatting a kind never leaks ``PageKind.``."""
        return self.value


class Verdict(str, enum.Enum):
    """Coarse readability judgement produced by :func:`Quality.measure_quality`."""

    CLEAN = "clean"
    DEGRADED = "degraded"
    UNREADABLE = "unreadable"

    def __str__(self) -> str:
        """The bare value, so formatting a verdict never leaks ``Verdict.``."""
        return self.value


class RegionKind(str, enum.Enum):
    """Layout region types worth routing on."""

    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FIGURE = "figure"
    STAMP = "stamp"
    SIGNATURE = "signature"
    HANDWRITING = "handwriting"
    BARCODE = "barcode"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """The bare value, so formatting a kind never leaks ``RegionKind.``."""
        return self.value


@dataclass(frozen=True)
class BBox(object):
    """An axis-aligned rectangle on a specific page, in canonical page points.

    Frozen because bboxes get shared liberally between spans, fields and eval
    records; an accidental in-place mutation would corrupt provenance silently.
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        """Normalise the rectangle so width and height are never negative."""
        # Normalise inverted rectangles rather than rejecting them: several
        # OCR engines emit quads in arbitrary vertex order.
        if self.x1 < self.x0:
            low, high = self.x1, self.x0
            object.__setattr__(self, "x0", low)
            object.__setattr__(self, "x1", high)
        if self.y1 < self.y0:
            low, high = self.y1, self.y0
            object.__setattr__(self, "y0", low)
            object.__setattr__(self, "y1", high)

    # -- geometry ---------------------------------------------------------
    @property
    def width(self) -> float:
        """Width in points (never negative -- see :meth:`__post_init__`)."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Height in points (never negative -- see :meth:`__post_init__`)."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Area in square points."""
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        """``(x, y)`` of the centre, in points."""
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def union(self, other: BBox) -> BBox:
        """Smallest box containing both.  Requires the same page.

        :param other: a box on the same page
        :returns: the smallest box enclosing both
        :raises ValueError: the boxes are on different pages.  A union across
            pages has no geometric meaning, and silently returning one would put
            a reviewer highlight in the wrong place.
        """
        if other.page != self.page:
            raise ValueError("cannot union boxes from pages %d and %d" % (self.page, other.page))
        return BBox(self.page, min(self.x0, other.x0), min(self.y0, other.y0),
                    max(self.x1, other.x1), max(self.y1, other.y1))

    def intersection(self, other: BBox) -> Optional[BBox]:
        """Overlap rectangle, or ``None`` when they do not overlap.

        :param other: any box
        :returns: the overlapping rectangle, or ``None`` when the boxes are on
            different pages or merely touch.  Edge contact is not overlap: two
            adjacent table cells intersect in zero area and return ``None``.
        """
        if other.page != self.page:
            return None
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return BBox(self.page, x0, y0, x1, y1)

    def iou(self, other: BBox) -> float:
        """Intersection over union in ``[0, 1]``.

        :param other: any box
        :returns: overlap area divided by combined area, in ``0.0..1.0``.
            ``0.0`` for boxes on different pages or with no overlap; ``1.0`` for
            identical boxes.
        """
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        denom = self.area + other.area - inter.area
        return _safe_div(inter.area, denom)

    def contains(self, other: BBox, tolerance: float = 0.0) -> bool:
        """True when ``other`` lies inside this box on the same page.

        ``tolerance`` slackens every edge outward, which matters because OCR
        boxes and layout boxes come from different estimators and rarely nest
        exactly.

        :param other: the box that might be inside this one
        :param tolerance: slack in points applied outward on every edge.  ``0.0``
            demands exact containment; ``2.0`` (about a character's width at
            12pt) is realistic when testing OCR word boxes against a detected
            table region.
        :returns: whether ``other`` lies within this box, on the same page
        """
        return (other.page == self.page
                and other.x0 >= self.x0 - tolerance and other.y0 >= self.y0 - tolerance
                and other.x1 <= self.x1 + tolerance and other.y1 <= self.y1 + tolerance)

    def overlaps(self, other: BBox) -> bool:
        """True when the two boxes share any area on the same page.

        :param other: any box
        :returns: whether the two share positive area.  Touching edges do not
            count -- see :meth:`intersection`.
        """
        return self.intersection(other) is not None

    def expand(self, margin: float) -> BBox:
        """This box grown by ``margin`` points on every side (negative shrinks).

        :param margin: points to add to every side; negative shrinks.  Useful for
            padding a highlight so it does not clip the glyphs it marks.
        :returns: a new box.  Not clipped to the page, so follow it up
            with :meth:`clipped` when the result must stay on the sheet.
        """
        return BBox(self.page, self.x0 - margin, self.y0 - margin,
                    self.x1 + margin, self.y1 + margin)

    def scaled(self, factor: float) -> BBox:
        """This box with every coordinate multiplied by ``factor``.

        Scales position as well as size, since it multiplies the coordinates
        rather than resizing about the centre.

        :param factor: multiplier applied to all four coordinates
        :returns: a new box on the same page
        """
        return BBox(self.page, self.x0 * factor, self.y0 * factor,
                    self.x1 * factor, self.y1 * factor)

    def translated(self, dx: float, dy: float) -> BBox:
        """This box shifted by ``(dx, dy)`` points.

        :param dx: horizontal shift in points; positive moves right
        :param dy: vertical shift in points; positive moves down, since the origin
            is the top-left corner
        :returns: a new box on the same page
        """
        return BBox(self.page, self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def clipped(self, width: float, height: float) -> BBox:
        """This box confined to a ``width`` x ``height`` page.

        :param width: page width in points (595 for A4, 612 for US Letter)
        :param height: page height in points (842 for A4, 792 for US Letter)
        :returns: a new box with every coordinate clamped into the page.  A box
            entirely outside collapses to zero area rather than raising.
        """
        return BBox(self.page,
                    clamp(self.x0, 0.0, width), clamp(self.y0, 0.0, height),
                    clamp(self.x1, 0.0, width), clamp(self.y1, 0.0, height))

    # -- unit conversion --------------------------------------------------
    def to_pixels(self, dpi: float) -> Tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` in integer pixels at ``dpi``.

        :param dpi: the resolution to express the box at.  Use ``page.raster_dpi``
            to index into that page's current raster -- passing a different DPI
            gives coordinates for an image that does not exist.
        :returns: ``(x0, y0, x1, y1)`` as rounded integer pixels
        """
        s = dpi / 72.0
        return (int(round(self.x0 * s)), int(round(self.y0 * s)),
                int(round(self.x1 * s)), int(round(self.y1 * s)))

    @classmethod
    def from_pixels(cls, page: int, x0: float, y0: float, x1: float, y1: float,
                    dpi: float) -> BBox:
        """Build a point-space BBox from pixel coordinates measured at ``dpi``.

        The conversion every backend needs: engines report pixels, the IR stores
        points, and storing pixels would silently invalidate every box the moment
        a resolution-changing op ran.

        :param page: zero-based page index the box belongs to
        :param x0: left edge in pixels
        :param y0: top edge in pixels
        :param x1: right edge in pixels
        :param y1: bottom edge in pixels
        :param dpi: resolution the pixel coordinates were measured at, normally
            ``page.raster_dpi``
        :returns: the box in points (1/72 inch), which stays valid across
            rescaling
        """
        s = 72.0 / float(dpi)
        return cls(page, x0 * s, y0 * s, x1 * s, y1 * s)

    @classmethod
    def from_xywh(cls, page: int, x: float, y: float, w: float, h: float) -> BBox:
        """Build from a top-left corner plus a width and height.

        For engines reporting ``(x, y, w, h)`` rather than two corners --
        Tesseract's TSV output among them.

        :param page: zero-based page index
        :param x: left edge, in points
        :param y: top edge, in points
        :param w: width in points
        :param h: height in points
        :returns: the equivalent corner-based box
        """
        return cls(page, x, y, x + w, y + h)

    @classmethod
    def from_quad(cls, page: int, points: Sequence[Sequence[float]]) -> BBox:
        """Axis-aligned hull of a polygon -- what most detectors actually emit.

        PaddleOCR, RapidOCR and EasyOCR all detect quadrilaterals so that rotated
        text is found correctly.  The IR stores axis-aligned boxes, so the hull is
        taken here; backends that need the original keep it in ``span.meta``.

        :param page: zero-based page index
        :param points: the polygon's vertices as ``[(x, y), ...]`` in points --
            any number of them, though detectors emit four
        :returns: the smallest axis-aligned box containing every vertex
        """
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return cls(page, min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def whole_page(cls, page: int, width_pt: float, height_pt: float) -> BBox:
        """A box covering the entire page -- the default evidence region.

        What a vision backend attaches to its spans, since it reports no
        coordinates.  Such spans are marked ``approximate_bbox`` so nothing
        mistakes a page-sized box for real geometry.

        :param page: zero-based page index
        :param width_pt: page width in points
        :param height_pt: page height in points
        :returns: a box spanning the whole page
        """
        return cls(page, 0.0, 0.0, width_pt, height_pt)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, float]:
        """JSON-ready dict; coordinates are rounded to 3 decimals (~0.001 pt)."""
        return {"page": self.page, "x0": round(self.x0, 3), "y0": round(self.y0, 3),
                "x1": round(self.x1, 3), "y1": round(self.y1, 3)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BBox:
        """Inverse of :meth:`to_dict`.

        :param d: a mapping with ``page``, ``x0``, ``y0``, ``x1``, ``y1``
        :returns: the reconstructed box
        :raises KeyError: a required key is missing
        """
        return cls(int(d["page"]), float(d["x0"]), float(d["y0"]),
                   float(d["x1"]), float(d["y1"]))

    def __repr__(self) -> str:
        """``BBox(page=3, 72.0, 90.0, 300.0, 110.0)``."""
        return "BBox(page=%d, %.1f, %.1f, %.1f, %.1f)" % (
            self.page, self.x0, self.y0, self.x1, self.y1)


def merge_bboxes(boxes: Sequence[BBox]) -> List[BBox]:
    """Union boxes per page.  Evidence spanning pages stays as one box per page.

    Per page rather than one box overall, because a value found on pages 2 and 7
    is two pieces of evidence -- collapsing them would produce a box spanning
    both and pointing at neither.

    :param boxes: any boxes, in any order, from any pages
    :returns: one merged box per page that appeared, ordered by page index.
        Empty input gives an empty list.
    """
    by_page: Dict[int, BBox] = {}
    for b in boxes:
        cur = by_page.get(b.page)
        by_page[b.page] = b if cur is None else cur.union(b)
    return [by_page[p] for p in sorted(by_page)]


@dataclass
class TextSpan(object):
    """A run of text with a location, a producer and (maybe) a confidence.

    ``confidence`` is deliberately ``Optional``: several backends report
    nothing, and inventing 1.0 for them would poison confidence fusion
    downstream.  ``None`` means "no evidence", not "certain".
    """

    text: str
    bbox: BBox
    source: str = "unknown"
    confidence: Optional[float] = None
    script: Optional[str] = None
    line: Optional[int] = None
    block: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def page(self) -> int:
        """Index of the page this span sits on."""
        return self.bbox.page

    @property
    def is_empty(self) -> bool:
        """True when the span carries no non-whitespace text."""
        return not self.text.strip()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict.

        Optional fields are omitted when unset rather than serialised as null:
        a document's span list is the bulk of its JSON, and absent-means-unknown
        round-trips through :meth:`from_dict` unchanged.
        """
        d = {"text": self.text, "bbox": self.bbox.to_dict(), "source": self.source}
        if self.confidence is not None:
            d["confidence"] = round(float(self.confidence), 4)
        if self.script:
            d["script"] = self.script
        if self.line is not None:
            d["line"] = self.line
        if self.block is not None:
            d["block"] = self.block
        if self.meta:
            d["meta"] = _as_jsonable(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TextSpan:
        """Inverse of :meth:`to_dict`.

        :param d: a mapping as produced by :meth:`to_dict`
        :returns: the reconstructed span
        :raises KeyError: ``text`` or ``bbox`` is missing
        """
        return cls(text=d["text"], bbox=BBox.from_dict(d["bbox"]),
                   source=d.get("source", "unknown"), confidence=d.get("confidence"),
                   script=d.get("script"), line=d.get("line"), block=d.get("block"),
                   meta=dict(d.get("meta") or {}))


@dataclass
class PageQuality(object):
    """Measured channel degradation for one page.

    Every number here is a *measurement*, not a guess, and the preprocessing
    policy and confidence prior are both functions of these values.  See
    :func:`Quality.measure_quality` for how each is computed.
    """

    blur: float = 1.0                  #: 0 = smeared, 1 = crisp (normalised VoL)
    skew_deg: float = 0.0              #: positive = content rotated clockwise
    contrast: float = 1.0              #: 0 = flat grey, 1 = full range
    ink_coverage: float = 0.0          #: fraction of dark pixels
    effective_dpi: int = 0             #: estimated *content* resolution
    raster_dpi: int = 0                #: dpi the raster was rendered at
    noise: float = 0.0                 #: 0 = clean, 1 = heavy speckle
    illumination: float = 1.0          #: 1 = even lighting, 0 = severe gradient
    verdict: Verdict = Verdict.CLEAN
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def is_readable(self) -> bool:
        """True unless the page was judged :attr:`Verdict.UNREADABLE`."""
        return self.verdict is not Verdict.UNREADABLE

    @property
    def score(self) -> float:
        """Single scalar in ``[0, 1]`` summarising page health.

        Used as the prior term in confidence fusion.  Weights are deliberately
        blunt -- they exist to be recalibrated against a labelled set (Layer 5),
        not to be believed as-is.
        """
        dpi_term = clamp(self.effective_dpi / 300.0, 0.0, 1.0) if self.effective_dpi else 0.5
        terms = [
            (0.30, clamp(self.blur, 0.0, 1.0)),
            (0.25, dpi_term),
            (0.20, clamp(self.contrast, 0.0, 1.0)),
            (0.15, 1.0 - clamp(self.noise, 0.0, 1.0)),
            (0.10, clamp(self.illumination, 0.0, 1.0)),
        ]
        return round(sum(w * v for w, v in terms), 4)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict, including the derived :attr:`score`."""
        return {
            "blur": round(self.blur, 4), "skew_deg": round(self.skew_deg, 3),
            "contrast": round(self.contrast, 4), "ink_coverage": round(self.ink_coverage, 4),
            "effective_dpi": self.effective_dpi, "raster_dpi": self.raster_dpi,
            "noise": round(self.noise, 4), "illumination": round(self.illumination, 4),
            "verdict": str(self.verdict), "score": self.score,
            "extra": dict((k, round(v, 4)) for k, v in self.extra.items()),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PageQuality:
        """Inverse of :meth:`to_dict` (the derived ``score`` key is ignored).

        :param d: a mapping as produced by :meth:`to_dict`
        :returns: the reconstructed quality.  Every key has a default that means
            "undegraded", so a partial dict loads as a clean page rather than a
            spuriously bad one.  ``score`` is recomputed from the components.
        """
        return cls(blur=float(d.get("blur", 1.0)), skew_deg=float(d.get("skew_deg", 0.0)),
                   contrast=float(d.get("contrast", 1.0)),
                   ink_coverage=float(d.get("ink_coverage", 0.0)),
                   effective_dpi=int(d.get("effective_dpi", 0)),
                   raster_dpi=int(d.get("raster_dpi", 0)),
                   noise=float(d.get("noise", 0.0)),
                   illumination=float(d.get("illumination", 1.0)),
                   verdict=Verdict(d.get("verdict", "clean")),
                   extra=dict(d.get("extra") or {}))

    def __repr__(self) -> str:
        """The handful of numbers worth seeing in a log line or a REPL."""
        return ("PageQuality(verdict=%s, blur=%.2f, skew=%.2f deg, contrast=%.2f, "
                "dpi~%d, score=%.2f)" % (self.verdict, self.blur, self.skew_deg,
                                         self.contrast, self.effective_dpi, self.score))


@dataclass
class LayoutRegion(object):
    """A detected region of a page: table, stamp, signature, handwriting..."""

    kind: RegionKind
    bbox: BBox
    confidence: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return {"kind": str(self.kind), "bbox": self.bbox.to_dict(),
                "confidence": round(self.confidence, 4), "meta": _as_jsonable(self.meta)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LayoutRegion:
        """Inverse of :meth:`to_dict`.

        :param d: a mapping as produced by :meth:`to_dict`
        :returns: the reconstructed region
        :raises KeyError: ``bbox`` is missing
        :raises ValueError: ``kind`` is not a :class:`RegionKind` value
        """
        return cls(RegionKind(d.get("kind", "unknown")), BBox.from_dict(d["bbox"]),
                   float(d.get("confidence", 1.0)), dict(d.get("meta") or {}))


@dataclass
class Layout(object):
    """Region inventory for a page, with the predicates routing cares about."""

    regions: List[LayoutRegion] = field(default_factory=list)

    def of_kind(self, kind: RegionKind) -> List[LayoutRegion]:
        """Every region of exactly ``kind``, in detection order.

        :param kind: the :class:`RegionKind` to filter on, e.g.
            ``RegionKind.TABLE`` or ``RegionKind.SIGNATURE``.  Exact, not
            hierarchical -- asking for ``TABLE`` never returns table cells.
        :returns: the matching regions, in the order the detector found them
        """
        return [r for r in self.regions if r.kind is kind]

    @property
    def is_tabular(self) -> bool:
        """True when at least one table region was detected."""
        return bool(self.of_kind(RegionKind.TABLE))

    @property
    def has_handwriting(self) -> bool:
        """True when at least one handwriting region was detected."""
        return bool(self.of_kind(RegionKind.HANDWRITING))

    @property
    def has_stamps(self) -> bool:
        """True when at least one stamp region was detected."""
        return bool(self.of_kind(RegionKind.STAMP))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return {"regions": [r.to_dict() for r in self.regions]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Layout:
        """Inverse of :meth:`to_dict`.

        :param d: a mapping as produced by :meth:`to_dict`
        :returns: the reconstructed layout; a missing ``regions`` key yields an
            empty layout rather than an error
        """
        return cls([LayoutRegion.from_dict(r) for r in (d.get("regions") or [])])


@dataclass
class OpRecord(object):
    """One entry in a page's processing history.

    This is what makes an experiment reproducible and lets the eval harness
    attribute an accuracy delta to a specific op rather than to "the new build".
    """

    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict, omitting empty params and notes."""
        d = {"op": self.op, "ms": round(self.ms, 2)}
        if self.params:
            d["params"] = _as_jsonable(self.params)
        if self.note:
            d["note"] = self.note
        return d

    def __str__(self) -> str:
        """``op(param=value, ...)`` -- how history reads in a log line."""
        if not self.params:
            return self.op
        args = ", ".join("%s=%r" % (k, v) for k, v in sorted(self.params.items()))
        return "%s(%s)" % (self.op, args)


#: Default rendering resolution.  300 DPI is the point where Tesseract's
#: character models stop degrading on 8-10pt body text; below ~200 accuracy
#: falls off a cliff, above ~400 you pay for pixels that carry no extra signal.
DEFAULT_DPI = 300

#: Assumed resolution for a bare image file with no DPI metadata.  A phone
#: photo of an A4 page is typically 150-250 effective DPI.
DEFAULT_IMAGE_DPI = 200

RasterProvider = Callable[[int], Any]  # (dpi) -> np.ndarray


@dataclass
class Page(object):
    """One page: its geometry, its measured quality, its text, its history.

    The raster is *lazy by contract*, not as an optimisation.  A 300-page legal
    bundle rendered eagerly at 400 DPI is tens of gigabytes of RAM, so laziness
    has to live in the type -- it cannot be retrofitted by a caller.
    """

    index: int
    width_pt: float = 612.0
    height_pt: float = 792.0
    kind: PageKind = PageKind.SCANNED
    rotation: int = 0
    quality: PageQuality = field(default_factory=PageQuality)
    spans: List[TextSpan] = field(default_factory=list)
    layout: Layout = field(default_factory=Layout)
    history: List[OpRecord] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Raster state.  Private by convention; use raster()/set_raster().
    _raster: Optional[Any] = field(default=None, repr=False, compare=False)
    _raster_dpi: int = field(default=0, repr=False, compare=False)
    _raster_provider: Optional[RasterProvider] = field(default=None, repr=False, compare=False)

    # -- raster -----------------------------------------------------------
    def raster(self, dpi: Optional[int] = None) -> ImageArray:
        """Return the page image as a uint8 NumPy array, rendering on demand.

        Semantics that matter:

        * With a provider still attached (nothing has been preprocessed yet), a
          request at a different DPI re-renders from source -- always better
          than upsampling.
        * Once an op has run, the provider is dropped and the processed raster
          is authoritative; a differing DPI request rescales it instead, so an
          op chain is never silently discarded.

        :param dpi: the resolution wanted.  ``None`` keeps the page's current
            DPI, falling back to :data:`DEFAULT_DPI`.  Ask for what you need
            rather than rescaling afterwards -- a re-render from the PDF beats
            an upsample every time, and this method knows which is possible.
        :returns: the page image as a ``uint8`` NumPy array, greyscale or RGB
        :raises DocpipeError: the page has neither a raster nor a provider,
            e.g. an email-body page, which carries text but no pixels.  Guard
            with :meth:`has_raster`.
        """
        want = int(dpi or self._raster_dpi or DEFAULT_DPI)
        if self._raster is not None:
            if self._raster_dpi == want or self._raster_provider is None:
                if self._raster_dpi and self._raster_dpi != want and self._raster_provider is None:
                    scale = float(want) / float(self._raster_dpi)
                    if abs(scale - 1.0) > 1e-6:
                        img = resize_image(self._raster, scale=scale)
                        self._raster = img
                        self._raster_dpi = want
                return self._raster
        if self._raster_provider is not None:
            img = self._raster_provider(want)
            self._raster = img
            self._raster_dpi = want
            if not self.quality.raster_dpi:
                self.quality.raster_dpi = want
            return img
        raise DocpipeError(
            "page %d has no raster and no raster provider; ingest it from a "
            "PDF/image or call set_raster()" % self.index)

    def has_raster(self) -> bool:
        """True when a raster is materialised or can be produced without I/O errors."""
        return self._raster is not None or self._raster_provider is not None

    def set_raster(self, img: ImageArray, dpi: Optional[int] = None,
                   keep_provider: bool = False) -> Page:
        """Install ``img`` as this page's raster.  Ops use this.

        Dropping the provider is the default and is deliberate: after a deskew,
        re-rendering from the PDF would silently undo the op.

        :param img: the new raster, a ``uint8`` NumPy array
        :param dpi: the resolution ``img`` represents.  ``None`` keeps the page's
            current DPI -- correct for an op that preserved size, wrong for one
            that rescaled, which must pass the new value.
        :param keep_provider: retain the lazy source.  Leave this ``False``
            except when installing a raster equivalent to what the provider
            would return; keeping it lets a later DPI change discard your image.
        :returns: ``self``, so calls chain
        """
        self._raster = img
        self._raster_dpi = int(dpi or self._raster_dpi or DEFAULT_DPI)
        if not keep_provider:
            self._raster_provider = None
        return self

    def set_raster_provider(self, provider: RasterProvider,
                            dpi: Optional[int] = None) -> Page:
        """Attach a lazy raster source, e.g. a PDF page renderer.

        :param provider: called as ``provider(dpi)`` and must return an ImageArray
        :param dpi: the resolution to assume until one is requested explicitly
        """
        self._raster_provider = provider
        if dpi:
            self._raster_dpi = int(dpi)
        return self

    def release_raster(self) -> Page:
        """Drop the materialised raster.  Only safe while a provider remains."""
        if self._raster_provider is not None:
            self._raster = None
        return self

    @property
    def raster_dpi(self) -> int:
        """Resolution of the current raster, falling back to :data:`DEFAULT_DPI`."""
        return self._raster_dpi or DEFAULT_DPI

    @property
    def pixel_size(self) -> Tuple[int, int]:
        """``(width, height)`` of the current raster in pixels, ``(0, 0)`` if none."""
        if self._raster is None:
            return (0, 0)
        shape = self._raster.shape
        return (int(shape[1]), int(shape[0]))

    # -- text -------------------------------------------------------------
    def text(self, separator: str = "\n") -> str:
        """Reading-order text.  Spans are sorted by line, then by x.

        :param separator: placed between lines; the default single newline
            preserves the page's line structure, which is what amount-and-label
            parsing depends on
        :returns: the page's text.  Empty for a page that has not been read.
        """
        return separator.join(l.strip() for l in self.lines() if l.strip())

    def lines(self, y_tolerance: Optional[float] = None) -> List[str]:
        """Group spans into visual lines and return them top-to-bottom.

        ``y_tolerance`` defaults to 60% of the median span height, which tracks
        font size instead of assuming one.

        :param y_tolerance: vertical distance in points within which two spans
            count as the same line.  ``None`` derives it from the page's own
            median span height, which is right across font sizes.  Pass a value
            only for pages that defeat that -- a table whose rows are closer
            than its text is tall.
        :returns: one string per visual line, top to bottom, words joined by
            single spaces.  Empty lines are dropped.
        """
        return [" ".join(s.text.strip() for s in group if s.text.strip())
                for group in _group_spans_into_lines(self, y_tolerance)]

    def spans_in(self, bbox: BBox, min_overlap: float = 0.5) -> List[TextSpan]:
        """Spans whose area overlaps ``bbox`` by at least ``min_overlap``.

        The fraction is of the *span's* area, not the box's, so a small word
        inside a large region counts as fully contained.

        :param bbox: the region to search, in points
        :param min_overlap: least fraction of each span's area that must fall
            inside, in ``0.0..1.0``.  ``0.5`` requires the majority of the word;
            ``0.3`` is more forgiving and is what evidence matching uses, since
            OCR boxes and layout boxes come from different estimators.
        :returns: the qualifying spans, in page order
        """
        out = []
        for s in self.spans:
            inter = s.bbox.intersection(bbox)
            if inter is not None and _safe_div(inter.area, s.bbox.area, 1.0) >= min_overlap:
                out.append(s)
        return out

    @property
    def char_count(self) -> int:
        """Total non-whitespace characters across every span on the page."""
        return sum(len(s.text.strip()) for s in self.spans)

    @property
    def is_blank(self) -> bool:
        """True when the page is worth skipping entirely.

        Either it was classified blank at ingest, or it has no text *and*
        essentially no ink -- both checks are needed, because an unread scan of a
        full page also has a character count of zero.
        """
        return self.kind is PageKind.BLANK or (self.char_count == 0
                                               and self.quality.ink_coverage < 0.002)

    # -- bookkeeping ------------------------------------------------------
    def record(self, op: str, params: Optional[Dict[str, Any]] = None, ms: float = 0.0,
               note: str = "") -> Page:
        """Append an entry to this page's processing history.  Returns ``self``.

        :param op: the op's registered name
        :param params: the arguments it ran with, for reproducibility
        :param ms: wall-clock duration
        :param note: anything worth knowing, e.g. why an op declined to run
        """
        self.history.append(OpRecord(op=op, params=dict(params or {}), ms=ms, note=note))
        return self

    def history_summary(self) -> str:
        """The op history as one ``a -> b -> c`` line."""
        return " -> ".join(str(h) for h in self.history)

    def bbox(self) -> BBox:
        """A box covering the whole page, in points."""
        return BBox.whole_page(self.index, self.width_pt, self.height_pt)

    def copy(self, deep_spans: bool = True) -> Page:
        """Shallow-copy the page, sharing the raster but not the span list.

        Ops use this so that a pipeline never mutates the caller's document --
        A/B'ing two pipelines against one ingested document has to be safe.

        The raster is shared, not copied: it is large, and ops replace it rather
        than writing into it.  Everything else -- spans, layout, history, meta --
        is copied, so the two pages diverge cleanly.

        :param deep_spans: copy each :class:`TextSpan` rather than sharing the
            objects.  ``True`` is right whenever the copy will be read or
            normalised, since span mutation would otherwise reach back into the
            original.
        :returns: the new page
        """
        new = Page(
            index=self.index, width_pt=self.width_pt, height_pt=self.height_pt,
            kind=self.kind, rotation=self.rotation,
            quality=dataclasses.replace(self.quality, extra=dict(self.quality.extra)),
            spans=[dataclasses.replace(s) for s in self.spans] if deep_spans else list(self.spans),
            layout=Layout(list(self.layout.regions)),
            history=list(self.history), meta=dict(self.meta),
        )
        new._raster = self._raster
        new._raster_dpi = self._raster_dpi
        new._raster_provider = self._raster_provider
        return new

    def to_dict(self, include_spans: bool = True) -> Dict[str, Any]:
        """JSON-ready dict.

        ``include_spans=False`` keeps the structure and the measurements but drops
        the text, which is what the eval harness stores for a fixture.

        :param include_spans: keep the per-span text and geometry
        :returns: a JSON-serialisable dict.  The raster is never included --
            see :meth:`Page.from_dict`.
        """
        d = {
            "index": self.index, "kind": str(self.kind),
            "width_pt": round(self.width_pt, 2), "height_pt": round(self.height_pt, 2),
            "rotation": self.rotation, "quality": self.quality.to_dict(),
            "layout": self.layout.to_dict(),
            "history": [h.to_dict() for h in self.history],
            "meta": _as_jsonable(self.meta),
        }
        if include_spans:
            d["spans"] = [s.to_dict() for s in self.spans]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Page:
        """Inverse of :meth:`to_dict`.  The raster is *not* restored.

        Pixels are deliberately not serialised -- a JSON document would be
        enormous -- so a restored page has spans, geometry and history but
        :meth:`has_raster` is ``False``.  Re-ingest the source to read it again.

        :param d: a mapping as produced by :meth:`to_dict`
        :returns: the reconstructed page
        :raises KeyError: ``index`` is missing; every other key has a default
        """
        page = cls(
            index=int(d["index"]),
            width_pt=float(d.get("width_pt", 612.0)),
            height_pt=float(d.get("height_pt", 792.0)),
            kind=PageKind(d.get("kind", "scanned")),
            rotation=int(d.get("rotation", 0)),
            quality=PageQuality.from_dict(d.get("quality") or {}),
            spans=[TextSpan.from_dict(s) for s in (d.get("spans") or [])],
            layout=Layout.from_dict(d.get("layout") or {}),
            meta=dict(d.get("meta") or {}),
        )
        for h in (d.get("history") or []):
            page.record(h.get("op", "?"), h.get("params"), float(h.get("ms", 0.0)),
                        h.get("note", ""))
        return page

    def __repr__(self) -> str:
        """Index, kind, size, span count and quality verdict."""
        return "Page(index=%d, kind=%s, %dx%d pt, spans=%d, %s)" % (
            self.index, self.kind, self.width_pt, self.height_pt,
            len(self.spans), self.quality.verdict)


@dataclass
class Document(object):
    """An ordered collection of pages plus provenance about where it came from."""

    source_uri: str = ""
    pages: List[Page] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Number of pages."""
        return len(self.pages)

    def __iter__(self) -> Iterator[Page]:
        """Iterate over pages in order."""
        return iter(self.pages)

    def __getitem__(self, i: int) -> Page:
        """Page by list position (see :meth:`page` for lookup by page index).

        Position and page index diverge the moment pages are filtered or split,
        which is why both accessors exist.

        :param i: list position, negative indexing allowed
        :returns: the page at that position
        :raises IndexError: out of range
        """
        return self.pages[i]

    def page(self, index: int) -> Page:
        """Page by its ``index`` attribute (not its list position).

        The accessor to use when resolving provenance: every ``bbox.page`` is a
        page index, not a position, and the two differ after :meth:`select`.

        :param index: the page's own ``index`` attribute
        :returns: the matching page
        :raises KeyError: no page carries that index
        """
        for p in self.pages:
            if p.index == index:
                return p
        raise KeyError("no page with index %d" % index)

    def text(self, separator: str = "\n\n") -> str:
        """Reading-order text of every non-empty page, joined by ``separator``.

        :param separator: placed between pages.  The default blank line reads
            naturally; use :func:`format_document_text` instead when the text is
            headed for a prompt, since that one adds page markers.
        :returns: the document's text.  Empty for a document that has been
            ingested but not read.
        """
        return separator.join(p.text() for p in self.pages if p.text().strip())

    def spans(self) -> List[TextSpan]:
        """Every span in the document, page by page, in order."""
        out: List[TextSpan] = []
        for p in self.pages:
            out.extend(p.spans)
        return out

    @property
    def char_count(self) -> int:
        """Total non-whitespace characters in the document."""
        return sum(p.char_count for p in self.pages)

    def kinds(self) -> Dict[str, int]:
        """Count of pages per :class:`PageKind`, e.g. ``{"scanned": 12}``."""
        counts: Dict[str, int] = {}
        for p in self.pages:
            key = str(p.kind)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def warn(self, message: str) -> Document:
        """Record a de-duplicated warning and log it.  Returns ``self``.

        Warnings survive on the Document rather than being raised because a bad
        page is not a bad document: the caller usually wants the other 40 pages.

        :param message: what went wrong.  Duplicates are dropped, so a warning
            raised once per page on a 300-page bundle appears once.
        :returns: ``self``, so calls chain
        """
        if message not in self.warnings:
            self.warnings.append(message)
        logger.warning("%s: %s", self.source_uri or "<doc>", message)
        return self

    def copy(self) -> Document:
        """Deep-enough copy: pages are copied, rasters are shared."""
        return Document(source_uri=self.source_uri, pages=[p.copy() for p in self.pages],
                        meta=dict(self.meta), warnings=list(self.warnings))

    def release_rasters(self) -> Document:
        """Free every materialised raster that can be re-rendered on demand."""
        for p in self.pages:
            p.release_raster()
        return self

    def select(self, predicate: Callable[[Page], bool]) -> Document:
        """A new Document containing only pages satisfying ``predicate``.

        Page ``index`` attributes are preserved, not renumbered, so provenance
        still resolves through :meth:`page`.  Use :func:`Ingest.merge_documents`
        when you do want contiguous renumbering.

        :param predicate: ``Callable[[Page], bool]``, e.g.
            ``lambda p: not p.is_blank`` or ``lambda p: p.quality.score > 0.5``
        :returns: a new document sharing the surviving page objects
        """
        return Document(source_uri=self.source_uri,
                        pages=[p for p in self.pages if predicate(p)],
                        meta=dict(self.meta), warnings=list(self.warnings))

    def to_dict(self, include_spans: bool = True) -> Dict[str, Any]:
        """JSON-ready dict; ``include_spans=False`` drops the text.

        :param include_spans: keep per-page spans.  ``False`` yields a compact
            structural summary -- geometry, quality, history -- which is what you
            want for a manifest or a diff between two runs.
        :returns: a JSON-serialisable dict.  Rasters are never included.
        """
        return {
            "source_uri": self.source_uri,
            "meta": _as_jsonable(self.meta),
            "warnings": list(self.warnings),
            "pages": [p.to_dict(include_spans=include_spans) for p in self.pages],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Document:
        """Inverse of :meth:`to_dict`.  Rasters are *not* restored.

        :param d: a mapping as produced by :meth:`to_dict`
        :returns: the reconstructed document, on which every page reports
            a :meth:`Page.has_raster` of ``False``.
        """
        return cls(source_uri=d.get("source_uri", ""),
                   pages=[Page.from_dict(p) for p in (d.get("pages") or [])],
                   meta=dict(d.get("meta") or {}),
                   warnings=list(d.get("warnings") or []))

    def save_json(self, path: str, include_spans: bool = True) -> str:
        """Write :meth:`to_dict` to ``path`` as UTF-8 JSON.  Returns ``path``.

        :param path: destination file; overwritten if it exists
        :param include_spans: keep per-page spans, as :meth:`to_dict`
        :returns: ``path``, so it can be used inline
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(to_json(self.to_dict(include_spans=include_spans)))
        return path

    @classmethod
    def load_json(cls, path: str) -> Document:
        """Read a document back from a file written by :meth:`save_json`.

        :param path: a file written by :meth:`save_json`
        :returns: the reconstructed document, without rasters
        :raises OSError: the file could not be read
        :raises ValueError: the file is not valid JSON
        """
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self) -> str:
        """Source, page count and character count."""
        return "Document(%r, pages=%d, chars=%d)" % (
            self.source_uri, len(self.pages), self.char_count)


@dataclass
class Cost(object):
    """Money and tokens spent.  Adds like a number so it can be accumulated."""

    currency: str = "USD"
    amount: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    wasted_calls: int = 0  #: attempts that failed *after* the provider billed us

    def __add__(self, other: Cost) -> Cost:
        """Sum two costs.

        Mixing currencies raises :class:`ConfigError` -- unless one side is zero,
        which is what makes ``sum(costs)`` (starting from ``0``) work.

        :param other: the cost to add
        :returns: a new :class:`Cost` with amounts, tokens, calls and wasted
            calls summed.  ``NotImplemented`` for a non-``Cost``, so Python falls
            back to the reflected operand.
        :raises ConfigError: both sides are non-zero and in different currencies.
            No conversion is attempted -- a guessed exchange rate is exactly the
            kind of confidently wrong number this library refuses to produce.
        """
        if not isinstance(other, Cost):
            return NotImplemented
        if self.amount and other.amount and self.currency != other.currency:
            raise ConfigError("cannot add costs in %s and %s" % (self.currency, other.currency))
        return Cost(currency=self.currency if self.amount or not other.amount else other.currency,
                    amount=self.amount + other.amount,
                    input_tokens=self.input_tokens + other.input_tokens,
                    output_tokens=self.output_tokens + other.output_tokens,
                    calls=self.calls + other.calls,
                    wasted_calls=self.wasted_calls + other.wasted_calls)

    __radd__ = __add__

    @classmethod
    def zero(cls, currency: str = "USD") -> Cost:
        """An empty cost in ``currency`` -- the identity for addition.

        :param currency: the currency label; no conversion is ever performed, so
            it must match the units of your registered prices
        :returns: a zero cost, safe to use as an accumulator seed
        """
        return cls(currency=currency)

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict; the amount keeps 6 decimals (sub-cent per page)."""
        return {"currency": self.currency, "amount": round(self.amount, 6),
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "calls": self.calls, "wasted_calls": self.wasted_calls}

    def __repr__(self) -> str:
        """Amount, currency and token counts."""
        return "Cost(%.6f %s, in=%d, out=%d, calls=%d)" % (
            self.amount, self.currency, self.input_tokens, self.output_tokens, self.calls)


# =============================================================================
# 4. Image primitives
# =============================================================================
#
# Every raster helper here has an OpenCV path and, where it is honestly
# feasible, a pure-NumPy fallback.  That is not gold-plating: OpenCV wheels are
# awkward in some locked-down deployment targets, and a library that hard-fails
# without it cannot be used at all there.  The fallbacks are slower and
# occasionally slightly different numerically; they are not toys, and the test
# suite runs the ops both ways.
#
# Array convention: uint8, (H, W) grayscale or (H, W, 3) RGB.  RGB, not BGR --
# the OpenCV boundary is crossed inside these helpers, never outside them.


def is_gray(img: ImageArray) -> bool:
    """True for a single-channel ``(H, W)`` array.

    :param img: an array, or anything without ``ndim`` (which reports ``False``)
    :returns: whether the array is two-dimensional
    """
    return getattr(img, "ndim", 0) == 2


def is_color(img: ImageArray) -> bool:
    """True for a multi-channel ``(H, W, C>=3)`` array.

    :param img: an array, or anything without ``ndim`` (which reports ``False``)
    :returns: whether the array is three-dimensional with at least 3 channels.
        A single-channel ``(H, W, 1)`` array is *not* colour by this test.
    """
    return getattr(img, "ndim", 0) == 3 and img.shape[2] >= 3


def image_shape(img: ImageArray) -> Tuple[int, int]:
    """``(height, width)`` in pixels.

    Note the order: NumPy's, not the ``(width, height)`` that
    :func:`Image.resize_image` takes.  Mixing the two silently transposes pages.

    :param img: a 2D or 3D array
    :returns: ``(height, width)``, ignoring any channel dimension
    """
    return (int(img.shape[0]), int(img.shape[1]))


def _integral(arr: ImageArray) -> FloatArray:
    """Summed-area table with a zero row/column, shape ``(H+1, W+1)``."""
    np = _np()
    cs = np.cumsum(np.cumsum(np.asarray(arr, dtype=np.float64), axis=0), axis=1)
    return np.pad(cs, ((1, 0), (1, 0)), mode="constant")


def _rect_box_sum(arr: ImageArray, kh: int, kw: int) -> Tuple[FloatArray, FloatArray]:
    """``(window_sums, window_counts)`` over a ``kh x kw`` rectangle, clamped."""
    np = _np()
    a = np.asarray(arr, dtype=np.float64)
    h, w = a.shape[:2]
    ry, rx = max(0, kh // 2), max(0, kw // 2)
    integral = _integral(a)
    ys = np.arange(h); xs = np.arange(w)
    y0 = np.clip(ys - ry, 0, h); y1 = np.clip(ys + ry + 1, 0, h)
    x0 = np.clip(xs - rx, 0, w); x1 = np.clip(xs + rx + 1, 0, w)
    sums = (integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)]
            - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)])
    counts = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float64)
    return sums, counts


class Image(object):
    """Raster primitives, each with an OpenCV path and a NumPy fallback.

    Array convention: uint8, ``(H, W)`` greyscale or ``(H, W, 3)`` RGB --
    RGB, not BGR.  The OpenCV boundary is crossed inside these helpers and
    never outside them, which is what makes
    :func:`Caps.set_opencv_enabled` a single switch rather than a hunt.

    These take arrays and return arrays; they know nothing about a
    :class:`Page`.  Pixel work that corrects a *measured* degradation is an
    op instead -- see :class:`Ops`.
    """

    @staticmethod
    def ensure_uint8(img: ImageArray) -> ImageArray:
        """Coerce any numeric array to uint8, scaling float [0,1] images by 255.

        The guard at the top of every image primitive, so ops need not care
        whether an upstream step handed back floats or booleans.

        :param img: any numeric array -- ``uint8`` passes straight through,
            floats peaking at or below ``1.0`` are scaled by 255, booleans map to
            ``0``/``255``, and anything else is clipped into ``0..255``
        :returns: a ``uint8`` array of the same shape
        """
        np = _np()
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            return arr
        if arr.dtype in (np.float32, np.float64, np.float16):
            peak = float(arr.max()) if arr.size else 0.0
            if peak <= 1.0 + 1e-6:
                arr = arr * 255.0
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8) * 255
            return arr
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def to_gray(img: ImageArray) -> GrayImage:
        """Grayscale view of ``img`` using ITU-R 601 luma weights.

        :param img: a greyscale or colour array.  Greyscale input is returned
            unchanged, so this is safe to call unconditionally; an alpha channel
            is dropped.
        :returns: a single-channel ``uint8`` array
        """
        np = _np()
        arr = ensure_uint8(img)
        if arr.ndim == 2:
            return arr
        cv = _cv2()
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if cv is not None:
            return cv.cvtColor(arr, cv.COLOR_RGB2GRAY)
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return np.clip(arr[:, :, :3].astype(np.float32).dot(weights), 0, 255).astype(np.uint8)

    @staticmethod
    def to_rgb(img: ImageArray) -> ImageArray:
        """3-channel RGB view of ``img``.

        :param img: a greyscale or colour array.  Greyscale is replicated across
            three channels; a 4-channel image loses its alpha.
        :returns: an ``(H, W, 3)`` ``uint8`` array
        """
        np = _np()
        arr = ensure_uint8(img)
        if arr.ndim == 3:
            return arr[:, :, :3]
        return np.repeat(arr[:, :, None], 3, axis=2)

    @staticmethod
    def window_stats(gray: GrayImage, window: int) -> Tuple[FloatArray, FloatArray]:
        """Local ``(mean, std)`` over a ``window x window`` neighbourhood.

        Uses summed-area tables, so cost is independent of window size -- which is
        what makes Sauvola-family binarisation practical at 300 DPI.  Windows are
        clipped at the borders and the divisor tracks the clipped area, so edge
        pixels are not biased toward black the way zero-padding would make them.

        :param gray: a single-channel array
        :param window: neighbourhood side in pixels.  Cost does not grow with it,
            so pick it by what the page needs -- a little taller than one text
            line for binarisation.
        :returns: ``(mean, std)``, both float arrays the same shape as ``gray``
        """
        np = _np()
        g = np.asarray(gray, dtype=np.float64)
        h, w = g.shape[:2]
        r = max(1, int(window) // 2)
        i1 = _integral(g)
        i2 = _integral(g * g)

        ys = np.arange(h)
        xs = np.arange(w)
        y0 = np.clip(ys - r, 0, h)
        y1 = np.clip(ys + r + 1, 0, h)
        x0 = np.clip(xs - r, 0, w)
        x1 = np.clip(xs + r + 1, 0, w)

        def block(integral: FloatArray) -> FloatArray:
            """Sum over every clipped window at once, from a summed-area table."""
            return (integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)]
                    - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)])

        counts = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float64)
        counts[counts == 0] = 1.0
        mean = block(i1) / counts
        var = np.maximum(block(i2) / counts - mean * mean, 0.0)
        return mean, np.sqrt(var)

    @staticmethod
    def box_blur(img: ImageArray, radius: int) -> ImageArray:
        """Mean filter of radius ``radius`` (integral-image based in the fallback).

        :param img: greyscale or colour; colour is filtered per channel
        :param radius: half-width in pixels, so the kernel is ``2 * radius + 1``
            square.  Below ``1`` the image is returned unchanged.
        :returns: the blurred ``uint8`` array
        """
        np = _np()
        if radius < 1:
            return ensure_uint8(img)
        cv = _cv2()
        arr = ensure_uint8(img)
        k = 2 * int(radius) + 1
        if cv is not None:
            return cv.blur(arr, (k, k))
        if arr.ndim == 3:
            return np.stack([box_blur(arr[:, :, c], radius) for c in range(arr.shape[2])], axis=2)
        mean, _ = window_stats(arr, k)
        return np.clip(mean, 0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_blur(img: ImageArray, sigma: float) -> ImageArray:
        """Gaussian blur.  The fallback is three variance-matched box blurs.

        The fallback still loses a little tail mass, because each pass rounds back
        to uint8, so its effective sigma runs slightly under the request at large
        radii.  Measured against :func:`Quality.measure_blur` the two paths agree to within
        about 0.02 over sigma 0.8-4.0, which is what the tests assert.

        :param img: greyscale or colour
        :param sigma: standard deviation in pixels.  Zero or negative returns the
            image unchanged.  The kernel radius is derived from it, so cost grows
            with sigma on the OpenCV path and stays flat on the fallback.
        :returns: the blurred ``uint8`` array
        """
        if sigma <= 0:
            return ensure_uint8(img)
        cv = _cv2()
        arr = ensure_uint8(img)
        if cv is not None:
            k = int(2 * round(3.0 * sigma) + 1)
            return cv.GaussianBlur(arr, (k, k), sigma)
        out = arr
        for width in _box_widths_for_gaussian(sigma):
            if width > 1:
                out = box_blur(out, (width - 1) // 2)
        return out

    @staticmethod
    def median_blur(img: ImageArray, ksize: int = 3) -> ImageArray:
        """Median filter -- the right tool for salt-and-pepper scanner speckle.

        Unlike a mean filter it removes isolated specks without softening the
        stroke edges around them, which is why denoising uses it rather than a
        blur.

        :param img: greyscale or colour; colour is filtered per channel
        :param ksize: kernel side in pixels, forced odd.  ``3`` removes single
            speckles; ``5`` handles heavier fax noise and starts to round off the
            corners of glyphs.
        :returns: the filtered ``uint8`` array
        """
        np = _np()
        cv = _cv2()
        arr = ensure_uint8(img)
        if cv is not None:
            k = int(ksize) | 1
            return cv.medianBlur(arr, k)
        if arr.ndim == 3:
            return np.stack([median_blur(arr[:, :, c], ksize) for c in range(arr.shape[2])], axis=2)
        r = max(1, int(ksize) // 2)
        padded = np.pad(arr, r, mode="edge")
        h, w = arr.shape
        stack = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                stack.append(padded[r + dy:r + dy + h, r + dx:r + dx + w])
        return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)

    @staticmethod
    def resize_image(img: ImageArray, size: Optional[Tuple[int, int]] = None,
                     scale: Optional[float] = None, interpolation: str = "auto") -> ImageArray:
        """Resize to ``size=(width, height)`` or by ``scale``.

        ``interpolation="auto"`` picks cubic when upscaling (preserves stroke edges
        that OCR character models rely on) and area when downscaling (avoids the
        aliasing that makes fine print look like noise).

        :param img: greyscale or colour
        :param size: target ``(width, height)`` in pixels -- note this is the
            opposite order to :func:`Image.image_shape`.  Takes precedence over
            ``scale``.
        :param scale: uniform multiplier, used when ``size`` is ``None``.  A
            scale of ``1.0`` returns the image unchanged.
        :param interpolation: ``"auto"`` (cubic up, area down), or one of
            ``"nearest"``, ``"linear"``, ``"cubic"``, ``"area"``.  Use
            ``"nearest"`` on an already-binarised page, where interpolation would
            reintroduce grey.
        :returns: the resized ``uint8`` array
        """
        np = _np()
        arr = ensure_uint8(img)
        h, w = image_shape(arr)
        if size is None:
            if not scale or abs(scale - 1.0) < 1e-9:
                return arr
            size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        tw, th = int(size[0]), int(size[1])
        if tw == w and th == h:
            return arr
        cv = _cv2()
        if cv is not None:
            if interpolation == "auto":
                interp = cv.INTER_CUBIC if (tw * th) > (w * h) else cv.INTER_AREA
            else:
                interp = {"nearest": cv.INTER_NEAREST, "linear": cv.INTER_LINEAR,
                          "cubic": cv.INTER_CUBIC, "area": cv.INTER_AREA,
                          "lanczos": cv.INTER_LANCZOS4}.get(interpolation, cv.INTER_LINEAR)
            return cv.resize(arr, (tw, th), interpolation=interp)
        # Bilinear fallback with half-pixel centre alignment (matches OpenCV).
        ys = (np.arange(th, dtype=np.float64) + 0.5) * (float(h) / th) - 0.5
        xs = (np.arange(tw, dtype=np.float64) + 0.5) * (float(w) / tw) - 0.5
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)
        y0 = np.floor(ys).astype(np.int64)
        x0 = np.floor(xs).astype(np.int64)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]
        src = arr.astype(np.float64)
        if src.ndim == 3:
            wy = wy[:, :, None]
            wx = wx[:, :, None]
        top = src[np.ix_(y0, x0)] * (1 - wx) + src[np.ix_(y0, x1)] * wx
        bot = src[np.ix_(y1, x0)] * (1 - wx) + src[np.ix_(y1, x1)] * wx
        return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)

    @staticmethod
    def rotate_image(img: ImageArray, angle_deg: float, border_value: Optional[int] = None,
                     expand: bool = True) -> ImageArray:
        """Rotate counter-clockwise by ``angle_deg`` about the image centre.

        ``expand=True`` grows the canvas so no content is clipped -- deskewing a
        page and losing the top-right corner of the letterhead is a real failure
        mode of the naive implementation.  ``border_value`` defaults to the
        image's median, which keeps a white page white and a dark scan dark rather
        than stamping black wedges that later confuse ink-coverage measurement.

        :param img: greyscale or colour
        :param angle_deg: counter-clockwise rotation in degrees.  Angles under
            ``1e-4`` return the image unchanged.
        :param border_value: fill level for exposed corners, ``0..255``.
            ``None`` uses the image's median, which is what keeps a white page
            white and a dark scan dark.
        :param expand: grow the canvas so nothing is clipped.  Leave it ``True``
            for deskewing -- ``False`` keeps the original dimensions and cuts the
            corners off, which loses letterheads.
        :returns: the rotated ``uint8`` array, larger than the input when
            ``expand`` is set
        """
        np = _np()
        arr = ensure_uint8(img)
        if abs(angle_deg) < 1e-4:
            return arr
        h, w = image_shape(arr)
        if border_value is None:
            border_value = int(np.median(to_gray(arr)))
        theta = math.radians(angle_deg)
        cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))
        if expand:
            nw = int(math.ceil(w * cos_t + h * sin_t))
            nh = int(math.ceil(w * sin_t + h * cos_t))
        else:
            nw, nh = w, h

        cv = _cv2()
        if cv is not None:
            m = cv.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
            m[0, 2] += (nw - w) / 2.0
            m[1, 2] += (nh - h) / 2.0
            bv = (border_value, border_value, border_value) if arr.ndim == 3 else border_value
            return cv.warpAffine(arr, m, (nw, nh), flags=cv.INTER_LINEAR,
                                 borderMode=cv.BORDER_CONSTANT, borderValue=bv)

        # Inverse mapping with bilinear sampling.
        cy, cx = (nh - 1) / 2.0, (nw - 1) / 2.0
        sy, sx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(nh, dtype=np.float64) - cy,
                             np.arange(nw, dtype=np.float64) - cx, indexing="ij")
        # Inverse of OpenCV's forward matrix [[c, s], [-s, c]], so that a positive
        # angle rotates counter-clockwise in both code paths.
        c, s = math.cos(theta), math.sin(theta)
        src_x = c * xx - s * yy + sx
        src_y = s * xx + c * yy + sy
        valid = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
        xc = np.clip(src_x, 0, w - 1)
        yc = np.clip(src_y, 0, h - 1)
        x0 = np.floor(xc).astype(np.int64); x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(yc).astype(np.int64); y1 = np.minimum(y0 + 1, h - 1)
        fx = (xc - x0); fy = (yc - y0)
        src = arr.astype(np.float64)
        if src.ndim == 3:
            fx = fx[:, :, None]; fy = fy[:, :, None]
            valid3 = valid[:, :, None]
        else:
            valid3 = valid
        top = src[y0, x0] * (1 - fx) + src[y0, x1] * fx
        bot = src[y1, x0] * (1 - fx) + src[y1, x1] * fx
        out = top * (1 - fy) + bot * fy
        out = np.where(valid3, out, float(border_value))
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def row_projection_at_angle(gray: GrayImage, angle_deg: float) -> FloatArray:
        """Horizontal projection profile as if the image were rotated by ``angle_deg``.

        Note carefully which axis is sheared.  Shearing *rows* horizontally --
        the obvious reading of "shear to simulate rotation" -- leaves every row sum
        unchanged, so a skew search built on it measures nothing but border
        artefacts.  The columns must be shifted *vertically* instead, so that
        content from a tilted text line lands in the same output row.

        For angles under ~15 degrees a shear and a rotation give indistinguishable
        profiles, and the shear costs one integer fancy-index instead of a full
        resampling pass.

        :param gray: a single-channel array
        :param angle_deg: the tilt to simulate, in degrees.  Accurate for small
            angles only -- beyond about 15 degrees a shear stops approximating a
            rotation, which is why the skew search is bounded there.
        :returns: the per-row ink sums as a float array, whose variance peaks at
            the angle that makes text lines horizontal
        """
        np = _np()
        g = np.asarray(gray, dtype=np.float64)
        h, w = g.shape[:2]
        if abs(angle_deg) < 1e-6:
            return g.sum(axis=1)
        tan_t = math.tan(math.radians(angle_deg))
        shifts = np.round((np.arange(w) - w / 2.0) * tan_t).astype(np.int64)
        rows = np.clip(np.arange(h)[:, None] + shifts[None, :], 0, h - 1)
        cols = np.arange(w)[None, :]
        return g[rows, cols].sum(axis=1)

    @staticmethod
    def otsu_threshold(gray: GrayImage) -> int:
        """Otsu's global threshold via between-class variance maximisation.

        :param gray: a greyscale or colour array; colour is converted first
        :returns: the threshold as an integer in ``0..255``.  Global, so it is
            exact on an evenly-lit page and badly wrong under a shadow gradient
            -- which is what :func:`Image.ink_mask` detects and works around.
        """
        np = _np()
        g = to_gray(gray)
        hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
        total = float(g.size)
        if total == 0:
            return 128
        p = hist / total
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        denom = omega * (1.0 - omega)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0.0)
        # A cleanly bimodal image gives an entire *plateau* of equally optimal
        # thresholds spanning the empty valley between the modes.  argmax would
        # return its left edge, putting the threshold right against the dark mode,
        # where a little sensor noise flips pixels to background.  Take the centre.
        best = float(sigma_b.max())
        if best <= 0.0:
            return 128
        optimal = np.nonzero(sigma_b >= best - 1e-9)[0]
        return int(round(float(optimal.mean())))

    @staticmethod
    def ink_mask(gray: GrayImage, threshold: Optional[int] = None,
                 adaptive: Any = "auto") -> GrayImage:
        """Boolean mask of "ink" pixels (dark on light).

        Otsu is the right default: it is cheap, global, and exact on an evenly-lit
        page.  It is also catastrophically wrong on an unevenly-lit one -- under a
        shadow gradient the darker *paper* falls below the global threshold, so a
        third of the page is classified as ink.  Everything built on this mask then
        fails together: ink coverage reads 40%, the skew search locks onto the
        shadow boundary instead of the text lines, and stroke width becomes
        meaningless.  Shadowed phone photographs are a primary input here, so the
        mask checks for that case and switches to a local (Sauvola) threshold,
        which is immune to it.

        :param gray: a greyscale or colour array
        :param threshold: force a fixed grey level, ``0..255``; pixels at or
            below it are ink.  ``None`` chooses a threshold automatically, which
            is almost always what you want.
        :param adaptive: ``"auto"`` (default) uses a local Sauvola threshold only
            when the page looks unevenly lit; ``True`` always does; ``False``
            always uses global Otsu.  Ignored when ``threshold`` is given.
        :returns: a boolean mask, ``True`` where there is ink
        """
        g = to_gray(gray)
        if threshold is not None:
            return g <= int(threshold)
        use_local = adaptive is True or (
            adaptive == "auto" and measure_illumination(g) < _UNEVEN_ILLUMINATION)
        if use_local and g.size >= 4096:
            h, w = image_shape(g)
            window = max(15, int(min(h, w) // 24) | 1)
            return binarize_array(g, method="sauvola", window=window) == 0
        return g <= otsu_threshold(g)

    @staticmethod
    def gradient_magnitude(gray: GrayImage) -> FloatArray:
        """Sobel gradient magnitude as float64.  Used by the sharpness metric.

        :param gray: a greyscale or colour array; colour is converted first
        :returns: a float array of edge strengths, the same shape as the input.
            Unnormalised, so compare it against itself rather than an absolute
            number.
        """
        np = _np()
        g = to_gray(gray).astype(np.float64)
        cv = _cv2()
        if cv is not None:
            gx = cv.Sobel(g, cv.CV_64F, 1, 0, ksize=3)
            gy = cv.Sobel(g, cv.CV_64F, 0, 1, ksize=3)
        else:
            p = np.pad(g, 1, mode="edge")
            # Separable Sobel: smooth with [1, 2, 1] across, differentiate with
            # [-1, 0, 1] along -- identical kernel, three cheap array adds.
            smooth_y = p[:-2, :] + 2.0 * p[1:-1, :] + p[2:, :]
            gx = smooth_y[:, 2:] - smooth_y[:, :-2]
            smooth_x = p[:, :-2] + 2.0 * p[:, 1:-1] + p[:, 2:]
            gy = smooth_x[2:, :] - smooth_x[:-2, :]
        return np.hypot(gx, gy)

    @staticmethod
    def morph_binary(mask: GrayImage, kh: int, kw: int, operation: str = "erode") -> GrayImage:
        """Binary morphology with a rectangular structuring element.

        Implemented with summed-area tables so that a 1x120 kernel (what line
        removal needs) costs the same as a 3x3 one.

        :param mask: a boolean or ``0``/``1`` array
        :param kh: kernel height in pixels
        :param kw: kernel width in pixels.  Asymmetry is the point: ``(1, 120)``
            finds horizontal rules and ignores everything else.
        :param operation: ``"erode"``, ``"dilate"``, ``"open"`` (erode then
            dilate, removing specks) or ``"close"`` (dilate then erode, filling
            gaps).  Named from the mask's point of view, so ``"dilate"`` grows
            the ``True`` region.
        :returns: the transformed boolean mask
        """
        np = _np()
        m = np.asarray(mask).astype(np.uint8)
        cv = _cv2()
        if cv is not None:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (max(1, kw), max(1, kh)))
            ops = {"erode": cv.MORPH_ERODE, "dilate": cv.MORPH_DILATE,
                   "open": cv.MORPH_OPEN, "close": cv.MORPH_CLOSE}
            if operation not in ops:
                raise ConfigError("unknown morphological operation %r" % operation)
            return cv.morphologyEx(m * 255, ops[operation], kernel).astype(bool)
        if operation == "open":
            return morph_binary(morph_binary(m, kh, kw, "erode"), kh, kw, "dilate")
        if operation == "close":
            return morph_binary(morph_binary(m, kh, kw, "dilate"), kh, kw, "erode")
        sums, counts = _rect_box_sum(m, kh, kw)
        if operation == "erode":
            return sums >= counts - 1e-6
        if operation == "dilate":
            return sums > 0.5
        raise ConfigError("unknown morphological operation %r" % operation)

    @staticmethod
    def estimate_background(gray: GrayImage, radius: Optional[int] = None) -> GrayImage:
        """Estimate the page background (paper + lighting) with the text erased.

        The kernel has to be *larger than the tallest glyph*, or the middle of a
        text line survives as background and the correction punches a hole through
        it.  Sized at 1/12 of the shorter side and clamped: big enough for body
        text at any sane DPI, small enough to still follow a shadow gradient.

        :param gray: a greyscale or colour array
        :param radius: kernel size in pixels, forced odd, minimum ``3``.
            ``None`` derives one from the page size, which is right unless the
            text is unusually large -- a kernel narrower than a glyph is tall
            leaves text in the background estimate.
        :returns: the estimated background as a ``uint8`` array, the same shape
            as the input.  Divide the page by it to flatten lighting, which is
            what :func:`Ops.normalize_illumination` does.
        """
        np = _np()
        g = to_gray(gray)
        h, w = image_shape(g)
        if radius is None:
            radius = int(clamp(min(h, w) // 12, 15, 101))
        radius = max(3, int(radius) | 1)

        cv = _cv2()
        if cv is not None:
            # RECT rather than ELLIPSE: OpenCV decomposes rectangular kernels into
            # separable passes, so a 101x101 close stays affordable on a full page.
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (radius, radius))
            closed = cv.morphologyEx(g, cv.MORPH_CLOSE, kernel)
            return cv.GaussianBlur(closed, (0, 0), radius / 4.0)

        # Fallback: block-maximum downsampling is grayscale dilation plus
        # subsampling, which erases dark text in one pass and costs almost nothing.
        # A median pass first is not optional -- the maximum over a 50x50 block is
        # the brightest noise spike in 2500 samples, so on a grainy scan the raw
        # block max sits several sigma above the true paper level and dividing by
        # it darkens the whole page unevenly.
        smoothed = median_blur(g, 3)
        block = max(2, radius // 2)
        pad_h = (-h) % block
        pad_w = (-w) % block
        padded = np.pad(smoothed, ((0, pad_h), (0, pad_w)), mode="edge")
        ph, pw = padded.shape
        coarse = padded.reshape(ph // block, block, pw // block, block).max(axis=(1, 3))
        if min(coarse.shape) >= 3:
            coarse = median_blur(ensure_uint8(coarse), 3)
        smoothed = box_blur(ensure_uint8(coarse), 1)
        return resize_image(smoothed, size=(w, h), interpolation="linear")

    @staticmethod
    def encode_png(img: ImageArray) -> bytes:
        """Encode an array as PNG bytes -- what vision backends actually send.

        Lossless, so prefer it for archival and for debugging output.  For a
        model call use :func:`Image.encode_jpeg`, which is several times smaller
        at a quality the page cannot tell apart.

        :param img: greyscale or colour
        :returns: the PNG bytes
        :raises MissingDependency: neither OpenCV nor Pillow is importable
        """
        arr = ensure_uint8(img)
        cv = _cv2()
        if cv is not None:
            rgb = to_rgb(arr) if arr.ndim == 3 else arr
            data = rgb[:, :, ::-1] if rgb.ndim == 3 else rgb  # RGB -> BGR for cv2
            ok, buf = cv.imencode(".png", data)
            if ok:
                return buf.tobytes()
        pil = _try_import("PIL.Image")
        if pil is not None:
            bio = io.BytesIO()
            pil.fromarray(arr).save(bio, format="PNG")
            return bio.getvalue()
        raise MissingDependency("PIL", "PNG encoding", "pillow")

    @staticmethod
    def encode_jpeg(img: ImageArray, quality: int = 88) -> bytes:
        """Encode as JPEG.  Preferred for VLM calls: a 300 DPI page as PNG is
        several megabytes, and the artefacts JPEG adds at q>=85 are well below what
        the scanner already introduced.

        :param img: greyscale or colour; greyscale is expanded to RGB
        :param quality: JPEG quality in ``1..100``.  ``88`` is the default and
            sits above the point where compression artefacts become visible to a
            recogniser; below ``75`` ringing around glyph edges starts to cost
            accuracy.
        :returns: the JPEG bytes
        :raises MissingDependency: neither OpenCV nor Pillow is importable
        """
        arr = ensure_uint8(img)
        cv = _cv2()
        if cv is not None:
            rgb = to_rgb(arr)
            ok, buf = cv.imencode(".jpg", rgb[:, :, ::-1],
                                  [int(cv.IMWRITE_JPEG_QUALITY), int(quality)])
            if ok:
                return buf.tobytes()
        pil = _try_import("PIL.Image")
        if pil is not None:
            bio = io.BytesIO()
            pil.fromarray(to_rgb(arr)).save(bio, format="JPEG", quality=int(quality))
            return bio.getvalue()
        raise MissingDependency("PIL", "JPEG encoding", "pillow")

    @staticmethod
    def decode_image(data: bytes) -> ImageArray:
        """Decode image bytes to a uint8 RGB/gray array.

        Raises :class:`IngestError` -- not a Pillow or OpenCV exception -- when the
        bytes are not a decodable image.  Callers batch-processing a directory or
        walking email attachments need one exception type to catch, or a single
        corrupt scan takes down the run.

        :param data: the encoded bytes -- PNG, JPEG, BMP, GIF or WebP.  Format is
            detected from content, so no filename is needed.
        :returns: a ``uint8`` array, greyscale ``(H, W)`` or RGB ``(H, W, 3)``
        :raises IngestError: the bytes are empty or not a decodable image
        :raises MissingDependency: neither OpenCV nor Pillow is importable
        """
        np = _np()
        if not data:
            raise IngestError("cannot decode an empty image")
        cv = _cv2()
        if cv is not None:
            try:
                arr = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_UNCHANGED)
            except Exception:
                arr = None
            if arr is not None and arr.size:
                if arr.ndim == 3:
                    if arr.shape[2] == 4:
                        arr = arr[:, :, :3]
                    arr = arr[:, :, ::-1]  # BGR -> RGB
                return ensure_uint8(np.ascontiguousarray(arr))
        pil = _try_import("PIL.Image")
        if pil is None:
            raise MissingDependency("PIL", "image decoding", "pillow")
        try:
            im = pil.open(io.BytesIO(data))
            im.load()
            if im.mode not in ("L", "RGB"):
                im = im.convert("RGB" if im.mode not in ("1", "I;16", "I", "F") else "L")
            return ensure_uint8(np.array(im))
        except Exception as exc:
            raise IngestError("could not decode image (%d bytes, first bytes %r): %s"
                              % (len(data), data[:8], exc))

    @staticmethod
    def save_image(img: ImageArray, path: str) -> str:
        """Write an array to disk, choosing the encoder from the extension.

        :param img: greyscale or colour
        :param path: destination.  ``.jpg``/``.jpeg`` encode as JPEG; every other
            extension, including none, encodes as PNG.
        :returns: ``path``, so it can be used inline
        :raises MissingDependency: neither OpenCV nor Pillow is importable
        :raises OSError: the file could not be written
        """
        ext = os.path.splitext(path)[1].lower()
        data = encode_jpeg(img) if ext in (".jpg", ".jpeg") else encode_png(img)
        with open(path, "wb") as fh:
            fh.write(data)
        return path


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Image from the inside.
# ``Image.x`` is the documented path; ``x`` is the one that already works.
ensure_uint8            = Image.ensure_uint8
to_gray                 = Image.to_gray
to_rgb                  = Image.to_rgb
window_stats            = Image.window_stats
box_blur                = Image.box_blur
gaussian_blur           = Image.gaussian_blur
median_blur             = Image.median_blur
resize_image            = Image.resize_image
rotate_image            = Image.rotate_image
row_projection_at_angle = Image.row_projection_at_angle
otsu_threshold          = Image.otsu_threshold
ink_mask                = Image.ink_mask
gradient_magnitude      = Image.gradient_magnitude
morph_binary            = Image.morph_binary
estimate_background     = Image.estimate_background
encode_png              = Image.encode_png
encode_jpeg             = Image.encode_jpeg
decode_image            = Image.decode_image
save_image              = Image.save_image


def _box_widths_for_gaussian(sigma: float, passes: int = 3) -> List[int]:
    """Odd box-filter widths whose repeated application matches ``sigma``.

    Three successive box filters converge on a Gaussian by the central limit
    theorem, but only if their widths are chosen so the *variances* add up to
    the target.  A box of width ``w`` has variance ``(w^2 - 1) / 12``; picking
    the radius by eye (say ``1.5 * sigma``) overshoots by about a factor of
    two, which quietly makes every fallback blur far heavier than the OpenCV
    path it is standing in for.

    This is the standard mixed-width construction: use the largest odd width
    that undershoots for ``m`` of the passes and the next odd width up for the
    rest, choosing ``m`` so the total variance lands on ``12 * sigma^2``.
    """
    target = 12.0 * sigma * sigma
    ideal = math.sqrt(target / passes + 1.0)
    lower = int(math.floor(ideal))
    if lower % 2 == 0:
        lower -= 1
    lower = max(1, lower)
    upper = lower + 2
    denominator = -4.0 * lower - 4.0
    if abs(denominator) < 1e-9:
        count = passes
    else:
        count = int(round((target - passes * lower * lower
                           - 4.0 * passes * lower - 3.0 * passes) / denominator))
    count = int(clamp(count, 0, passes))
    return [lower if i < count else upper for i in range(passes)]


#: Below this illumination score a global threshold is unsafe -- see `ink_mask`.
_UNEVEN_ILLUMINATION = 0.55


def _percentile_np(arr: ImageArray, q: float) -> float:
    """``q``-th percentile of an array, via NumPy when it is loaded."""
    np = _np()
    return float(np.percentile(arr, q))


# =============================================================================
# 5. Page quality measurement
# =============================================================================
#
# Preprocessing is channel equalisation, so it must be driven by measurement.
# Every metric below is normalised into [0, 1] (except skew, in degrees) so
# that thresholds are portable across scanners and DPIs, and so the same
# numbers can feed both the preprocessing policy and the confidence prior.


@dataclass
class QualityThresholds:
    """Tunable decision boundaries for quality verdicts and policy.

    These are starting points from scanned Indian hospital bills and court
    filings, not universal constants.  Recalibrate them against a labelled set
    (Layer 5) per document class; that is exactly the kind of finding that
    should ship in a version bump instead of living in one person's notebook.
    """

    blur_floor: float = 0.35            #: below this a page is visibly soft
    blur_unreadable: float = 0.08
    contrast_floor: float = 0.30
    contrast_unreadable: float = 0.10
    min_dpi: int = 300
    dpi_unreadable: int = 120
    skew_correct_deg: float = 0.4       #: worth correcting above this
    skew_max_deg: float = 15.0          #: beyond this it is a rotation, not skew
    noise_ceiling: float = 0.35
    illumination_floor: float = 0.55
    ink_blank: float = 0.002            #: below this the page carries no marks
    ink_saturated: float = 0.60         #: above this it is inverted or all-black


DEFAULT_THRESHOLDS = QualityThresholds()


#: Bracketing values for the two sharpness sub-scores, calibrated on synthetic
#: text degraded by known Gaussian blur and by known downsample-upsample loss;
#: see tests/test_quality.py, which asserts monotonicity across both.
_BLUR_RATIO_FLOOR, _BLUR_RATIO_CEIL = 0.35, 0.80
_BLUR_STEEP_FLOOR, _BLUR_STEEP_CEIL = 0.80, 3.50


class Quality(object):
    """Page quality measurement -- the evidence preprocessing acts on.

    Every measure is deliberately cheap and scale-aware, because the point
    is to decide what to *do* to a page, not to score it for its own sake.
    A measure that cost as much as the OCR it is protecting would be worse
    than no measure at all.
    """

    @staticmethod
    def measure_blur(gray: GrayImage) -> float:
        """Sharpness in ``[0, 1]``, from edge-transition width.

        Variance of the Laplacian is the textbook answer and it is the wrong one
        here.  It rises with noise, so a grainy scan scores *sharper* than a clean
        one, and it falls on sparse pages, so a crisp page with little ink scores
        blurry.  Both are everyday cases in scanned bills.

        Two independent sub-scores are computed instead, and the **worse of the two
        wins**, because each covers the other's blind spot:

        *Concentration* -- the fraction of edge pixels carrying strong gradient.  A
        sharp edge puts its whole intensity step into one or two pixels; blurring
        spreads it over many pixels of medium gradient.  Being a ratio it is
        immune to contrast and to noise, and it correctly catches a low-resolution
        page that was upsampled.  It *saturates* under extreme blur, where the
        gradient field becomes smooth and self-similar.

        *Steepness* -- peak gradient divided by the ink-to-paper step, i.e. the
        reciprocal of the edge's transition width in pixels.  It degrades cleanly
        all the way into heavy blur, but it is fooled by noise (which supplies
        spurious steep gradients) and by upsampling.

        The result is contrast-invariant by construction: a faded page and a dark
        one score the same, which is correct, because fading is
        :func:`Quality.measure_contrast`'s job to report, not this one's.

        Roughly: crisp print ~0.9, Gaussian sigma=1 ~0.6, sigma=2 ~0.45,
        sigma=4 ~0.3, sigma=8 ~0.05, a 120 DPI scan upsampled to 300 ~0.18.
        
        :param gray: the page as a greyscale array; colour input is converted
        :returns: sharpness in ``0.0..1.0``; ``1.0`` is crisp.  Compare against
            ``QualityThresholds.blur_floor`` (degraded) and ``blur_unreadable``.
        """
        np = _np()
        g = to_gray(gray)
        if g.size < 256:
            return 0.0
        mag = gradient_magnitude(g)
        peak = float(np.percentile(mag, 99.9))
        if peak < 5.0:  # no edges at all: a blank page is not "blurry"
            return 1.0 if float(np.std(g)) < 2.0 else 0.0
        weak = float((mag > 0.15 * peak).sum())
        if weak < 32:
            return 1.0

        concentration = float((mag > 0.50 * peak).sum()) / weak
        concentration_score = clamp(
            (concentration - _BLUR_RATIO_FLOOR) / (_BLUR_RATIO_CEIL - _BLUR_RATIO_FLOOR),
            0.0, 1.0)

        step = max(measure_contrast(g) * 255.0, 1.0)
        steepness = peak / step
        steepness_score = clamp(
            (steepness - _BLUR_STEEP_FLOOR) / (_BLUR_STEEP_CEIL - _BLUR_STEEP_FLOOR),
            0.0, 1.0)

        return round(float(min(concentration_score, steepness_score)), 4)

    @staticmethod
    def measure_contrast(gray: GrayImage) -> float:
        """Contrast in ``[0, 1]`` -- specifically, ink-to-paper separation.

        A generic 5th-95th percentile spread is wrong for documents: a perfectly
        crisp page that is 97% white paper has both percentiles sitting at 255 and
        reports zero contrast.  What actually matters to a recogniser is how far
        the ink sits from the paper, so we split on Otsu and measure the gap
        between the two populations, using inner percentiles on each side so that
        antialiased edge pixels do not flatter the result.

        Falls back to the percentile spread when the page has too little (or too
        much) ink for the split to mean anything.
        
        :param gray: the page as a greyscale array; colour input is converted
        :returns: ink-to-paper separation in ``0.0..1.0``.  A near-blank page
            reports high contrast rather than zero, since there is no ink whose
            separation could be poor.
        """
        np = _np()
        g = to_gray(gray)
        if g.size == 0:
            return 0.0
        mask = ink_mask(g)
        coverage = float(mask.mean())
        if 0.001 < coverage < 0.6:
            ink_level = float(np.percentile(g[mask], 60))
            paper_level = float(np.percentile(g[~mask], 40))
            return round(float(clamp((paper_level - ink_level) / 255.0, 0.0, 1.0)), 4)
        lo = _percentile_np(g, 5.0)
        hi = _percentile_np(g, 95.0)
        return round(float(clamp((hi - lo) / 255.0, 0.0, 1.0)), 4)

    @staticmethod
    def measure_ink_coverage(gray: GrayImage) -> float:
        """Fraction of pixels that are ink, using an illumination-aware mask.

        :param gray: the page as a greyscale array; colour input is converted
        :returns: inked fraction in ``0.0..1.0``.  Body text lands near ``0.05``,
            dense Devanagari near ``0.25``; above ``0.4`` suggests a negative
            scan -- see :func:`Ops.invert_if_dark`.
        """
        mask = ink_mask(gray)
        return round(float(mask.mean()), 5) if mask.size else 0.0

    @staticmethod
    def measure_noise(gray: GrayImage) -> float:
        """Speckle estimate in ``[0, 1]``.

        The high-frequency residual (image minus its median-filtered self) contains
        both text edges and noise.  Text edges are sparse and large; sensor noise is
        dense and small, so the *median* absolute residual tracks noise while being
        largely blind to text.
        
        :param gray: the page as a greyscale array; colour input is converted
        :returns: speckle in ``0.0..1.0``; ``0.0`` is clean.  Above
            ``QualityThresholds.noise_ceiling`` the page is treated as degraded
            and :func:`Ops.denoise` is reached for.
        """
        np = _np()
        g = to_gray(gray)
        if g.size < 64:
            return 0.0
        small = g if max(g.shape) <= 1200 else resize_image(g, scale=1200.0 / max(g.shape))
        smooth = median_blur(small, 3).astype(np.float64)
        residual = np.abs(small.astype(np.float64) - smooth)
        mad = float(np.median(residual))
        return round(float(clamp(mad / 12.0, 0.0, 1.0)), 4)

    @staticmethod
    def measure_illumination(gray: GrayImage) -> float:
        """Lighting evenness in ``[0, 1]``; 1 is flat, 0 is a hard shadow gradient.

        The page background is estimated by heavy downsampling (which averages text
        away) and its spread measured.  This is what catches phone photos with a
        shadow across one half -- the exact case where global binarisation destroys
        a third of the page.
        
        :param gray: the page as a greyscale array; colour input is converted
        :returns: evenness in ``0.0..1.0``; ``1.0`` is perfectly flat lighting
            and ``0.0`` a hard shadow gradient.  Below the
            ``QualityThresholds.illumination_floor`` cut-off,
            the op :func:`Ops.normalize_illumination` is applied.
        """
        np = _np()
        g = to_gray(gray).astype(np.float64)
        if g.size < 256:
            return 1.0
        tiny = resize_image(ensure_uint8(g), size=(16, 16), interpolation="area").astype(np.float64)
        spread = float(np.percentile(tiny, 90) - np.percentile(tiny, 10))
        return round(float(clamp(1.0 - spread / 120.0, 0.0, 1.0)), 4)

    @staticmethod
    def estimate_skew(gray: GrayImage, max_angle: float = 15.0, method: str = "auto",
                      coarse_step: float = 1.0, fine_step: float = 0.1,
                      min_improvement: float = 0.15) -> float:
        """Estimate the page's skew in degrees (positive = content tilted clockwise).

        ``method``:

        ``projection``
            Coarse-to-fine search maximising the variance of the horizontal
            projection profile.  Text lines align into sharp peaks only at the
            correct angle.  Robust on sparse pages, needs no OpenCV, and is the
            default because it degrades gracefully on forms and tables.
        ``hough``
            Probabilistic Hough transform over Canny edges, taking the median angle
            of near-horizontal segments.  Excellent when ruled lines exist.
        ``minarearect``
            Minimum-area rectangle of all ink pixels.  Fast, but fooled by a single
            stray margin mark, so it is only used inside ``auto``'s median.
        ``auto``
            Median of whichever of the above are available -- a cheap consensus
            that avoids each method's individual failure mode.

        ``min_improvement`` is how much better than the straight-page hypothesis a
        candidate angle must score before it is believed; see ``by_projection``.

        Returns ``0.0`` when the page has too little ink to judge, or when no angle
        is convincingly better than leaving the page alone.

        :param gray: the page as a greyscale array; colour input is converted
        :param max_angle: widest tilt to search, in degrees either way.  Beyond
            ``15`` a page is not skewed but rotated -- a different problem, and
            searching that far mostly finds table rules.
        :param method: ``"auto"``, ``"projection"``, ``"hough"``, or
            ``"minarearect"``, each described above.  ``"auto"`` takes the median
            of whichever are available.
        :param coarse_step: degrees per step in the first projection pass.
            ``1.0`` is ample; smaller only costs time, since the fine pass
            refines whatever the coarse pass found.
        :param fine_step: degrees per step in the refinement pass.  ``0.1`` is
            below what deskewing can act on anyway -- :func:`Ops.deskew` declines
            under ``0.4`` degrees.
        :param min_improvement: how much better than the straight-page hypothesis
            a candidate must score to be believed, as a relative margin.  ``0.15``
            keeps a page that is genuinely straight from being rotated by noise.
        :returns: the skew in degrees, positive meaning content tilted clockwise;
            ``0.0`` when the page is straight, nearly blank, or ink-saturated
        """
        np = _np()
        g = to_gray(gray)
        if g.size < 1024:
            return 0.0
        # Work small: skew is a global property and 800px is plenty of evidence.
        scale = 800.0 / max(g.shape)
        small = resize_image(g, scale=scale) if scale < 1.0 else g
        mask = ink_mask(small)
        if mask.mean() < 0.0005 or mask.mean() > 0.9:
            return 0.0

        estimates: List[float] = []

        def by_projection() -> float:
            """Coarse-to-fine search for the angle with the sharpest projection.

            Returns ``0.0`` when no angle beats the straight-page hypothesis by
            ``min_improvement``.
            """
            # Only ink pixels can contribute to the profile, and they are a few
            # percent of the page, so shifting the *coordinates* of the ink is
            # dramatically cheaper than shearing the whole raster once per angle --
            # and this search evaluates ~50 angles.
            height, width = mask.shape[:2]
            ink_y, ink_x = np.nonzero(mask)
            if ink_y.size < 32:
                return 0.0
            centred_x = (ink_x - width / 2.0)

            def score(angle: float) -> float:
                """Variance of the row-projection profile after shearing by ``angle``."""
                # Variance of the projection peaks sharply at the true angle:
                # text lines only stack into tall, narrow bands when horizontal.
                # Forward map, so the sign is the negative of the inverse-mapped
                # shear in row_projection_at_angle -- the two must agree.
                shifted = ink_y - np.round(centred_x * math.tan(math.radians(angle)))
                # Ink sheared off the page is *discarded*, not clamped.  Clamping
                # heaps every out-of-range pixel onto the first or last row, and
                # since the amount sheared off grows with the angle, that artefact
                # hands ever-larger variance to ever-more-extreme angles -- the
                # search then reliably reports 15 degrees for a straight page.
                inside = (shifted >= 0) & (shifted < height)
                if not inside.any():
                    return 0.0
                profile = np.bincount(shifted[inside].astype(np.int64), minlength=height)
                total = float(profile.sum())
                if total <= 0:
                    return 0.0
                # Normalise to a distribution so angles that retain different
                # amounts of ink remain comparable.
                return float(np.var(profile / total))

            # Scan outward from zero so that ties keep the smaller angle.
            candidates = [0.0]
            step = coarse_step
            while step <= max_angle + 1e-9:
                candidates.extend([-step, step])
                step += coarse_step

            baseline = score(0.0)
            best_angle, best_score = 0.0, baseline
            for angle in candidates:
                s = score(angle)
                if s > best_score:
                    best_score, best_angle = s, angle

            a = max(-max_angle, best_angle - coarse_step)
            stop = min(max_angle, best_angle + coarse_step)
            refined_angle, refined_score = best_angle, best_score
            while a <= stop + 1e-9:
                s = score(a)
                if s > refined_score * (1.0 + 1e-9):
                    refined_score, refined_angle = s, a
                a += fine_step

            # A "pages are roughly straight" prior, and it earns its keep.  On a
            # noisy scan the ink mask picks up a few pathological dense rows, and
            # at an extreme shear those smear into a concentration that can just
            # beat the genuine text lines -- yielding a confident 15-degree answer
            # for a page skewed by two.  Rotating on that is far more destructive
            # than leaving a small skew uncorrected, so an angle has to beat the
            # straight-page hypothesis by a clear margin before it is believed.
            if baseline > 0 and refined_score <= baseline * (1.0 + min_improvement):
                return 0.0
            return refined_angle

        def by_hough() -> Optional[float]:
            """Median angle of near-horizontal Hough segments, or ``None``.

            ``None`` means "no opinion" -- OpenCV missing, no lines found, or too few
            to be worth a median -- and the caller drops it from the consensus.
            """
            cv = _cv2()
            if cv is None:
                return None
            edges = cv.Canny(small, 50, 150, apertureSize=3)
            min_len = max(30, int(small.shape[1] * 0.25))
            lines = cv.HoughLinesP(edges, 1, math.pi / 720.0, threshold=80,
                                   minLineLength=min_len, maxLineGap=12)
            if lines is None:
                return None
            angles = []
            # OpenCV has shipped both (N, 1, 4) and (N, 4) for this over the years.
            flat = lines.reshape(-1, 4)
            for line in flat[:600]:
                x1, y1, x2, y2 = line
                if x2 == x1:
                    continue
                ang = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
                if abs(ang) <= max_angle:
                    angles.append(ang)
            if len(angles) < 3:
                return None
            return float(statistics.median(angles))

        def by_minarearect() -> Optional[float]:
            """Angle of the minimum-area rectangle around all ink, or ``None``.

            ``None`` means "no opinion" (see ``by_hough``).
            """
            cv = _cv2()
            if cv is None:
                return None
            pts = cv.findNonZero((mask.astype(np.uint8)) * 255)
            if pts is None or len(pts) < 50:
                return None
            angle = cv.minAreaRect(pts)[-1]
            # OpenCV's convention flipped at 4.5; normalise into (-45, 45].
            while angle > 45.0:
                angle -= 90.0
            while angle <= -45.0:
                angle += 90.0
            if abs(angle) > max_angle:
                return None
            return float(angle)

        if method in ("projection", "auto"):
            estimates.append(by_projection())
        if method in ("hough", "auto"):
            v = by_hough()
            if v is not None:
                estimates.append(v)
        if method in ("minarearect", "auto"):
            v = by_minarearect()
            if v is not None:
                estimates.append(v)
        if method not in ("projection", "hough", "minarearect", "auto"):
            raise ConfigError("unknown skew method %r" % method)
        if not estimates:
            return 0.0
        return round(float(clamp(statistics.median(estimates), -max_angle, max_angle)), 3)

    @staticmethod
    def estimate_stroke_width(gray: GrayImage) -> float:
        """Mean ink stroke thickness in pixels, from the area-to-perimeter ratio.

        For an elongated shape, ``2 * area / perimeter`` is its width.  Cheap and
        exact on clean print -- but it *inflates under blur*, because a soft edge
        puts the Otsu threshold outside the true stroke and the halo is counted as
        ink.  Callers must not use it as a resolution estimate on a soft page;
        see :func:`Quality.estimate_line_pitch`.
        
        :param gray: the page as a greyscale array; colour input is converted
        :returns: mean stroke thickness in pixels, or ``0.0`` when the page has
            too little ink.  Roughly 3 px for body text at 300 DPI -- but it
            inflates under blur, so prefer line pitch for resolution.
        """
        np = _np()
        mask = ink_mask(to_gray(gray))
        area = float(mask.sum())
        if area < 200 or mask.mean() > 0.5:
            return 0.0
        m = mask.astype(np.uint8)
        padded = np.pad(m, 1, mode="constant")
        neighbours = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                      + padded[1:-1, :-2] + padded[1:-1, 2:])
        perimeter = float(((m == 1) & (neighbours < 4)).sum())
        if perimeter < 1:
            return 0.0
        return 2.0 * area / perimeter

    @staticmethod
    def estimate_line_pitch(gray: GrayImage, min_pitch: int = 8, max_pitch: int = 200) -> float:
        """Dominant text-line spacing in pixels, via autocorrelation, or ``0.0``.

        Blur, fading and threshold choice all move stroke width around; none of
        them move the distance between one baseline and the next.  That makes line
        pitch the sturdiest resolution cue a page offers, and it is what
        :func:`Quality.estimate_effective_dpi` prefers.

        Returns ``0.0`` when the projection has no convincing periodicity -- a
        title page, a photograph, or a form with irregular row heights.

        :param gray: the page as a greyscale array
        :param min_pitch: shortest spacing considered, in pixels.  ``8`` is below
            body text at any usable resolution; raising it rejects the harmonics
            that dense text can produce.
        :param max_pitch: longest spacing considered, in pixels.  ``200`` covers
            double-spaced large print at 600 DPI.
        :returns: the dominant line spacing in pixels, or ``0.0`` when the page
            has too little ink or no convincing periodicity
        """
        np = _np()
        g = to_gray(gray)
        mask = ink_mask(g).astype(np.float64)
        if mask.mean() < 0.002:
            return 0.0
        profile = mask.sum(axis=1)
        profile = profile - profile.mean()
        if float(np.std(profile)) < 1e-6:
            return 0.0
        correlation = np.correlate(profile, profile, mode="full")[len(profile) - 1:]
        if correlation[0] <= 0:
            return 0.0
        correlation = correlation / correlation[0]
        lo = int(min_pitch)
        hi = min(int(max_pitch), len(correlation) - 1)
        if hi <= lo + 2:
            return 0.0
        window = correlation[lo:hi]
        # Genuine local maxima only.  A mean-plus-k-sigma test does not work here:
        # a strongly periodic profile inflates its own standard deviation with
        # harmonics, so the truer the periodicity the harder such a test is to
        # pass.  Prominence above the preceding trough is the right measure.
        peaks = [i for i in range(1, len(window) - 1)
                 if window[i] > window[i - 1] and window[i] >= window[i + 1]]
        if not peaks:
            return 0.0
        highest = max(float(window[i]) for i in peaks)
        if highest < 0.30:
            return 0.0
        fundamental = max(peaks, key=lambda i: float(window[i]))

        # Sub-harmonic correction.  Blank lines between paragraphs make a page
        # partly periodic at twice the line pitch, and that longer lag can
        # correlate *more* strongly than the true one -- reporting 2x the spacing
        # would then halve the DPI estimate.  If a real peak sits near an integer
        # fraction of the winner and is respectably strong, it is the fundamental.
        for divisor in (4, 3, 2):
            target = fundamental / float(divisor)
            tolerance = max(1.5, target * 0.15)
            nearby = [i for i in peaks if abs(i - target) <= tolerance]
            if nearby:
                candidate = max(nearby, key=lambda i: float(window[i]))
                if float(window[candidate]) >= 0.6 * highest:
                    fundamental = candidate
                    break

        trough = float(np.min(window[:fundamental + 1]))
        if float(window[fundamental]) - trough < 0.25:
            return 0.0
        return float(lo + fundamental)

    @staticmethod
    def estimate_effective_dpi(gray: GrayImage, fallback: int = 0) -> int:
        """Estimate the *content* resolution of a page image.

        A page can be rendered at 600 DPI and still carry only 120 DPI of signal,
        because it was scanned once at 120 and upsampled by three tools since.
        Rendering DPI is therefore not resolution, and this is the number that
        should drive :func:`Ops.ensure_dpi` and the confidence prior.

        Two independent cues are used.  Line pitch is preferred: it survives blur,
        fading and threshold choice, all of which distort stroke width.  Stroke
        width is the fallback for pages with no regular text lines, and the two are
        averaged when they roughly agree, which tightens the estimate on ordinary
        prose.

        Returns ``fallback`` when the page carries too little ink to judge.

        :param gray: the page as a greyscale array
        :param fallback: returned when neither cue yields an estimate.  Pass the
            page's nominal ``raster_dpi`` so callers always get a usable number.
        :returns: estimated content DPI, clamped to ``30..1200``.  Compare it
            against ``page.raster_dpi``: a large gap means the page was upsampled
            somewhere and carries less signal than its size suggests.
        """
        g = to_gray(gray)
        pitch = estimate_line_pitch(g)
        pitch_dpi = pitch / _INCHES_PER_TEXT_LINE if pitch else 0.0

        stroke = estimate_stroke_width(g)
        stroke_dpi = 300.0 * stroke / _STROKE_PX_AT_300 if stroke else 0.0

        if pitch_dpi and stroke_dpi and 0.6 <= stroke_dpi / pitch_dpi <= 1.6:
            estimate = (pitch_dpi + stroke_dpi) / 2.0
        elif pitch_dpi:
            estimate = pitch_dpi
        elif stroke_dpi:
            estimate = stroke_dpi
        else:
            return int(fallback)
        return int(clamp(round(estimate), 30, 1200))

    @staticmethod
    def classify_quality(q: PageQuality,
                         thresholds: Optional[QualityThresholds] = None) -> Verdict:
        """Turn measurements into ``clean`` / ``degraded`` / ``unreadable``.

        ``unreadable`` is a routing signal, not a discard: those pages get the
        vision model and a confidence penalty rather than being dropped, because a
        human reviewer would rather see a low-confidence guess than a blank.

        :param q: the measured :class:`PageQuality`
        :param thresholds: cut-offs to compare against.  ``None`` uses the
            defaults in :data:`DEFAULT_THRESHOLDS`
        :returns: a :class:`Verdict`.  A near-blank page reports ``CLEAN`` rather
            than ``UNREADABLE`` -- there is nothing on it to misread, and calling
            it unreadable would route empty pages to the expensive backend.
        """
        th = thresholds or DEFAULT_THRESHOLDS
        if q.ink_coverage < th.ink_blank:
            return Verdict.CLEAN  # blank, but not degraded -- nothing to misread
        fatal = (q.blur < th.blur_unreadable
                 or q.contrast < th.contrast_unreadable
                 or (q.effective_dpi and q.effective_dpi < th.dpi_unreadable)
                 or q.ink_coverage > th.ink_saturated)
        if fatal:
            return Verdict.UNREADABLE
        degraded = (q.blur < th.blur_floor
                    or q.contrast < th.contrast_floor
                    or (q.effective_dpi and q.effective_dpi < th.min_dpi)
                    or q.noise > th.noise_ceiling
                    or q.illumination < th.illumination_floor
                    or abs(q.skew_deg) > th.skew_correct_deg)
        return Verdict.DEGRADED if degraded else Verdict.CLEAN

    @staticmethod
    def measure_quality(img: ImageArray, raster_dpi: int = 0,
                        thresholds: Optional[QualityThresholds] = None,
                        skew_method: str = "auto", max_skew: float = 15.0) -> PageQuality:
        """Measure every channel-degradation signal for one page image.

        The measurement half of adaptive preprocessing: policies read the result
        and reach only for the ops a page's actual degradation warrants.

        :param img: the page raster, greyscale or colour
        :param raster_dpi: the resolution the image was rendered at, recorded on
            the result and used as the fallback for ``effective_dpi``.  ``0``
            means unknown.
        :param thresholds: cut-offs for the verdict.  ``None`` uses the
            defaults in :data:`DEFAULT_THRESHOLDS`
        :param skew_method: passed to :func:`Quality.estimate_skew` -- ``"auto"``,
            ``"projection"``, ``"hough"``, or ``"minarearect"``
        :param max_skew: widest skew to search, in degrees
        :returns: a :class:`PageQuality` carrying blur, skew, contrast, ink
            coverage, noise, illumination, effective DPI and an overall verdict
        """
        gray = to_gray(img)
        q = PageQuality(
            blur=measure_blur(gray),
            skew_deg=estimate_skew(gray, max_angle=max_skew, method=skew_method),
            contrast=measure_contrast(gray),
            ink_coverage=measure_ink_coverage(gray),
            noise=measure_noise(gray),
            illumination=measure_illumination(gray),
            raster_dpi=int(raster_dpi),
        )
        q.effective_dpi = estimate_effective_dpi(gray, fallback=int(raster_dpi))
        h, w = image_shape(gray)
        q.extra["pixels"] = float(w * h)
        q.verdict = classify_quality(q, thresholds)
        return q

    @staticmethod
    def measure_page(page: Page, dpi: Optional[int] = None,
                     thresholds: Optional[QualityThresholds] = None, **kwargs: Any) -> Page:
        """Measure ``page`` in place and return it.  No-op for pages without rasters.

        :param page: the page to measure; ``page.quality`` is replaced and the
            measurement recorded in ``page.history``
        :param dpi: render the raster at this DPI for measuring; ``None`` uses
            whatever the page already has
        :param thresholds: cut-offs for the verdict.  ``None`` uses the
            defaults in :data:`DEFAULT_THRESHOLDS`
        :param kwargs: forwarded to :func:`Quality.measure_quality` --
            ``skew_method``, ``max_skew``
        :returns: the same page, mutated.  A page with no raster is returned
            untouched, its quality left at defaults.
        """
        if not page.has_raster():
            return page
        with Timer() as t:
            img = page.raster(dpi)
            page.quality = measure_quality(img, raster_dpi=page.raster_dpi,
                                           thresholds=thresholds, **kwargs)
        page.record("measure_quality", {"dpi": page.raster_dpi}, t.ms)
        return page

    @staticmethod
    def measure_document(doc: Document, dpi: Optional[int] = None,
                         thresholds: Optional[QualityThresholds] = None, max_workers: int = 0,
                         **kwargs: Any) -> Document:
        """Measure every page's quality.  Returns the same Document, mutated.

        :param doc: the document to measure, page by page
        :param dpi: render DPI for measuring; ``None`` uses each page's own
        :param thresholds: cut-offs for the verdict.  ``None`` uses the
            defaults in :data:`DEFAULT_THRESHOLDS`
        :param max_workers: threads to measure with.  ``0`` runs serially.
        :param kwargs: forwarded to :func:`Quality.measure_quality`
        :returns: the same document, mutated.  :func:`preprocess` does this for
            you -- call it directly to inspect quality without changing pixels,
            which is what ``docpipe quality`` does.
        """
        _map_maybe_parallel(
            lambda p: measure_page(p, dpi=dpi, thresholds=thresholds, **kwargs),
            doc.pages, max_workers, "measure")
        return doc


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Quality from the inside.
# ``Quality.x`` is the documented path; ``x`` is the one that already works.
measure_blur           = Quality.measure_blur
measure_contrast       = Quality.measure_contrast
measure_ink_coverage   = Quality.measure_ink_coverage
measure_noise          = Quality.measure_noise
measure_illumination   = Quality.measure_illumination
estimate_skew          = Quality.estimate_skew
estimate_stroke_width  = Quality.estimate_stroke_width
estimate_line_pitch    = Quality.estimate_line_pitch
estimate_effective_dpi = Quality.estimate_effective_dpi
classify_quality       = Quality.classify_quality
measure_quality        = Quality.measure_quality
measure_page           = Quality.measure_page
measure_document       = Quality.measure_document


#: Body text is typically set with ~1/6 inch of leading (12pt on 10pt type),
#: which is what converts a measured line pitch into a resolution.
_INCHES_PER_TEXT_LINE = 1.0 / 6.0

#: Body text strokes run ~3.2px wide at 300 DPI.
_STROKE_PX_AT_300 = 3.2


# =============================================================================
# 6. Layer 0 -- ingest and normalisation
# =============================================================================
#
# Ingest's job is to produce a Document whose pages know their geometry, know
# whether they carry a trustworthy text layer, and can render themselves on
# demand.  It does *not* rasterise anything: a 300-page bundle must survive
# ingest in a few megabytes of RAM.


#: Magic-byte signatures.  Extensions lie -- especially on email attachments,
#: where a JPEG named ``scan.pdf`` is a weekly occurrence.
_MAGIC = [
    (b"%PDF-", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"BM", "bmp"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"\x00\x00\x01\x00", "ico"),
]

IMAGE_FORMATS = frozenset(["png", "jpeg", "tiff", "bmp", "gif", "webp", "ico"])


#: A page needs at least this many extracted characters before we will believe
#: its text layer.  Scanned pages routinely carry a handful of characters from
#: a header stamp or a producer watermark; trusting those and skipping OCR is
#: the single most damaging false positive in this whole library.
MIN_NATIVE_CHARS = 48


class Ingest(object):
    """Layer 0 -- bytes of unknown provenance to a :class:`Document`.

    Ingest never rasterises eagerly.  Pages carry a lazy ``raster()``
    provider instead, because a 300-page bundle at 400 DPI rendered up
    front is tens of gigabytes for a job that may only need page 2.
    """

    @staticmethod
    def sniff_format(data: bytes, filename: str = "") -> str:
        """Identify a document format from its bytes, falling back to its name.

        Content first, extension last, because in practice the extension lies:
        scanners emit ``.pdf`` files that are really TIFFs, and mail gateways
        strip extensions entirely.

        :param data: the leading bytes of the file; the first 2 KB are enough
        :param filename: used only for its extension, and only when the magic
            bytes were inconclusive
        :returns: one of ``"pdf"``, ``"png"``, ``"jpeg"``, ``"tiff"``, ``"bmp"``,
            ``"gif"``, ``"webp"``, ``"eml"``, or ``"unknown"``.  A ``.msg`` file
            reports ``"eml"``, since it takes the same path.
        """
        head = data[:64] if data else b""
        for magic, name in _MAGIC:
            if head.startswith(magic):
                return name
        if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
        # PDFs occasionally carry junk before %PDF- (mail gateways, bad scanners).
        idx = data[:2048].find(b"%PDF-") if data else -1
        if idx > 0:
            return "pdf"
        if head.lstrip()[:5].lower() in (b"from:", b"retur", b"recei", b"messa", b"mime-",
                                         b"date:", b"subje"):
            return "eml"
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext in ("jpg", "jpeg"):
            return "jpeg"
        if ext in ("tif", "tiff"):
            return "tiff"
        if ext in ("pdf", "png", "bmp", "gif", "webp", "eml", "msg"):
            return "eml" if ext == "msg" else ext
        return "unknown"

    @staticmethod
    def classify_page_kind(char_count: int, image_area_ratio: float, has_images: bool,
                           min_chars: int = MIN_NATIVE_CHARS,
                           ink_hint: Optional[float] = None) -> PageKind:
        """Decide how a page stores its text, from cheap PDF-level evidence.

        ``image_area_ratio`` is the fraction of the page covered by embedded raster
        images.  The interesting case is HYBRID: a native-text page with a scanned
        insert pasted in (a photographed bill stapled into a typed claim form).
        Those pages need *both* paths, and treating them as native silently loses
        the insert -- which is usually where the numbers are.

        :param char_count: characters the PDF's text layer reports for this page
        :param image_area_ratio: fraction of the page covered by embedded raster
            images, in ``0.0..1.0``.  Above ``0.35`` alongside real text means
            HYBRID -- a scanned insert on a typed form.
        :param has_images: whether the page embeds any raster image at all
        :param min_chars: characters below which the text layer is not trusted.
            Defaults to :data:`MIN_NATIVE_CHARS`.  Scanned pages routinely carry
            a handful of stray characters from a header stamp, so the floor
            cannot be ``1``.
        :param ink_hint: measured ink coverage in ``0.0..1.0``, when known.  Lets
            a page with no text and no embedded image still be called SCANNED
            rather than BLANK, which is the case for a page whose content is one
            large drawing.
        :returns: the :class:`PageKind` -- ``DIGITAL_NATIVE``, ``HYBRID``,
            ``SCANNED``, or ``BLANK``.  This drives routing, as decided
            by :func:`default_router`.
        """
        if char_count >= min_chars:
            if has_images and image_area_ratio > 0.35:
                return PageKind.HYBRID
            return PageKind.DIGITAL_NATIVE
        if has_images or (ink_hint is not None and ink_hint > 0.002):
            return PageKind.SCANNED
        if char_count > 0:
            return PageKind.SCANNED
        return PageKind.BLANK

    @staticmethod
    def ingest_pdf(source: Source, password: Optional[str] = None, max_pages: int = 0,
                   page_range: Optional[Sequence[int]] = None, render_dpi: int = DEFAULT_DPI,
                   extract_text: bool = True,
                   min_native_chars: int = MIN_NATIVE_CHARS) -> Document:
        """Open a PDF, read any native text layer, and attach lazy page renderers.

        Handles the three things that break naive PDF ingest in production:
        encrypted-but-openable files (empty owner password), files with leading
        junk before ``%PDF-``, and structurally damaged files that PyMuPDF can
        repair on a second pass.

        :param source: path, path-like, raw ``bytes``, or an open binary file
        :param password: for an encrypted PDF.  Files encrypted with an empty
            owner password -- the common case for bank and hospital statements --
            open without this.
        :param max_pages: stop after this many pages; ``0`` means all of them.
            Applied after ``page_range``.
        :param page_range: zero-based page indices to ingest, e.g.
            ``range(10, 20)`` or ``[0, 5, 9]``.  ``None`` takes every page.
        :param render_dpi: resolution for the lazy page renderers.  ``300`` is
            the floor for reliable OCR, ``400`` helps on small print.  Nothing is
            rendered until a page's raster is actually asked for -- a 300-page
            bundle at 400 DPI would be tens of gigabytes eagerly.
        :param extract_text: read the embedded text layer.  Leave it on: it is
            nearly free, and it is what lets :func:`default_router` route a
            digital page to the exact, free ``"pymupdf"`` read.
        :param min_native_chars: per-page character floor for trusting that text
            layer; passed to :func:`Ingest.classify_page_kind`.
        :returns: a :class:`Document` whose pages carry native spans where they
            exist and a lazy raster provider throughout
        :raises IngestError: the file could not be opened even after a repair
            pass, or the password was wrong
        """
        fitz = _fitz()
        if fitz is None:
            return _ingest_pdf_pdfium(source, max_pages=max_pages, page_range=page_range,
                                      render_dpi=render_dpi)

        data, uri = _read_source(source)
        junk = data[:2048].find(b"%PDF-")
        if junk > 0:
            data = data[junk:]

        try:
            fdoc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise IngestError("could not open %s as PDF: %s" % (uri, exc))

        doc = Document(source_uri=uri)
        if getattr(fdoc, "needs_pass", False):
            for candidate in ([password] if password else []) + ["", "owner"]:
                try:
                    if fdoc.authenticate(candidate or ""):
                        break
                except Exception:
                    continue
            else:
                raise IngestError("%s is encrypted and no supplied password worked" % uri)
            if password is None:
                doc.warn("PDF was encrypted; opened with an empty password")

        if getattr(fdoc, "is_repaired", False):
            doc.warn("PDF was structurally damaged and has been repaired in memory")

        total = fdoc.page_count
        indices = list(page_range) if page_range is not None else list(range(total))
        indices = [i for i in indices if 0 <= i < total]
        if max_pages and len(indices) > max_pages:
            doc.warn("truncated to first %d of %d pages" % (max_pages, total))
            indices = indices[:max_pages]

        doc.meta.update({
            "format": "pdf",
            "page_count": total,
            "producer": (fdoc.metadata or {}).get("producer", ""),
            "creator": (fdoc.metadata or {}).get("creator", ""),
            "title": (fdoc.metadata or {}).get("title", ""),
            "encrypted": bool(getattr(fdoc, "needs_pass", False)),
            "checksum": stable_hash(data),
            "ingested_at": _utcnow(),
        })

        for pno in indices:
            fpage = fdoc[pno]
            rect = fpage.rect
            page = Page(index=pno, width_pt=float(rect.width), height_pt=float(rect.height),
                        rotation=int(getattr(fpage, "rotation", 0) or 0))
            page.meta["source"] = uri

            char_count = 0
            if extract_text:
                try:
                    for w in fpage.get_text("words"):
                        x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
                        if not str(word).strip():
                            continue
                        block_no = int(w[5]) if len(w) > 5 else None
                        line_no = int(w[6]) if len(w) > 6 else None
                        page.spans.append(TextSpan(
                            text=str(word), bbox=BBox(pno, float(x0), float(y0),
                                                      float(x1), float(y1)),
                            source="pymupdf", confidence=None,
                            line=line_no, block=block_no))
                        char_count += len(str(word).strip())
                except Exception as exc:
                    doc.warn("text extraction failed on page %d: %s" % (pno, exc))

            image_area = 0.0
            has_images = False
            try:
                for info in fpage.get_images(full=True):
                    has_images = True
                    try:
                        for r in fpage.get_image_rects(info[0]):
                            image_area += float(r.width) * float(r.height)
                    except Exception:
                        pass
            except Exception:
                pass
            page_area = float(rect.width) * float(rect.height) or 1.0
            ratio = clamp(image_area / page_area, 0.0, 1.0)

            page.kind = classify_page_kind(char_count, ratio, has_images,
                                           min_chars=min_native_chars)
            page.meta["image_area_ratio"] = round(ratio, 4)
            page.meta["native_chars"] = char_count
            page.quality.raster_dpi = render_dpi
            # A native text layer is exact by construction; skip the (expensive,
            # and here meaningless) raster measurement unless something rasterises.
            if page.kind is PageKind.DIGITAL_NATIVE:
                page.quality.effective_dpi = max(render_dpi, DEFAULT_DPI)
                page.quality.verdict = Verdict.CLEAN
                page.quality.blur = 1.0
                page.quality.contrast = 1.0

            page.set_raster_provider(_pdf_raster_provider(fdoc, pno), dpi=render_dpi)
            doc.pages.append(page)

        if not doc.pages:
            doc.warn("PDF contained no pages in the requested range")
        return doc

    @staticmethod
    def ingest_image(source: Source, dpi: Optional[int] = None,
                     page_index: int = 0) -> Document:
        """Ingest a single-page image (PNG/JPEG/BMP/WebP) as a one-page Document.

        :param source: path, path-like, raw ``bytes``, or an open binary file
        :param dpi: the image's true resolution.  ``None`` reads it from the
            file's metadata and falls back to :data:`DEFAULT_IMAGE_DPI`.  Worth
            passing explicitly for phone photos, whose EXIF DPI is meaningless --
            every DPI-relative threshold downstream depends on this number.
        :param page_index: the index this page takes in the document, for
            assembling a multi-page document from separate image files
        :returns: a one-page :class:`Document` with the raster already attached
        :raises IngestError: the bytes could not be decoded as an image
        """
        data, uri = _read_source(source)
        img = decode_image(data)
        detected_dpi = dpi or _image_dpi_hint(data) or DEFAULT_IMAGE_DPI
        doc = Document(source_uri=uri, meta={
            "format": sniff_format(data, uri), "page_count": 1,
            "checksum": stable_hash(data), "ingested_at": _utcnow(),
            "declared_dpi": detected_dpi,
        })
        doc.pages.append(_page_from_array(img, page_index, detected_dpi, uri))
        return doc

    @staticmethod
    def ingest_tiff(source: Source, dpi: Optional[int] = None) -> Document:
        """Ingest a (possibly multi-page) TIFF.  Fax archives are full of these.

        Needs Pillow for multi-page files; without it, only the first frame is
        read, via :func:`Ingest.ingest_image`.

        :param source: path, path-like, raw ``bytes``, or an open binary file
        :param dpi: true resolution; ``None`` reads the TIFF tag and falls back
            to :data:`DEFAULT_IMAGE_DPI`.  Fax TIFFs commonly declare 204x98,
            which is anisotropic and worth overriding.
        :returns: a :class:`Document` with one page per frame
        :raises IngestError: the file could not be opened as a TIFF
        """
        pil = _try_import("PIL.Image")
        data, uri = _read_source(source)
        if pil is None:
            return ingest_image(data, dpi=dpi)
        np = _np()
        try:
            im = pil.open(io.BytesIO(data))
            frames = getattr(im, "n_frames", 1)
        except Exception as exc:
            raise IngestError("could not open %s as TIFF: %s" % (uri, exc))
        detected_dpi = dpi or _image_dpi_hint(data) or DEFAULT_IMAGE_DPI
        doc = Document(source_uri=uri, meta={
            "format": "tiff", "page_count": frames, "checksum": stable_hash(data),
            "ingested_at": _utcnow(), "declared_dpi": detected_dpi,
        })
        for i in range(frames):
            im.seek(i)
            frame = im.convert("L" if im.mode in ("1", "L", "I;16", "I") else "RGB")
            arr = ensure_uint8(np.array(frame))
            doc.pages.append(_page_from_array(arr, i, detected_dpi, uri))
        return doc

    @staticmethod
    def ingest_email(source: Source, dpi: Optional[int] = None, include_body: bool = True,
                     max_attachments: int = 50) -> Document:
        """Ingest an ``.eml`` message: every readable attachment becomes pages.

        Inbound claim registration lives on this path.  Attachments arrive as
        native PDFs, scans, and images pasted into the body, all in one message,
        and each part has to be routed on its actual bytes rather than its
        filename.  Pages carry ``meta['attachment']`` so a downstream reviewer can
        be told *which* attachment a field came from.

        :param source: path, path-like, raw ``bytes``, or an open binary file
            holding an RFC 822 message
        :param dpi: passed to each image attachment's ingest; ``None`` detects
            per attachment
        :param include_body: turn the message body into a leading text page.
            Useful -- claim references and policy numbers are often typed in the
            covering mail rather than attached.  That page carries no raster.
        :param max_attachments: ceiling on attachments processed, guarding
            against a mail bomb.  Attachments past the limit are noted in the
            document's warnings rather than silently dropped.
        :returns: a :class:`Document` whose pages span every readable
            attachment, in message order, each tagged with
            ``meta["attachment"]``.  An attachment that cannot be read is
            recorded as a warning and skipped, so one corrupt part does not cost
            the message.
        """
        import email
        from email import policy as _email_policy

        data, uri = _read_source(source)
        msg = email.message_from_bytes(data, policy=_email_policy.default)
        doc = Document(source_uri=uri, meta={
            "format": "eml", "checksum": stable_hash(data), "ingested_at": _utcnow(),
            "subject": str(msg.get("subject", "")), "from": str(msg.get("from", "")),
            "to": str(msg.get("to", "")), "date": str(msg.get("date", "")),
        })

        next_index = 0
        attachments = 0
        body_parts: List[str] = []

        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            filename = part.get_filename() or ""
            disposition = (part.get("content-disposition") or "").lower()
            ctype = part.get_content_type()

            if ctype in ("text/plain", "text/html") and "attachment" not in disposition:
                if include_body:
                    try:
                        body_parts.append(part.get_content())
                    except Exception:
                        pass
                continue

            if attachments >= max_attachments:
                doc.warn("stopped after %d attachments" % max_attachments)
                break
            try:
                payload = part.get_payload(decode=True)
            except Exception as exc:
                doc.warn("undecodable attachment %r: %s" % (filename, exc))
                continue
            # decode=True yields bytes or None; anything else means this part
            # was not an encoded payload at all and cannot be ingested.
            if not isinstance(payload, (bytes, bytearray)) or not payload:
                continue
            attachments += 1
            try:
                sub = ingest(payload, filename=filename, dpi=dpi)
            except (IngestError, MissingDependency) as exc:
                doc.warn("skipped attachment %r: %s" % (filename or ctype, exc))
                continue
            for p in sub.pages:
                p.meta["attachment"] = filename or ctype
                p.meta["attachment_index"] = attachments
                p = _reindex_page(p, next_index)
                next_index += 1
                doc.pages.append(p)
            doc.warnings.extend(w for w in sub.warnings if w not in doc.warnings)

        if include_body and body_parts:
            text = "\n\n".join(t for t in body_parts if t and t.strip())
            if text.strip():
                doc.meta["body_text"] = text
                doc.pages.append(_text_only_page(text, next_index, "email-body"))

        doc.meta["page_count"] = len(doc.pages)
        doc.meta["attachments"] = attachments
        if not doc.pages:
            doc.warn("email contained no readable attachments or body")
        return doc

    @staticmethod
    def page_from_image(img: ImageArray, index: int = 0, dpi: int = DEFAULT_IMAGE_DPI) -> Page:
        """Public helper: build a :class:`Page` from an in-memory array.

        The entry point for pages that never came from a file -- a frame grabbed
        from a scanner SDK, a crop produced upstream, a synthetic test page.

        :param img: a NumPy array, greyscale or RGB.  Converted to ``uint8``.
        :param index: the page's index within its document, which every span's
            ``bbox.page`` refers back to
        :param dpi: the array's true resolution.  Defaults to the
            constant :data:`DEFAULT_IMAGE_DPI`.  Get this right: every
            DPI-relative threshold downstream is computed from it.
        :returns: a :class:`Page` with the raster attached and its point-space
            geometry derived from the array's shape and ``dpi``
        """
        return _page_from_array(ensure_uint8(img), index, dpi)

    @staticmethod
    def document_from_images(images: Sequence[ImageArray], dpi: int = DEFAULT_IMAGE_DPI,
                             source_uri: str = "<arrays>") -> Document:
        """Public helper: build a :class:`Document` from in-memory arrays.

        :param images: the page rasters, in order
        :param dpi: resolution assumed for every array, as described
            in :func:`Ingest.page_from_image`
        :param source_uri: label recorded on the document and used in reports
        :returns: a :class:`Document` with one page per array, indexed from ``0``
        """
        doc = Document(source_uri=source_uri,
                       meta={"format": "arrays", "page_count": len(images),
                             "ingested_at": _utcnow()})
        for i, img in enumerate(images):
            doc.pages.append(_page_from_array(ensure_uint8(img), i, dpi))
        return doc

    @staticmethod
    def ingest(source: Source, filename: str = "", dpi: Optional[int] = None,
               password: Optional[str] = None, max_pages: int = 0,
               page_range: Optional[Sequence[int]] = None, render_dpi: int = DEFAULT_DPI,
               min_native_chars: int = MIN_NATIVE_CHARS) -> Document:
        """Ingest anything: PDF, image, multi-page TIFF, ``.eml``, path or bytes.

        Format is decided by content, not by extension.  Returns a :class:`Document`
        whose pages are geometry-complete but *not* rasterised.

        :param source: a path or path-like (``"scan.pdf"``), raw ``bytes``, or an
            open binary file object.  A directory is not accepted here; walk one
            with :func:`Ingest.ingest_dir` instead.
        :param filename: name hint, used only when ``source`` is bytes or a
            stream and only to break a tie the magic bytes could not.  Also
            becomes the document's ``source_uri``.
        :param dpi: assumed resolution for bare images.  ``None`` reads the
            file's metadata and falls back to :data:`DEFAULT_IMAGE_DPI`.  Ignored
            for PDFs, which use ``render_dpi``.
        :param password: for an encrypted PDF; ignored for other formats
        :param max_pages: stop after this many pages; ``0`` means no limit
        :param page_range: explicit zero-based page indices to keep, e.g.
            ``range(10, 20)``.  PDFs only.
        :param render_dpi: resolution for PDF page rendering.  ``300`` is the
            floor for reliable OCR; ``400`` helps on small print.
        :param min_native_chars: per-page character floor for trusting a PDF text
            layer; see :func:`Ingest.classify_page_kind`.
        :returns: a :class:`Document` whose pages are geometry-complete but not
            yet rasterised -- ``raster()`` is lazy by contract, not as an
            optimisation.
        :raises IngestError: the input was empty, the format was unrecognised, or
            the file could not be opened

        Bytes off a queue, with the name as the only hint::

            doc = ingest(payload, filename="claim-4471.pdf")
        """
        data, uri = _read_source(source)
        if filename and uri in ("<bytes>", "<stream>"):
            uri = filename
        if not data:
            raise IngestError("empty input: %s" % uri)
        kind = sniff_format(data, filename or uri)

        if kind == "pdf":
            doc = ingest_pdf(data, password=password, max_pages=max_pages,
                             page_range=page_range, render_dpi=render_dpi,
                             min_native_chars=min_native_chars)
        elif kind == "tiff":
            doc = ingest_tiff(data, dpi=dpi)
        elif kind in IMAGE_FORMATS:
            doc = ingest_image(data, dpi=dpi)
        elif kind == "eml":
            doc = ingest_email(data, dpi=dpi)
        else:
            # Last resort: some scanners emit headerless JPEG-in-a-wrapper.
            try:
                doc = ingest_image(data, dpi=dpi)
            except Exception:
                raise IngestError("unrecognised format for %s (first bytes: %r)"
                                  % (uri, data[:16]))
        doc.source_uri = uri
        if max_pages and len(doc.pages) > max_pages:
            doc.pages = doc.pages[:max_pages]
        return doc

    @staticmethod
    def ingest_dir(directory: str, pattern: Optional[str] = None, recursive: bool = True,
                   **kwargs: Any) -> List[Document]:
        """Ingest every readable document in a directory tree.

        Unreadable files are reported as warnings on an empty Document rather than
        aborting the batch -- a 5000-file inbox with three corrupt scans should
        still process 4997 of them.

        :param directory: the root to walk.  Dotfiles are skipped; files are
            taken in sorted order per directory, so the result is stable.
        :param pattern: a regular expression matched against each *filename*
            with ``re.search``, e.g. ``"[.]pdf$"`` or ``"^claim-"``.  ``None``
            attempts every file, relying on content sniffing.
        :param recursive: descend into subdirectories.  ``False`` reads only the
            top level.
        :param kwargs: forwarded to :func:`Ingest.ingest` for each file --
            ``render_dpi``, ``max_pages``, ``password``, ...
        :returns: one :class:`Document` per file attempted, in walk order.  A
            file that failed yields an empty document carrying the reason in its
            ``warnings``, so the count always matches the files seen.
        """
        out: List[Document] = []
        regex = re.compile(pattern) if pattern else None
        for root, _dirs, files in os.walk(directory):
            for name in sorted(files):
                if name.startswith("."):
                    continue
                if regex is not None and not regex.search(name):
                    continue
                path = os.path.join(root, name)
                try:
                    out.append(ingest(path, **kwargs))
                except Exception as exc:
                    bad = Document(source_uri=path)
                    bad.warn("ingest failed: %s" % exc)
                    out.append(bad)
            if not recursive:
                break
        return out

    @staticmethod
    def merge_documents(docs: Sequence[Document], source_uri: str = "<merged>") -> Document:
        """Concatenate documents into one, renumbering pages and their bboxes.

        Renumbering is the point: every span's ``bbox.page`` refers to a page
        index, so concatenating without reindexing would leave provenance
        pointing at the wrong pages.

        :param docs: the documents to concatenate, in order
        :param source_uri: label for the merged document
        :returns: one :class:`Document` whose pages are numbered from ``0``, with
            warnings unioned and duplicates dropped
        """
        merged = Document(source_uri=source_uri, meta={"ingested_at": _utcnow(),
                                                       "merged_from": len(docs)})
        idx = 0
        for d in docs:
            for p in d.pages:
                merged.pages.append(_reindex_page(p, idx))
                idx += 1
            for w in d.warnings:
                if w not in merged.warnings:
                    merged.warnings.append(w)
        merged.meta["page_count"] = len(merged.pages)
        return merged


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Ingest from the inside.
# ``Ingest.x`` is the documented path; ``x`` is the one that already works.
sniff_format         = Ingest.sniff_format
classify_page_kind   = Ingest.classify_page_kind
ingest_pdf           = Ingest.ingest_pdf
ingest_image         = Ingest.ingest_image
ingest_tiff          = Ingest.ingest_tiff
ingest_email         = Ingest.ingest_email
page_from_image      = Ingest.page_from_image
document_from_images = Ingest.document_from_images
ingest               = Ingest.ingest
ingest_dir           = Ingest.ingest_dir
merge_documents      = Ingest.merge_documents


def _read_source(source: Source) -> Tuple[bytes, str]:
    """Normalise a path / bytes / file-like into ``(data, uri)``."""
    if isinstance(source, (bytes, bytearray)):
        return (bytes(source), "<bytes>")
    if hasattr(source, "read"):
        stream = cast(IO[bytes], source)
        data = stream.read()
        name = getattr(stream, "name", "<stream>")
        if isinstance(data, str):
            data = data.encode("utf-8")
        return (data, str(name))
    if not isinstance(source, (str, os.PathLike)):
        raise IngestError("cannot read from %s: expected a path, bytes or a "
                          "file-like object" % type(source).__name__)
    path = os.fspath(source)
    if not os.path.exists(path):
        raise IngestError("no such file: %s" % path)
    with open(path, "rb") as fh:
        return (fh.read(), path)


# -- native text layer -------------------------------------------------------

def _pdf_raster_provider(fdoc: Any, pno: int) -> RasterProvider:
    """Closure that renders page ``pno`` at a requested DPI, on demand.

    Holds a reference to the open PyMuPDF document; that keeps the (in-memory)
    file alive exactly as long as some page might still need to render.
    """
    def provider(dpi: int) -> ImageArray:
        """Render this PDF page at ``dpi`` on demand."""
        np = _np()
        fitz: Module = _fitz(required=True)   # required=True never returns None
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = fdoc[pno].get_pixmap(matrix=matrix, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.width, pix.n)
        if pix.n == 1:
            return np.ascontiguousarray(arr[:, :, 0])
        return np.ascontiguousarray(arr[:, :, :3])
    return provider


def _ingest_pdf_pdfium(source: Source, max_pages: int = 0,
                       page_range: Optional[Sequence[int]] = None,
                       render_dpi: int = DEFAULT_DPI) -> Document:
    """Render-only PDF ingest via pypdfium2, when PyMuPDF is unavailable.

    No text-layer extraction, so every page is treated as SCANNED.  That is a
    correctness-preserving degradation -- it costs OCR money on native pages,
    it does not produce wrong text.
    """
    pdfium = _try_import("pypdfium2")
    if pdfium is None:
        raise MissingDependency("pymupdf", "PDF ingest", "pymupdf")
    np = _np()
    data, uri = _read_source(source)
    pdf = pdfium.PdfDocument(data)
    doc = Document(source_uri=uri)
    doc.warn("PyMuPDF unavailable: no native text layer extracted, every page "
             "will be treated as scanned")
    total = len(pdf)
    indices = list(page_range) if page_range is not None else list(range(total))
    indices = [i for i in indices if 0 <= i < total]
    if max_pages:
        indices = indices[:max_pages]
    doc.meta.update({"format": "pdf", "page_count": total, "renderer": "pypdfium2",
                     "checksum": stable_hash(data), "ingested_at": _utcnow()})

    for pno in indices:
        ppage = pdf[pno]
        width, height = ppage.get_size()
        page = Page(index=pno, width_pt=float(width), height_pt=float(height),
                    kind=PageKind.SCANNED)
        page.quality.raster_dpi = render_dpi

        def provider(dpi: int, _pno: int = pno) -> ImageArray:
            """Render this PDF page at ``dpi`` on demand."""
            bitmap = pdf[_pno].render(scale=dpi / 72.0)
            return np.ascontiguousarray(np.asarray(bitmap.to_pil().convert("RGB")))

        page.set_raster_provider(provider, dpi=render_dpi)
        doc.pages.append(page)
    return doc


def _image_dpi_hint(data: bytes) -> Optional[int]:
    """Read declared DPI from image metadata, if it looks plausible."""
    pil = _try_import("PIL.Image")
    if pil is None:
        return None
    try:
        with pil.open(io.BytesIO(data)) as im:
            info = im.info.get("dpi")
        if info:
            value = int(round(float(info[0])))
            # 72 is Pillow's "no idea" default; 1 and 0 show up in broken files.
            if 50 <= value <= 1200 and value != 72:
                return value
    except Exception:
        pass
    return None


def _page_from_array(img: ImageArray, index: int, dpi: int, uri: str = "") -> Page:
    """Wrap a decoded raster as a Page with correct point-space geometry."""
    h, w = image_shape(img)
    page = Page(index=index,
                width_pt=w * 72.0 / float(dpi), height_pt=h * 72.0 / float(dpi),
                kind=PageKind.SCANNED)
    page.set_raster(img, dpi=dpi)
    page.quality.raster_dpi = dpi
    page.meta["source"] = uri
    page.meta["pixel_size"] = [w, h]
    return page


def _reindex_page(page: Page, new_index: int) -> Page:
    """Renumber a page and every bbox on it.  Needed when merging documents."""
    if page.index == new_index:
        return page
    page.spans = [dataclasses.replace(
        s, bbox=BBox(new_index, s.bbox.x0, s.bbox.y0, s.bbox.x1, s.bbox.y1))
        for s in page.spans]
    page.layout.regions = [dataclasses.replace(
        r, bbox=BBox(new_index, r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1))
        for r in page.layout.regions]
    page.index = new_index
    return page


def _text_only_page(text: str, index: int, source: str) -> Page:
    """A synthetic page holding plain text (an email body) with no raster.

    Laid out as monospaced lines so that bboxes are at least ordinally
    meaningful -- a reviewer can be shown "line 12", which beats no provenance.
    """
    page = Page(index=index, kind=PageKind.DIGITAL_NATIVE)
    page.meta["synthetic"] = True
    page.meta["source"] = source
    line_h = 12.0
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        y = 36.0 + i * line_h
        page.spans.append(TextSpan(
            text=line.strip(),
            bbox=BBox(index, 36.0, y, 36.0 + 6.0 * len(line), y + line_h),
            source=source, line=i))
    page.height_pt = max(page.height_pt, 72.0 + line_h * (len(page.spans) + 1))
    page.quality.verdict = Verdict.CLEAN
    return page


# =============================================================================
# 7. Layer 2 -- preprocessing as composable ops
# =============================================================================
#
# Ops are ``Page -> Page``.  They never mutate their input, they record
# themselves in the output page's history, and they are built by small
# factories so that a whole policy is serialisable data rather than code.
#
# The policy is adaptive by default because preprocessing is channel
# equalisation: a fixed sequence over-processes clean pages (destroying signal
# that was never damaged) and under-processes bad ones.  Note in particular
# what ``default_policy`` does *not* do -- unconditional binarisation.  It
# reliably helps classical OCR and reliably hurts vision-language models, which
# use greyscale gradient to disambiguate faint strokes.  That trade-off is
# encoded once here, with an eval set behind it, instead of three times.


# -- op machinery ------------------------------------------------------------

OpFn = Callable[..., Any]


class OpFactory(Protocol):  # pragma: no cover - structural type only
    """What :func:`register_op` returns: a callable that builds an :class:`Op`.

    It carries the two attributes the decorator attaches, which are part of the
    contract rather than implementation detail -- ``fn`` is how one op reuses
    another's raster function without paying to build an :class:`Op` first (see
    :func:`Ops.auto_orient` calling ``rotate.fn``).
    """

    #: The name the op is registered under.
    op_name: str
    #: The undecorated raster function, ``fn(img, page, **params)``.
    fn: OpFn

    def __call__(self, *args: Any, **kwargs: Any) -> Op:
        """Build an :class:`Op` from positional/keyword op parameters."""
        ...


#: Registry of op factories by name, so a serialised policy can be rebuilt.
OP_FACTORIES: Dict[str, OpFactory] = {}


class Op(object):
    """A named, parameterised, serialisable ``Page -> Page`` transform.

    The underlying function has signature ``fn(img, page, **params) -> img``
    and may additionally mutate ``page`` (to update DPI or geometry).  It must
    not mutate ``img`` in place -- ops share rasters with their input page
    until the moment of replacement.
    """

    __slots__ = ("name", "fn", "params", "geometric", "needs_raster")

    def __init__(self, name: str, fn: OpFn, params: Optional[Dict[str, Any]] = None,
                 geometric: bool = False, needs_raster: bool = True) -> None:
        """Bind a raster function and its parameters into a reusable op.

        :param name: the registered name, used for history and serialisation
        :param fn: ``fn(img, page, **params) -> Optional[ImageArray]``; returning
            ``None`` means "declined, leave the raster alone"
        :param params: keyword arguments bound now and passed to ``fn`` at every
            application, e.g. ``{"max_angle": 10.0}``.  Copied, so the caller's
            dict cannot change the op afterwards.
        :param geometric: True when the op changes the page's pixel dimensions, so
            that point-space geometry is recomputed afterwards
        :param needs_raster: when True the op is skipped (and says so in the
            history) on a page that has no raster
        """
        self.name = name
        self.fn = fn
        self.params = dict(params or {})
        self.geometric = geometric
        self.needs_raster = needs_raster

    def __call__(self, page: Page) -> Page:
        """Run the op, returning a *new* page.  The input page is never mutated.

        Timing and parameters land in the returned page's history whether or not
        the op actually changed anything, so a pipeline is reconstructable from
        its output alone.

        :param page: the page to transform; it is copied, never mutated
        :returns: a new :class:`Page`.  Identical in pixels to the input when the
            op declined (returned ``None``) or was skipped for want of a raster --
            but its ``history`` records the attempt either way.
        """
        out = page.copy()
        if self.needs_raster and not out.has_raster():
            out.record(self.name, self.params, 0.0, note="skipped: no raster")
            return out
        with Timer() as t:
            before = out.pixel_size
            img = out.raster()
            result = self.fn(img, out, **self.params)
            if result is not None:
                out.set_raster(result, dpi=out._raster_dpi)
            after = out.pixel_size
        out.record(self.name, self.params, t.ms)
        if self.geometric and before != after:
            _sync_page_geometry(out)
        return out

    def then(self, other: Op) -> Op:
        """``a.then(b)`` -- a composite that runs this op, then ``other``.

        :param other: the op to run second
        :returns: a :class:`CompositeOp` of the two, equivalent to
            ``compose(self, other)`` and chainable further
        """
        return compose(self, other)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict; feed it back to :func:`op_from_dict`."""
        return {"op": self.name, "params": _as_jsonable(self.params)}

    def __repr__(self) -> str:
        """``Op(deskew, max_angle=15.0)``."""
        if not self.params:
            return "Op(%s)" % self.name
        return "Op(%s, %s)" % (self.name, ", ".join(
            "%s=%r" % kv for kv in sorted(self.params.items())))


def _sync_page_geometry(page: Page) -> None:
    """Recompute point-space page size after a geometry-changing op."""
    w, h = page.pixel_size
    if w and h:
        dpi = float(page.raster_dpi)
        page.width_pt = w * 72.0 / dpi
        page.height_pt = h * 72.0 / dpi


def _map_span_points(page: Page,
                     mapper: Callable[[float, float], Tuple[float, float]]) -> None:
    """Apply a point-space coordinate map to every span and region on a page.

    Geometric ops use this so that a native text layer survives deskewing
    instead of being silently invalidated -- provenance is unrecoverable once
    dropped, which is the whole reason it is threaded through the IR.
    """
    def remap(b: BBox) -> BBox:
        """Map a box's four corners, then take their axis-aligned hull."""
        corners = [mapper(b.x0, b.y0), mapper(b.x1, b.y0),
                   mapper(b.x1, b.y1), mapper(b.x0, b.y1)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return BBox(b.page, min(xs), min(ys), max(xs), max(ys))

    page.spans = [dataclasses.replace(s, bbox=remap(s.bbox)) for s in page.spans]
    page.layout.regions = [dataclasses.replace(r, bbox=remap(r.bbox))
                           for r in page.layout.regions]


def register_op(name: str, geometric: bool = False,
                needs_raster: bool = True
                ) -> Callable[[OpFn], OpFactory]:
    """Decorator turning ``fn(img, page, **params)`` into an :class:`Op` factory.

    The resulting factory keeps the decorated function's signature and
    defaults, so ``deskew(max_angle=10)`` reads naturally and ``help(deskew)``
    shows the real documentation.

    This is the extension point: an op registered from your own module works
    everywhere a built-in does, including :func:`op_from_dict` round-trips, with
    no edit to this file.

    :param name: registry key, and the label recorded in ``page.history``.  Must
        be unique -- registering an existing name replaces it, which is how you
        override a built-in, deliberately or by accident.
    :param geometric: ``True`` when the op moves pixels, so span and region
        coordinates must be remapped with it.  Getting this wrong is the classic
        silent provenance bug: the raster rotates, the boxes do not, and every
        bbox points at the wrong place.  See :func:`Ops.deskew` for the mapping
        pattern.
    :param needs_raster: ``True`` (default) skips the op, recording why, on a
        page with no raster.  Set ``False`` only for ops that work on spans or
        metadata alone and so never touch pixels.
    :returns: a decorator that registers the raster function and returns
        its :class:`Op` factory
    """
    def decorator(fn: OpFn) -> OpFactory:
        """Register ``fn`` under ``name`` and return its Op factory."""
        try:
            import inspect
            sig = inspect.signature(fn)
            names = [p.name for p in list(sig.parameters.values())[2:]]
            defaults = dict(
                (p.name, p.default) for p in list(sig.parameters.values())[2:]
                if p.default is not inspect.Parameter.empty)
        except Exception:  # pragma: no cover
            names, defaults = [], {}

        @functools.wraps(fn)
        def factory(*args: Any, **kwargs: Any) -> Op:
            """Build an :class:`Op` from positional/keyword op parameters.

            Unknown or surplus arguments raise :class:`TypeError` here rather than
            surfacing much later as a confusing failure inside the raster function.

            """
            params = dict(defaults)
            for i, value in enumerate(args):
                if i >= len(names):
                    raise TypeError("%s() takes at most %d arguments" % (name, len(names)))
                params[names[i]] = value
            for k, v in kwargs.items():
                if k not in names:
                    raise TypeError("%s() got an unexpected argument %r" % (name, k))
                params[k] = v
            return Op(name, fn, params, geometric=geometric, needs_raster=needs_raster)

        factory.op_name = name  # type: ignore[attr-defined]
        factory.fn = fn  # type: ignore[attr-defined]  # raw function, for reuse
        # The two attributes above are what make a plain function an OpFactory;
        # they are attached after the def, so the shape is asserted here.
        built = cast(OpFactory, factory)
        OP_FACTORIES[name] = built
        return built
    return decorator


def compose(*ops: Optional[Op]) -> CompositeOp:
    """Chain ops into one.  ``None`` entries are dropped, so conditional
    policies can be written without branching noise.

    Returns the :class:`CompositeOp` rather than a bare :class:`Op`, so that the
    ``.ops`` it was built from stay reachable for inspection and serialisation.
    """
    return CompositeOp([o for o in ops if o is not None])


class CompositeOp(Op):
    """An ordered sequence of ops applied as one."""

    __slots__ = ("ops",)

    def __init__(self, ops: Sequence[Op]) -> None:
        """Wrap an ordered sequence of ops as a single op.

        :param ops: the member ops, applied in this order.  Kept on ``.ops`` so
            they stay reachable for inspection and serialisation.
        """
        Op.__init__(self, "compose", lambda img, page: img,
                    {"ops": [o.name for o in ops]}, needs_raster=False)
        self.ops = list(ops)

    def __call__(self, page: Page) -> Page:
        """Run every op in order, threading the page through.

        :param page: the starting page; it is not mutated
        :returns: the page after every member op, carrying one ``history`` entry
            per op rather than a single entry for the composite
        """
        out = page
        for op in self.ops:
            out = op(out)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict containing each member op."""
        return {"op": "compose", "ops": [o.to_dict() for o in self.ops]}

    def __repr__(self) -> str:
        """``compose(Op(deskew), Op(binarize))``."""
        return "compose(%s)" % ", ".join(repr(o) for o in self.ops)


def op_from_dict(d: Mapping[str, Any]) -> Op:
    """Rebuild an op (or a composite) from its serialised form.

    The inverse of :meth:`Op.to_dict`, which is what lets a policy be recorded in
    an eval report and replayed later against a new build.

    :param d: ``{"op": name, "params": {...}}``, or ``{"op": "compose", "ops":
        [...]}`` for a composite.  Rebuilt recursively.
    :returns: an :class:`Op`, or a :class:`CompositeOp` for a ``"compose"`` entry
    :raises ConfigError: the name is not in :data:`OP_FACTORIES`; the message
        lists what is registered, since the usual cause is a custom op whose
        module has not been imported yet
    """
    name = d.get("op")
    if name == "compose":
        return CompositeOp([op_from_dict(o) for o in d.get("ops", [])])
    factory = OP_FACTORIES.get(str(name))
    if factory is None:
        raise ConfigError("unknown op %r (known: %s)"
                          % (name, ", ".join(sorted(OP_FACTORIES))))
    return factory(**dict(d.get("params") or {}))


def apply_ops(page: Page, ops: Sequence[Op]) -> Page:
    """Apply ops in order, returning a new page.

    :param page: the starting page; it is not mutated
    :param ops: ops to run in sequence, as built by the :class:`Ops` factories.
        An empty sequence returns the page unchanged.
    :returns: the page after every op, with one ``history`` entry per op
    """
    out = page
    for op in ops:
        out = op(out)
    return out


# -- colour / tone -----------------------------------------------------------

class Ops(object):
    """The preprocessing ops, as :class:`Op` factories.

    Calling one builds an op; applying it returns a new page.  Each
    corrects a *measured* degradation and returns ``None`` when there is
    nothing to correct, so a clean page passes through untouched -- a fixed
    sequence applied to every page destroys signal that was never damaged.

    Registering a new op does not require editing this class:
    :func:`register_op` adds to :data:`OP_FACTORIES` from anywhere.

    **Reading these signatures.**  Each is shown as ``fn(img, page, **knobs)``,
    which is how the raster function is written -- but :func:`register_op` strips
    the first two, so what you call is the knobs alone.  ``img`` and ``page`` are
    supplied by the machinery when the op runs::

        op = deskew(min_angle=0.5)   # build: only the knobs
        page = op(page)              # apply: img and page threaded in for you

    Every ``:param:`` documented below is therefore a keyword you may pass at
    build time.  Ops apply lazily in this sense: nothing is rasterised until the
    op runs, and an op that returns ``None`` leaves the page's pixels untouched
    while still recording the decision in ``page.history``.
    """

    @staticmethod
    @register_op("to_grayscale")
    def to_grayscale(img: ImageArray, page: Page) -> Optional[ImageArray]:
        """Collapse to a single channel.

        Do this before classical OCR (which discards colour anyway) but *not*
        before colour-based stamp removal, which needs hue.
        """
        return to_gray(img)

    @staticmethod
    @register_op("invert_if_dark")
    def invert_if_dark(img: ImageArray, page: Page,
                       ink_threshold: float = 0.55) -> Optional[ImageArray]:
        """Invert white-on-black pages (negative scans, some fax modes).

        Every downstream measurement assumes dark ink on light paper; a negative
        page reports 90% ink coverage and gets classified unreadable, when in fact
        it just needs one subtraction.

        :param ink_threshold: inked fraction above which the page is judged to be
            a negative, in ``0.0..1.0``.  The default ``0.55`` sits well above any
            real document -- dense Devanagari body text reaches about ``0.25``, a
            solid-black form header about ``0.4`` -- so this fires on true
            negatives and not on merely heavy pages.  Lower it towards ``0.45``
            only if you have genuinely ink-saturated scans; below that you will
            start inverting healthy pages.
        """
        np = _np()
        gray = to_gray(img)
        if float(ink_mask(gray).mean()) <= ink_threshold:
            return None
        page.meta["inverted"] = True
        return (255 - np.asarray(img)).astype(np.uint8)

    @staticmethod
    @register_op("autocontrast")
    def autocontrast(img: ImageArray, page: Page, low_pct: float = 1.0,
                     high_pct: float = 99.0) -> Optional[ImageArray]:
        """Stretch the ``[low_pct, high_pct]`` intensity range to full scale.

        Percentile-based so that a punch hole or a black scan border does not eat
        the entire dynamic range, which is what naive min/max stretching does.

        :param low_pct: percentile mapped to black, in ``0.0..100.0``.  ``1.0``
            discards the darkest 1% of pixels -- enough to absorb punch holes,
            staple shadows and a thin scan border.  ``0.0`` is exact min/max
            stretching and is what you want only on synthetic images.
        :param high_pct: percentile mapped to white, in ``0.0..100.0``.  ``99.0``
            pairs with the default ``low_pct``; widen to ``0.5``/``99.5`` for a
            gentler stretch on pages that are already close to full range.
        :returns: ``None`` when the two percentiles are within one grey level of
            each other -- a blank or already-saturated page, where stretching
            would only amplify noise.
        """
        np = _np()
        gray = to_gray(img)
        lo = _percentile_np(gray, low_pct)
        hi = _percentile_np(gray, high_pct)
        if hi - lo < 1.0:
            return None
        arr = np.asarray(img, dtype=np.float64)
        out = (arr - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    @register_op("gamma")
    def gamma(img: ImageArray, page: Page, value: float = 1.0) -> Optional[ImageArray]:
        """Apply gamma correction.  ``value < 1`` brightens, ``> 1`` darkens.

        Non-linear, unlike :func:`Ops.autocontrast`: it moves the midtones while
        leaving true black and true white fixed, which is what recovers faint
        pencil or carbon-copy strokes without blowing out the paper.

        :param value: gamma exponent, practically ``0.4..2.5``.  ``0.6`` lifts
            faint grey strokes on an underexposed phone photo; ``1.4`` deepens
            washed-out thermal-printer output; ``1.0`` is identity and returns
            ``None`` so a no-op costs nothing.
        :returns: ``None`` when ``value`` is 1.0 within tolerance
        """
        np = _np()
        if abs(value - 1.0) < 1e-6:
            return None
        table = np.clip(((np.arange(256) / 255.0) ** float(value)) * 255.0, 0, 255)
        return table.astype(np.uint8)[np.asarray(img)]

    @staticmethod
    @register_op("clahe")
    def clahe(img: ImageArray, page: Page, clip: float = 2.0,
              grid: int = 8) -> Optional[ImageArray]:
        """Contrast-limited adaptive histogram equalisation.

        The right tool for a page that is faded in one corner and fine elsewhere --
        a global stretch cannot fix that by construction.  Falls back to
        :func:`Ops.autocontrast` without OpenCV.

        :param clip: contrast ceiling per tile.  ``2.0`` is the usual document
            setting; ``1.0`` is nearly a no-op, and above ``4.0`` the noise in
            blank paper is amplified into visible grain that OCR reads as specks.
        :param grid: tile count along each axis, so ``8`` means an 8x8 grid.
            Fewer, larger tiles (``4``) approach a global stretch; more, smaller
            tiles (``16``) track tighter lighting gradients but start equalising
            *within* a glyph, which thins strokes.
        :returns: the equalised greyscale page; never ``None``, since the caller
            has already decided the page needs it
        """
        cv = _cv2()
        gray = to_gray(img)
        if cv is None:
            page.meta["clahe"] = "fell back to global autocontrast (no OpenCV)"
            return _autocontrast_array(gray)
        op = cv.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
        return op.apply(gray)

    @staticmethod
    @register_op("normalize_illumination")
    def normalize_illumination(img: ImageArray, page: Page, radius: Optional[int] = None,
                               strength: float = 1.0) -> Optional[ImageArray]:
        """Flatten uneven lighting by dividing out the estimated background.

        This is the fix for phone photos and book scans with a shadow gradient.
        Doing it *before* binarisation is what lets a global threshold work at all;
        skipping it is why so many pipelines lose the shadowed third of a page.

        :param radius: background-estimation window in pixels; ``None`` derives one
            from the page size, which is right unless your text is unusually large.
            The window must be comfortably wider than a glyph -- set it near a
            character width and the estimator treats the text itself as background
            and erases it.
        :param strength: how far to move towards the flattened result, in
            ``0.0..1.0``.  ``1.0`` applies it fully, whereas the gentler
            vlm policy uses ``0.7`` -- vision models read the greyscale
            gradient, and a fully flattened page discards the very cue that
            distinguishes a faint stroke from paper.  See the
            gentler :func:`Policies.vlm_policy`.
        """
        np = _np()
        gray = to_gray(img)
        background = estimate_background(gray, radius).astype(np.float64)
        background = np.maximum(background, 1.0)
        normalised = np.clip(gray.astype(np.float64) * 255.0 / background, 0, 255)
        if strength < 1.0:
            normalised = gray.astype(np.float64) * (1.0 - strength) + normalised * strength
        return normalised.astype(np.uint8)

    @staticmethod
    @register_op("remove_shadow")
    def remove_shadow(img: ImageArray, page: Page, radius: int = 21) -> Optional[ImageArray]:
        """Remove cast shadows via dilate-then-median background subtraction.

        The classic recipe: dilating with a mid-sized kernel removes text, the
        median then removes residual structure, and the difference is the page
        without its shadow.

        Overlaps with :func:`Ops.normalize_illumination`; prefer that one for a
        smooth lighting gradient and this one for a hard-edged cast shadow, such
        as the photographer's hand or the spine shadow of an open book.

        :param radius: dilation and median kernel in pixels, forced odd.  It must
            exceed the thickest stroke on the page or the text survives dilation
            and is subtracted away with the shadow.  ``21`` suits 300 DPI body
            text; raise towards ``41`` at 600 DPI or for large print.
        """
        np = _np()
        gray = to_gray(img)
        k = int(radius) | 1
        cv = _cv2()
        if cv is not None:
            dilated = cv.dilate(gray, np.ones((k, k), np.uint8))
            background = cv.medianBlur(dilated, k)
        else:
            background = median_blur(box_blur(gray, k // 2), 5)
        diff = 255 - np.abs(gray.astype(np.int16) - background.astype(np.int16))
        return _autocontrast_array(np.clip(diff, 0, 255).astype(np.uint8))

    @staticmethod
    @register_op("rescale", geometric=True)
    def rescale(img: ImageArray, page: Page, factor: float = 1.0,
                interpolation: str = "auto") -> Optional[ImageArray]:
        """Scale the raster by ``factor``, keeping physical page size constant.

        Point-space coordinates are unaffected because DPI scales with the pixels;
        that invariant is what keeps every previously-extracted bbox valid.

        :param factor: linear scale.  ``2.0`` doubles each side and quadruples the
            pixel count; ``0.5`` halves it.  ``1.0`` returns ``None``.
        :param interpolation: ``"auto"`` picks by direction -- cubic when
            enlarging, area when shrinking, which is the pairing that avoids
            both softness and aliasing.  Override with ``"nearest"`` (preserves
            hard edges on already-binarised pages), ``"linear"``, ``"cubic"``,
            or ``"area"``.
        :returns: ``None`` when ``factor`` is 1.0 within tolerance
        """
        if abs(factor - 1.0) < 1e-6:
            return None
        out = resize_image(img, scale=factor, interpolation=interpolation)
        page._raster_dpi = int(round(page.raster_dpi * factor))
        page.quality.raster_dpi = page._raster_dpi
        return out

    @staticmethod
    @register_op("ensure_dpi", geometric=True)
    def ensure_dpi(img: ImageArray, page: Page, min_dpi: int = 300, max_dpi: int = 600,
                   interpolation: str = "auto") -> Optional[ImageArray]:
        """Upsample until the page reaches ``min_dpi``, capped at ``max_dpi``.

        Upsampling adds no information -- but Tesseract's and PaddleOCR's character
        models are trained around 300 DPI glyph sizes and measurably lose accuracy
        on smaller input, so the resample buys real recall even though it buys no
        new signal.  The cap exists because beyond ~400 DPI you pay quadratically
        in pixels for nothing.

        Measures against ``page.quality.effective_dpi`` -- the resolution the text
        *actually* carries -- not the nominal raster DPI, so a 600 DPI upscan of a
        150 DPI fax is correctly seen as still needing help.

        :param min_dpi: floor to reach.  ``300`` is where Tesseract and PaddleOCR
            are trained; ``400`` measurably helps on small print and Indic scripts.
        :param max_dpi: ceiling that overrides ``min_dpi`` when honouring it would
            produce an unreasonably large raster.  ``600`` is already past the
            point of accuracy returns.
        :param interpolation: as :func:`Ops.rescale`; ``"auto"`` resolves to cubic
            here, since this op only ever enlarges.
        :returns: ``None`` when the page is already at or above ``min_dpi``, or
            when the cap leaves no room to scale
        """
        current = page.quality.effective_dpi or page.raster_dpi or DEFAULT_IMAGE_DPI
        if current >= min_dpi:
            return None
        factor = float(min_dpi) / float(current)
        if page.raster_dpi * factor > max_dpi:
            factor = float(max_dpi) / float(page.raster_dpi or min_dpi)
        if factor <= 1.0 + 1e-6:
            return None
        out = resize_image(img, scale=factor, interpolation=interpolation)
        page._raster_dpi = int(round(page.raster_dpi * factor))
        page.quality.raster_dpi = page._raster_dpi
        page.quality.effective_dpi = int(round(current * factor))
        return out

    @staticmethod
    @register_op("resize_max_side", geometric=True)
    def resize_max_side(img: ImageArray, page: Page, max_px: int = 2000,
                        interpolation: str = "area") -> Optional[ImageArray]:
        """Cap the longest side.  Vision-model cost is ~linear in pixels, and most
        providers downscale above ~1500px anyway -- paying to upload pixels the
        provider will discard is pure waste.

        The counterpart to :func:`Ops.ensure_dpi`: that one raises resolution for
        classical OCR, this one caps it for a VLM.  Running both is normal --
        they clamp opposite ends.

        :param max_px: ceiling for the longer side, in pixels.  ``2000`` is what
            the vlm policy uses, comfortably above every major provider's
            internal downscale; ``1500`` matches it more tightly and costs less;
            below ~1000 small print stops being legible to the model.
        :param interpolation: ``"area"`` by default, which is the correct filter
            for downscaling -- it averages the discarded pixels instead of point
            sampling, so thin strokes fade rather than disappear.
        :returns: ``None`` when the page is already within ``max_px``
        """
        h, w = image_shape(img)
        longest = max(h, w)
        if longest <= max_px:
            return None
        factor = float(max_px) / float(longest)
        out = resize_image(img, scale=factor, interpolation=interpolation)
        page._raster_dpi = max(1, int(round(page.raster_dpi * factor)))
        page.quality.raster_dpi = page._raster_dpi
        return out

    @staticmethod
    @register_op("deskew", geometric=True)
    def deskew(img: ImageArray, page: Page, max_angle: float = 15.0, min_angle: float = 0.4,
               method: str = "auto", angle: Optional[float] = None) -> Optional[ImageArray]:
        """Rotate the page so its text lines are horizontal.

        Rotation is skipped below ``min_angle`` because resampling always costs a
        little sharpness, and correcting 0.2 degrees costs more than it recovers.
        Above ``max_angle`` the page is not skewed, it is rotated -- a different
        problem, handled by :func:`Ops.auto_orient`.

        Existing span coordinates are rotated with the image rather than discarded.

        :param max_angle: refuse to correct beyond this many degrees.  A larger
            reading almost always means the detector locked onto a table rule or
            a page edge rather than the text baselines, and acting on it would
            wreck a page that was fine.
        :param min_angle: skip below this many degrees, since resampling costs
            sharpness that 0.2 degrees of skew does not.  ``0.4`` is the default;
            drop to ``0.2`` only if downstream is a classical OCR engine, which
            is far more skew-sensitive than a VLM.
        :param method: skew estimator -- ``"auto"`` (projection profile, then Hough
            if OpenCV is present), ``"projection"``, or ``"hough"``.  Projection is
            more robust on dense text; Hough does better on sparse forms with
            strong rules.
        :param angle: bypass detection and rotate by exactly this many degrees.
            Use when you already know the skew, e.g. from a calibration target or
            a previous page of the same batch.  Still subject to ``min_angle`` and
            ``max_angle``.
        :returns: ``None`` when the measured skew falls outside
            ``[min_angle, max_angle]`` -- the page is left untouched
        """
        detected = float(angle) if angle is not None else page.quality.skew_deg
        if angle is None and abs(detected) < 1e-9:
            detected = estimate_skew(to_gray(img), max_angle=max_angle, method=method)
        if abs(detected) < min_angle or abs(detected) > max_angle:
            return None

        before_h, before_w = image_shape(img)
        out = rotate_image(img, detected, expand=True)
        after_h, after_w = image_shape(out)
        dpi = float(page.raster_dpi)

        # Same affine mapping the raster underwent, expressed in points.
        theta = math.radians(-detected)  # image rotated CCW by `detected`
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx, cy = before_w / 2.0 * 72.0 / dpi, before_h / 2.0 * 72.0 / dpi
        nx, ny = after_w / 2.0 * 72.0 / dpi, after_h / 2.0 * 72.0 / dpi

        def mapper(x: float, y: float) -> Tuple[float, float]:
            """Point-space image of ``(x, y)`` under the same rotation the raster took."""
            dx, dy = x - cx, y - cy
            return (cos_t * dx - sin_t * dy + nx, sin_t * dx + cos_t * dy + ny)

        _map_span_points(page, mapper)
        page.quality.skew_deg = 0.0
        page.rotation = int(page.rotation)
        page.meta["deskew_applied_deg"] = round(detected, 3)
        return out

    @staticmethod
    @register_op("rotate", geometric=True)
    def rotate(img: ImageArray, page: Page, degrees: int = 0) -> Optional[ImageArray]:
        """Rotate by an exact multiple of 90 degrees (lossless).

        Quarter-turns resample nothing, so unlike :func:`Ops.deskew` this costs no
        sharpness and needs no threshold.  Span coordinates turn with the page.

        :param degrees: clockwise rotation, rounded to the nearest multiple of 90
            and taken modulo 360.  ``90``, ``180``, ``270`` are the useful values;
            ``-90`` is accepted and means the same as ``270``.  ``0`` returns
            ``None``.
        :returns: ``None`` when the rotation is a whole number of full turns
        """
        np = _np()
        turns = int(round(degrees / 90.0)) % 4
        if turns == 0:
            return None
        page.rotation = (page.rotation + turns * 90) % 360
        h, w = image_shape(img)
        dpi = float(page.raster_dpi)

        def mapper(x: float, y: float, _t: int = turns, _w: float = w * 72.0 / dpi,
                   _h: float = h * 72.0 / dpi) -> Tuple[float, float]:
            """Point-space image of ``(x, y)`` after ``_t`` clockwise quarter-turns."""
            # One clockwise quarter-turn sends (x, y) -> (H - y, x) and swaps the
            # page's width and height; apply that `turns` times.
            for _ in range(_t):
                x, y = _h - y, x
                _w, _h = _h, _w
            return (x, y)

        _map_span_points(page, mapper)
        return np.ascontiguousarray(np.rot90(np.asarray(img), k=-turns))

    @staticmethod
    @register_op("auto_orient", geometric=True)
    def auto_orient(img: ImageArray, page: Page, use_osd: bool = True,
                    min_confidence: float = 1.0) -> Optional[ImageArray]:
        """Detect and correct 90/180/270-degree page rotation.

        Tesseract's orientation-and-script-detection is used when available because
        it is the only cheap method that can tell upside-down text from right-way-up
        text.  Without it we fall back to a projection-profile test, which reliably
        distinguishes portrait from landscape text but **cannot** detect a 180-degree
        flip -- that limitation is reported in the page history rather than hidden.

        :param use_osd: allow the Tesseract OSD path when the binary or
            ``pytesseract`` is importable.  Set ``False`` to force the projection
            fallback -- worth doing when OSD is present but unreliable, which it
            is on sparse forms and on scripts it was not trained for.
        :param min_confidence: least OSD confidence to act on.  Tesseract reports
            roughly ``0.5..5.0`` here, and its low-confidence guesses on sparse
            pages are close to coin flips; ``1.0`` rejects those.  Raise to
            ``2.0`` if you would rather leave a page unrotated than risk a wrong
            quarter-turn.
        :returns: ``None`` when the page is already upright, or when the page has
            too little ink to judge
        """
        np = _np()
        rotation = None
        if use_osd and (have("pytesseract") or shutil.which("tesseract")):
            rotation = _osd_rotation(img, min_confidence)
        if rotation is None:
            gray = to_gray(img)
            small = resize_image(gray, scale=min(1.0, 800.0 / max(image_shape(gray))))
            mask = ink_mask(small).astype(np.float64)
            if mask.mean() < 0.001:
                return None
            horizontal = float(np.var(mask.sum(axis=1)))
            vertical = float(np.var(mask.sum(axis=0)))
            rotation = 90 if vertical > horizontal * 1.6 else 0
            page.meta["orientation_method"] = "projection (180-degree flips undetectable)"
        else:
            page.meta["orientation_method"] = "tesseract-osd"
        if not rotation:
            return None
        # OSD reports the counter-clockwise angle that would right the page.
        return rotate.fn(img, page, degrees=-rotation)

    @staticmethod
    @register_op("crop_to_content", geometric=True)
    def crop_to_content(img: ImageArray, page: Page, margin_pt: float = 6.0,
                        min_keep: float = 0.25) -> Optional[ImageArray]:
        """Crop away empty margins, keeping ``margin_pt`` of whitespace.

        Guarded by ``min_keep``: if the detected content occupies less than that
        fraction of the page, the detection is more likely to have locked onto a
        speck of dust than onto the content, and the crop is refused.

        :param margin_pt: whitespace to keep around the content, in typographic
            points (1/72 inch), so it means the same thing at any DPI.  ``6.0``
            is roughly a line of leading.  Do not drop to ``0`` -- OCR layout
            analysis degrades when glyphs touch the edge, which is precisely
            what :func:`Ops.pad` exists to prevent.
        :param min_keep: least fraction of the original area the crop may keep,
            in ``0.0..1.0``.  At ``0.25`` a crop discarding more than three
            quarters of the page is refused and the reason recorded in
            ``page.meta["crop_refused"]``.  Lower it only for documents that
            genuinely are a receipt on a big scan bed.
        :returns: ``None`` when the page is blank, when the crop would be a no-op,
            or when ``min_keep`` refuses it
        """
        np = _np()
        gray = to_gray(img)
        h, w = image_shape(gray)
        mask = ink_mask(gray)
        # Suppress isolated specks before deciding where the content ends.
        mask = morph_binary(mask, 3, 3, "open")
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return None
        dpi = float(page.raster_dpi)
        margin_px = int(round(margin_pt * dpi / 72.0))
        y0 = max(0, int(rows[0]) - margin_px); y1 = min(h, int(rows[-1]) + 1 + margin_px)
        x0 = max(0, int(cols[0]) - margin_px); x1 = min(w, int(cols[-1]) + 1 + margin_px)
        if (y1 - y0) * (x1 - x0) < min_keep * h * w:
            page.meta["crop_refused"] = "content region below min_keep"
            return None
        if (y0, x0, y1, x1) == (0, 0, h, w):
            return None
        dx = -x0 * 72.0 / dpi
        dy = -y0 * 72.0 / dpi
        _map_span_points(page, lambda x, y: (x + dx, y + dy))
        return np.ascontiguousarray(np.asarray(img)[y0:y1, x0:x1])

    @staticmethod
    @register_op("remove_border", geometric=True)
    def remove_border(img: ImageArray, page: Page, max_frac: float = 0.08,
                      dark_threshold: int = 90) -> Optional[ImageArray]:
        """Trim the black frame a flatbed leaves when the lid is open.

        Those frames wreck contrast measurement and ink coverage, and Tesseract
        frequently reads them as a column of punctuation.

        Trims only inward from the four edges, so unlike :func:`Ops.crop_to_content`
        it cannot cut into the page when content reaches the margin.

        :param max_frac: most of each dimension the trim may eat, in ``0.0..1.0``.
            ``0.08`` allows an 8% frame, which covers the usual lid-open border
            while making it impossible to consume a dark but legitimate header.
        :param dark_threshold: mean grey level, ``0..255``, below which a row or
            column counts as frame.  ``90`` clears true scanner black (near ``0``)
            with margin to spare; raise towards ``120`` for a grey plastic lid,
            but not so far that a dense text row qualifies.
        :returns: ``None`` when no edge rows or columns are dark enough to trim
        """
        np = _np()
        gray = to_gray(img)
        h, w = image_shape(gray)
        limit_y = max(1, int(h * max_frac))
        limit_x = max(1, int(w * max_frac))
        dark_rows = gray.mean(axis=1) < dark_threshold
        dark_cols = gray.mean(axis=0) < dark_threshold

        top = 0
        while top < limit_y and dark_rows[top]:
            top += 1
        bottom = h
        while bottom > h - limit_y and dark_rows[bottom - 1]:
            bottom -= 1
        left = 0
        while left < limit_x and dark_cols[left]:
            left += 1
        right = w
        while right > w - limit_x and dark_cols[right - 1]:
            right -= 1
        if (top, left, bottom, right) == (0, 0, h, w):
            return None
        dpi = float(page.raster_dpi)
        _map_span_points(page, lambda x, y: (x - left * 72.0 / dpi, y - top * 72.0 / dpi))
        return np.ascontiguousarray(np.asarray(img)[top:bottom, left:right])

    @staticmethod
    @register_op("pad", geometric=True)
    def pad(img: ImageArray, page: Page, margin_px: int = 16,
            value: int = 255) -> Optional[ImageArray]:
        """Add a quiet margin.  Tesseract's layout analysis degrades noticeably
        when glyphs touch the image edge.

        The usual last step of an OCR policy, after cropping and binarisation have
        both had the chance to leave text flush against the border.

        :param margin_px: border width in pixels, applied to all four sides.
            ``16`` is enough for Tesseract at 300 DPI; scale it with your DPI if
            you render higher.  ``0`` or negative returns ``None``.
        :param value: fill level, ``0..255``.  ``255`` (white) is right for a
            normal page; use ``0`` after :func:`Ops.invert_if_dark` has left you
            with light ink on dark paper, so the margin matches the paper rather
            than framing it.
        :returns: ``None`` when ``margin_px`` is not positive
        """
        np = _np()
        m = int(margin_px)
        if m <= 0:
            return None
        arr = np.asarray(img)
        width = ((m, m), (m, m)) + (((0, 0),) if arr.ndim == 3 else ())
        dpi = float(page.raster_dpi)
        _map_span_points(page, lambda x, y: (x + m * 72.0 / dpi, y + m * 72.0 / dpi))
        return np.pad(arr, width, mode="constant", constant_values=int(value))

    @staticmethod
    @register_op("perspective_correct", geometric=True)
    def perspective_correct(img: ImageArray, page: Page, min_area_ratio: float = 0.35,
                            epsilon_frac: float = 0.02) -> Optional[ImageArray]:
        """Flatten a photographed page by warping its detected quadrilateral.

        Only fires when a convincing four-sided contour covering most of the frame
        is found, because warping on a bad quad is far worse than not warping:
        it shears the text irrecoverably.  Requires OpenCV.

        :param min_area_ratio: least fraction of the frame the detected quad must
            cover, in ``0.0..1.0``.  ``0.35`` rejects the small quadrilaterals that
            table borders and photo frames produce.  Raise towards ``0.6`` when
            every photo is known to be page-filling; lowering it is how you get
            a page warped to the shape of a table.
        :param epsilon_frac: contour simplification tolerance as a fraction of the
            perimeter.  ``0.02`` is the standard value that collapses a slightly
            wavy page outline to four corners; too small and the contour keeps
            more than four vertices and is rejected, too large and unrelated
            shapes collapse into plausible-looking quads.
        :returns: ``None`` when OpenCV is missing, or when no quad passes both
            guards -- an unwarped page beats a sheared one
        """
        np = _np()
        cv = _cv2()
        if cv is None:
            page.meta["perspective_correct"] = "skipped: OpenCV unavailable"
            return None
        gray = to_gray(img)
        h, w = image_shape(gray)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 60, 180)
        edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
        best = None
        best_area = min_area_ratio * h * w
        for contour in contours:
            area = cv.contourArea(contour)
            if area < best_area:
                continue
            approx = cv.approxPolyDP(contour, epsilon_frac * cv.arcLength(contour, True), True)
            if len(approx) == 4 and cv.isContourConvex(approx):
                best, best_area = approx.reshape(4, 2).astype(np.float32), area
        if best is None:
            return None

        # Order corners tl, tr, br, bl.
        s = best.sum(axis=1)
        d = np.diff(best, axis=1).ravel()
        quad = np.array([best[np.argmin(s)], best[np.argmin(d)],
                         best[np.argmax(s)], best[np.argmax(d)]], dtype=np.float32)
        widths = [np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[3] - quad[2])]
        heights = [np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])]
        tw = int(max(widths)); th = int(max(heights))
        if tw < 32 or th < 32:
            return None
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        matrix = cv.getPerspectiveTransform(quad, dst)
        page.meta["perspective_corrected"] = True
        page.spans = []  # a projective warp cannot be applied to axis-aligned bboxes
        return cv.warpPerspective(np.asarray(img), matrix, (tw, th),
                                  flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    @staticmethod
    @register_op("denoise")
    def denoise(img: ImageArray, page: Page, strength: str = "light",
                method: str = "auto") -> Optional[ImageArray]:
        """Suppress scanner noise.

        ``method="auto"`` picks bilateral filtering, which smooths flat paper while
        preserving stroke edges -- ordinary Gaussian blur removes noise and glyph
        detail in equal measure, which is why "denoise then OCR" so often scores
        *worse* than doing nothing.

        :param strength: one of ``"light"``, ``"medium"``, ``"aggressive"``.
            ``"light"`` is the only safe default: each step up also erodes
            diacritics, Devanagari matras and thin CJK strokes, so
            ``"aggressive"`` is for genuinely filthy fax output and nothing else.
        :param method: ``"auto"`` (bilateral with OpenCV, median without),
            ``"bilateral"``, ``"nlmeans"``, ``"median"``, or ``"gaussian"``.
            ``"nlmeans"`` is the highest quality and by far the slowest;
            ``"gaussian"`` is the one to avoid, since it blurs strokes and noise
            alike.  ``"bilateral"`` and ``"nlmeans"`` fall through to a
            non-OpenCV path when OpenCV is missing.
        :raises ConfigError: ``strength`` or ``method`` is not one of the above --
            a typo here silently disabling denoising would be worse
        """
        np = _np()
        levels = {"light": 1, "medium": 2, "aggressive": 3}
        if strength not in levels:
            raise ConfigError("strength must be one of %s" % sorted(levels))
        level = levels[strength]
        cv = _cv2()
        gray = to_gray(img) if not is_color(img) else np.asarray(img)

        if method in ("auto", "bilateral") and cv is not None:
            diameter = 5 + 2 * level
            return cv.bilateralFilter(gray, diameter, 25 * level, 25 * level)
        if method == "nlmeans" and cv is not None:
            if is_color(gray):
                return cv.fastNlMeansDenoisingColored(gray, None, 3 * level, 3 * level, 7, 21)
            return cv.fastNlMeansDenoising(gray, None, 3 * level, 7, 21)
        if method in ("median", "auto"):
            return median_blur(gray, 3 if level < 3 else 5)
        if method == "gaussian":
            return gaussian_blur(gray, 0.5 * level)
        raise ConfigError("unknown denoise method %r" % method)

    @staticmethod
    @register_op("despeckle")
    def despeckle(img: ImageArray, page: Page, min_area_px: int = 6) -> Optional[ImageArray]:
        """Remove ink blobs smaller than ``min_area_px``.

        Dust, fax speckle and JPEG mosquito noise become spurious punctuation
        otherwise -- which then poisons amount parsing (``1.234`` vs ``1,234``).

        :param min_area_px: connected ink components smaller than this many pixels
            are painted back to the local paper colour.  ``6`` is tuned for 300
            DPI, where a full stop covers roughly 12-20 pixels -- so the default
            leaves real punctuation alone.  Scale it with the square of your DPI
            (about ``24`` at 600 DPI); leave it low if the script carries small
            marks, since Devanagari matras and Arabic dots are legitimately tiny.
        :returns: ``None`` when the page has no ink, or nothing is small enough
            to remove
        """
        np = _np()
        gray = to_gray(img)
        mask = ink_mask(gray)
        if not mask.any():
            return None
        cv = _cv2()
        if cv is not None:
            count, labels, stats, _ = cv.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8)
            small = np.zeros(count, dtype=bool)
            for i in range(1, count):
                if stats[i, cv.CC_STAT_AREA] < min_area_px:
                    small[i] = True
            remove = small[labels]
        else:
            # Density proxy: ink whose 5x5 neighbourhood holds too little ink.
            sums, _ = _rect_box_sum(mask.astype(np.float64), 5, 5)
            remove = mask & (sums < float(min_area_px))
        if not remove.any():
            return None
        out = np.array(gray, copy=True)
        background = int(np.percentile(gray[~mask], 50)) if (~mask).any() else 255
        out[remove] = background
        return out

    @staticmethod
    @register_op("unsharp")
    def unsharp(img: ImageArray, page: Page, amount: float = 1.0, radius: float = 2.0,
                threshold: int = 0) -> Optional[ImageArray]:
        """Unsharp masking: ``img + amount * (img - blur(img))``.

        The one operation that genuinely recovers readability on soft scans.
        ``threshold`` suppresses sharpening of low-contrast areas so that paper
        grain is not amplified along with the strokes.

        :param amount: how much of the high-frequency difference to add back.
            ``1.0`` is a normal correction, and ``1.2`` is what the default
            policy applies to a page measured as blurred.  Past about ``2.0``
            strokes gain white halos that OCR segments as extra characters --
            oversharpening reads worse than the soft original.
        :param radius: Gaussian radius in pixels defining "detail".  ``2.0`` suits
            300 DPI body text; raise it for large print, lower it towards ``1.0``
            for dense small type, where a wide radius sharpens the gaps between
            glyphs rather than the glyphs.
        :param threshold: least absolute difference, ``0..255``, that gets
            sharpened at all.  ``0`` sharpens everything including paper grain;
            ``5``-``10`` leaves flat paper alone and is worth setting on any noisy
            scan.
        """
        np = _np()
        base = to_gray(img) if not is_color(img) else np.asarray(img)
        blurred = gaussian_blur(base, radius)
        diff = base.astype(np.float64) - blurred.astype(np.float64)
        if threshold:
            diff = np.where(np.abs(diff) < float(threshold), 0.0, diff)
        return np.clip(base.astype(np.float64) + amount * diff, 0, 255).astype(np.uint8)

    @staticmethod
    @register_op("morphology")
    def morphology(img: ImageArray, page: Page, operation: str = "open", ksize: int = 3,
                   iterations: int = 1) -> Optional[ImageArray]:
        """Grayscale morphology.  ``close`` fills broken strokes in faded thermal
        print; ``open`` thins bleed-through from the reverse side.

        Named from the point of view of the *ink*, not the pixel values.  Because
        ink is dark, an OCR-sense "dilate the text" is a greyscale erosion, and
        this op flips the operation for you -- so ``"dilate"`` thickens strokes,
        as the name suggests.

        :param operation: one of ``"open"``, ``"close"``, ``"erode"``,
            ``"dilate"``, ``"tophat"``, ``"blackhat"``.  Use ``"close"`` for
            broken strokes, ``"open"`` for speckle and show-through,
            ``"dilate"`` to thicken hairline print before OCR, and ``"tophat"``
            to pull small bright detail off an uneven background.  Without
            OpenCV only the first four are meaningful, as the fallback runs on a
            binary ink mask.
        :param ksize: structuring element side in pixels, forced odd.  ``3`` is a
            one-pixel nudge; ``5`` and ``7`` act fast, and anything larger tends
            to merge adjacent glyphs into blobs.
        :param iterations: how many times to apply it.  Two passes of ``3`` are
            gentler and more controllable than one pass of ``5``.
        :raises ConfigError: ``operation`` is not one of the six listed (OpenCV
            path only)
        """
        np = _np()
        gray = to_gray(img)
        cv = _cv2()
        k = max(1, int(ksize)) | 1
        if cv is None:
            mask = ink_mask(gray)
            for _ in range(max(1, int(iterations))):
                mask = morph_binary(mask, k, k, operation)
            out = np.full_like(gray, 255)
            out[mask] = 0
            return out
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
        ops = {"open": cv.MORPH_OPEN, "close": cv.MORPH_CLOSE,
               "erode": cv.MORPH_ERODE, "dilate": cv.MORPH_DILATE,
               "tophat": cv.MORPH_TOPHAT, "blackhat": cv.MORPH_BLACKHAT}
        if operation not in ops:
            raise ConfigError("unknown morphological operation %r" % operation)
        # Ink is dark, so an OCR-sense "dilate the text" is a grayscale erosion.
        flipped = {"erode": cv.MORPH_DILATE, "dilate": cv.MORPH_ERODE,
                   "open": cv.MORPH_CLOSE, "close": cv.MORPH_OPEN}
        code = flipped.get(operation, ops[operation])
        return cv.morphologyEx(gray, code, kernel, iterations=max(1, int(iterations)))

    @staticmethod
    def binarize_array(gray: GrayImage, method: str = "sauvola", window: int = 31,
                       k: Optional[float] = None, offset: float = 10,
                       min_std: float = 8.0) -> GrayImage:
        """Binarise a grayscale array to a 0/255 image.

        ``otsu``
            One global threshold from between-class variance.  Fast and excellent
            on evenly-lit print; catastrophic under a shadow gradient.
        ``adaptive``
            Local mean minus ``offset``.  Cheap, robust, slightly noisy.
        ``sauvola`` *(default)*
            ``T = m * (1 + k * (s / R - 1))``.  Designed for degraded documents:
            it lowers the threshold where local contrast is low, so faint strokes
            survive without dragging the background in with them.
        ``niblack``
            ``T = m + k * s`` (``k`` negative).  Sauvola's predecessor; keeps more
            faint text, at the cost of needing the ``min_std`` guard below.
        ``wolf``
            Niblack normalised by the global minimum and maximum local deviation.
            Better than Sauvola when contrast varies wildly across the page.
        ``nick``
            ``T = m + k * sqrt(var + m^2)``.  Tuned for very low-contrast scans --
            the faded thermal-printer case.
        ``bradley``
            Integral-image mean with a percentage offset.  Very fast, and the most
            forgiving of a badly chosen window.

        ``window`` should be a little larger than one text line's height.  Too
        small and glyph interiors get their own threshold (hollow letters); too
        large and it degenerates to Otsu.

        ``min_std`` guards Niblack, and only Niblack.  In a uniform patch of paper
        the local deviation is ~0, so Niblack's threshold collapses onto the local
        mean and half the blank page comes out black -- its notorious failure mode.
        Where the local deviation is below ``min_std`` there is no local evidence
        to use, so the global Otsu decision is taken instead.  The other local
        methods need no such guard: Sauvola's ``(s/R - 1)`` term, Wolf's
        normalisation and NICK's ``sqrt(var + m^2)`` all drive the threshold well
        below the mean when deviation vanishes.

        :param gray: the page as a greyscale array; colour input is converted
        :param method: one of ``"otsu"``, ``"adaptive"``, ``"sauvola"``,
            ``"niblack"``, ``"wolf"``, ``"nick"``, ``"bradley"``, each described
            above.  ``"sauvola"`` is the default because it degrades most
            gracefully across the widest range of real scans.
        :param window: local window side in pixels, forced odd, minimum ``3``.
            ``31`` suits 300 DPI body text; use ``51``-``61`` at 600 DPI.  Ignored
            by ``"otsu"``, which is global.
        :param k: method-specific sensitivity; ``None`` selects the published
            default for the chosen method -- ``0.2`` for Sauvola, ``-0.2`` for
            Niblack, ``0.5`` for Wolf, ``-0.14`` for NICK, ``0.15`` for Bradley.
            Raising Sauvola's ``k`` keeps less faint ink; lowering it keeps more
            background.  Ignored by ``"otsu"`` and ``"adaptive"``.
        :param offset: grey levels subtracted from the local mean.  Used only by
            ``"adaptive"``, where ``10`` is a mild bias towards keeping ink.
        :param min_std: local deviation below which ``"niblack"`` falls back to
            the global Otsu threshold.  Guards Niblack, and only Niblack, against
            turning blank paper black.
        :returns: an array of the same shape holding only ``0`` and ``255``
        :raises ConfigError: ``method`` is not one of the seven listed
        """
        np = _np()
        g = to_gray(gray).astype(np.float64)
        win = max(3, int(window) | 1)

        if method == "otsu":
            binary = g > otsu_threshold(gray)
        elif method == "adaptive":
            mean, _ = window_stats(g, win)
            binary = g > (mean - float(offset))
        elif method == "bradley":
            mean, _ = window_stats(g, win)
            binary = g > mean * (1.0 - (0.15 if k is None else float(k)))
        elif method == "sauvola":
            kk = 0.2 if k is None else float(k)
            mean, std = window_stats(g, win)
            binary = g > mean * (1.0 + kk * (std / 128.0 - 1.0))
        elif method == "niblack":
            kk = -0.2 if k is None else float(k)
            mean, std = window_stats(g, win)
            binary = g > (mean + kk * std)
            flat = std < float(min_std)
            if flat.any():
                binary = np.where(flat, g > otsu_threshold(gray), binary)
        elif method == "wolf":
            kk = 0.5 if k is None else float(k)
            mean, std = window_stats(g, win)
            max_std = float(std.max()) or 1.0
            min_gray = float(g.min())
            binary = g > (mean - kk * (1.0 - std / max_std) * (mean - min_gray))
        elif method == "nick":
            kk = -0.14 if k is None else float(k)
            mean, std = window_stats(g, win)
            binary = g > (mean + kk * np.sqrt(std ** 2 + mean ** 2))
        else:
            raise ConfigError("unknown binarisation method %r" % method)
        return (binary.astype(np.uint8)) * 255

    @staticmethod
    @register_op("binarize")
    def binarize(img: ImageArray, page: Page, method: str = "sauvola", window: int = 31,
                 k: Optional[float] = None, offset: int = 10,
                 min_std: float = 8.0) -> Optional[ImageArray]:
        """Binarise the page.

        Deliberately *not* part of :func:`Policies.default_policy`.  Binarisation reliably
        lifts classical OCR accuracy and reliably degrades vision-language model
        accuracy, because VLMs use greyscale gradient to disambiguate faint strokes
        that a threshold has already destroyed.  Apply it in an OCR-bound branch
        only -- see :func:`Policies.ocr_policy` versus :func:`Policies.vlm_policy`.

        The op wrapper around :func:`Ops.binarize_array`; every parameter has the
        same meaning and defaults there, where each method is described in full.

        :param method: ``"sauvola"`` (default), ``"otsu"``, ``"adaptive"``,
            ``"niblack"``, ``"wolf"``, ``"nick"``, or ``"bradley"``
        :param window: local window side in pixels, forced odd; ``31`` at 300 DPI
        :param k: method sensitivity; ``None`` takes each method's published default
        :param offset: grey levels below the local mean, ``"adaptive"`` only
        :param min_std: flat-region guard for ``"niblack"`` only
        :raises ConfigError: ``method`` is not one of the seven listed
        """
        return binarize_array(img, method=method, window=window, k=k, offset=offset,
                              min_std=min_std)

    @staticmethod
    @register_op("remove_lines")
    def remove_lines(img: ImageArray, page: Page, direction: str = "both",
                     min_len_ratio: float = 0.4, thickness: int = 2,
                     keep_text: bool = True) -> Optional[ImageArray]:
        """Erase long ruled lines (table borders, form rules, underlines).

        Ruled lines merge with glyphs during OCR line segmentation and turn
        ``|1,234|`` into ``11,2341``.  Removing them costs nothing on prose pages
        and buys a lot on the ruled tables that dominate hospital bills.
        ``keep_text`` restores pixels where a line crossed a glyph, so struck-through
        or underlined text is not punched full of holes.

        :param direction: ``"both"``, ``"horizontal"``, or ``"vertical"``.  Use
            ``"horizontal"`` alone on statements ruled only between rows, which
            avoids mistaking a tall bracket or a devanagari shirorekha stem for a
            vertical rule.
        :param min_len_ratio: least run length for a line to count, as a fraction
            of the page dimension, in ``0.0..1.0``.  ``0.4`` means a horizontal
            rule must span 40% of the width.  Lower it towards ``0.25`` for
            narrow-column tables; raising it protects underlines from removal.
        :param thickness: dilation in pixels applied to the detected run, so that
            the anti-aliased edges of a rule are erased along with its core.
            ``2`` suits 300 DPI; ``3``-``4`` for heavier print or higher DPI.
        :param keep_text: restore pixels that also look like glyph, so a rule
            crossing a digit does not punch a hole in it.  Leave this ``True``
            unless you are removing rules from a page with no text on them.
        :returns: ``None`` when the page has no ink, or no run is long enough to
            count as a rule
        :raises ConfigError: ``direction`` is not one of the three listed
        """
        np = _np()
        if direction not in ("both", "horizontal", "vertical"):
            raise ConfigError("direction must be both/horizontal/vertical")
        gray = to_gray(img)
        h, w = image_shape(gray)
        mask = ink_mask(gray)
        if not mask.any():
            return None

        removal = np.zeros_like(mask)
        if direction in ("both", "horizontal"):
            length = max(10, int(w * min_len_ratio))
            opened = morph_binary(mask, 1, length, "open")
            removal |= morph_binary(opened, max(1, int(thickness)), 3, "dilate")
        if direction in ("both", "vertical"):
            length = max(10, int(h * min_len_ratio))
            opened = morph_binary(mask, length, 1, "open")
            removal |= morph_binary(opened, 3, max(1, int(thickness)), "dilate")
        removal &= mask
        if not removal.any():
            return None
        if keep_text:
            # A pixel that is part of a thick vertical run is glyph, not rule.
            text_like = morph_binary(mask & ~removal, 3, 3, "dilate")
            removal &= ~text_like
        out = np.array(gray, copy=True)
        background = int(np.percentile(gray[~mask], 50)) if (~mask).any() else 255
        out[removal] = background
        page.meta["lines_removed_px"] = int(removal.sum())
        return out

    @staticmethod
    @register_op("remove_stamps")
    def remove_stamps(img: ImageArray, page: Page, saturation_min: int = 70,
                      value_min: int = 40, coverage_max: float = 0.25) -> Optional[ImageArray]:
        """Suppress coloured stamps, seals and signatures, keeping black print.

        Purple "PAID" stamps and blue ink signatures land on top of the numbers
        that matter, and OCR reads the overlap as garbage.  Detection is by
        saturation: black toner is unsaturated by definition, so anything strongly
        coloured is an overlay.  ``coverage_max`` aborts the removal on genuinely
        colourful documents (a colour brochure) where the premise does not hold.
        Requires a colour raster; on grayscale input this is a no-op.

        Order matters: run this *before* :func:`Ops.to_grayscale`, which throws
        away the hue this op detects with.

        :param saturation_min: least HSV saturation, ``0..255``, for a pixel to
            count as coloured.  ``70`` clears black and grey toner, which sit near
            ``0``, while catching blue and purple ink.  Lower it towards ``50``
            for faded stamps, at the risk of eating colour-tinted paper.
        :param value_min: least HSV value (brightness), ``0..255``.  ``40``
            excludes near-black pixels whose hue is meaningless noise -- without
            it, dark toner gets classified by whatever hue the sensor guessed.
        :param coverage_max: abort if more than this fraction of the page is
            coloured, in ``0.0..1.0``.  At ``0.25`` a colour brochure or a
            coloured-paper form is left alone, because there the premise "colour
            means overlay" is simply false.  The reason lands in
            ``page.meta["remove_stamps"]``.
        :returns: ``None`` on greyscale input, when nothing is coloured, or when
            ``coverage_max`` aborts the removal
        """
        np = _np()
        if not is_color(img):
            page.meta["remove_stamps"] = "skipped: grayscale input"
            return None
        cv = _cv2()
        rgb = to_rgb(img)
        if cv is not None:
            hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
            sat = hsv[:, :, 1].astype(np.float64)
            val = hsv[:, :, 2].astype(np.float64)
        else:
            arr = rgb.astype(np.float64)
            mx = arr.max(axis=2)
            mn = arr.min(axis=2)
            val = mx
            sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0) * 255.0, 0.0)
        coloured = (sat >= float(saturation_min)) & (val >= float(value_min))
        coverage = float(coloured.mean())
        if coverage > coverage_max:
            page.meta["remove_stamps"] = "skipped: %.0f%% of page is coloured" % (coverage * 100)
            return None
        if coverage <= 0.0:
            return None
        coloured = morph_binary(coloured, 3, 3, "close")
        gray = to_gray(rgb)
        out = np.array(gray, copy=True)
        ink = ink_mask(gray)
        background = int(np.percentile(gray[~ink], 50)) if (~ink).any() else 255
        out[coloured] = background
        page.meta["stamp_pixels_removed"] = int(coloured.sum())
        return out

    @staticmethod
    def split_multi_bill_page(page: Page, axis: str = "auto", min_gap_frac: float = 0.06,
                              max_parts: int = 3, min_part_frac: float = 0.2) -> List[Page]:
        """Split a page carrying several documents into one page per document.

        Two bills photocopied side by side onto one A4 is routine, and every
        downstream assumption (one header, one total) breaks on it.  Splitting is
        by wide whitespace gutters in the ink projection profile.

        Returns ``[page]`` unchanged when no confident split is found -- a false
        split is much more damaging than a missed one, so the thresholds are
        deliberately conservative.

        Not an :class:`Op`: ops map one page to one page, and this changes the
        page count.  Reach for it through ``Pipeline(split_pages=True)``, or apply
        it document-wide with :func:`Ops.split_document`.

        :param page: a page with a raster; a page without one is returned as-is
        :param axis: ``"auto"``, ``"vertical"``, or ``"horizontal"``.  ``"auto"``
            prefers whichever axis yields more gutters, which handles both
            side-by-side and stacked layouts.  Name the axis when you know it --
            forcing ``"vertical"`` on side-by-side bills stops a wide gap between
            table sections being read as a horizontal split.
        :param min_gap_frac: least whitespace-gutter width for a split, as a
            fraction of the page dimension, in ``0.0..1.0``.  ``0.06`` is about
            half an inch on A4.  Raise it if inter-column spacing is being read
            as a document boundary.
        :param max_parts: refuse the split entirely if it would produce more than
            this many pages.  ``3`` reflects what fits on a sheet; a result of
            eight parts means the detector found text columns, not documents.
        :param min_part_frac: least fraction of the page each part must occupy,
            in ``0.0..1.0``.  At ``0.2`` a cut near the edge -- a margin note, a
            punch-hole strip -- cannot become its own document.
        :returns: one page per detected document, each with ``split_from`` and
            ``split_part`` in ``page.meta`` and spans remapped into its own
            coordinate space; or ``[page]`` when no confident split was found
        :raises ConfigError: ``axis`` is not one of the three listed
        """
        np = _np()
        if not page.has_raster():
            return [page]
        img = page.raster()
        gray = to_gray(img)
        h, w = image_shape(gray)
        mask = ink_mask(gray)
        if mask.mean() < 0.005:
            return [page]

        def gutters(profile: FloatArray, length: int, min_gap: int) -> List[int]:
            """Midpoints of runs of near-empty rows/columns at least ``min_gap`` long.

            :param profile: the ink projection profile along the splitting axis
            :param length: the page's extent along that axis, in pixels
            """
            empty = profile < (profile.max() * 0.02)
            cuts = []
            start = None
            for i, is_empty in enumerate(empty):
                if is_empty and start is None:
                    start = i
                elif not is_empty and start is not None:
                    if i - start >= min_gap and start > length * min_part_frac \
                            and i < length * (1 - min_part_frac):
                        cuts.append((start + i) // 2)
                    start = None
            return cuts

        col_profile = mask.sum(axis=0).astype(np.float64)
        row_profile = mask.sum(axis=1).astype(np.float64)
        vertical_cuts = gutters(col_profile, w, int(w * min_gap_frac))
        horizontal_cuts = gutters(row_profile, h, int(h * min_gap_frac))

        if axis == "auto":
            use_vertical = len(vertical_cuts) >= len(horizontal_cuts) and vertical_cuts
        elif axis == "vertical":
            use_vertical = bool(vertical_cuts)
        elif axis == "horizontal":
            use_vertical = False
        else:
            raise ConfigError("axis must be auto/vertical/horizontal")

        cuts = vertical_cuts if use_vertical else horizontal_cuts
        if not cuts or len(cuts) + 1 > max_parts:
            return [page]

        bounds = [0] + sorted(cuts) + [w if use_vertical else h]
        parts: List[Page] = []
        dpi = float(page.raster_dpi)
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            crop = np.asarray(img)[:, lo:hi] if use_vertical else np.asarray(img)[lo:hi, :]
            part = page.copy()
            part.set_raster(np.ascontiguousarray(crop), dpi=page.raster_dpi)
            part.meta = dict(page.meta)
            part.meta["split_from"] = page.index
            part.meta["split_part"] = i
            dx = -lo * 72.0 / dpi if use_vertical else 0.0
            dy = 0.0 if use_vertical else -lo * 72.0 / dpi
            _map_span_points(part, lambda x, y, _dx=dx, _dy=dy: (x + _dx, y + _dy))
            _sync_page_geometry(part)
            part.record("split_multi_bill_page",
                        {"axis": "vertical" if use_vertical else "horizontal", "part": i})
            parts.append(part)
        return parts

    @staticmethod
    def split_document(doc: Document, **kwargs: Any) -> Document:
        """Apply :func:`Ops.split_multi_bill_page` across a document, renumbering pages.

        Renumbering matters: every span's ``bbox.page`` and every provenance
        reference is keyed by page index, so the new pages are reindexed
        contiguously rather than inheriting the index they were split from.

        :param doc: the document to split; it is not mutated
        :param kwargs: forwarded per page to :func:`Ops.split_multi_bill_page` --
            ``axis``, ``min_gap_frac``, ``max_parts``, ``min_part_frac``
        :returns: a new :class:`Document` whose pages are numbered from ``0``;
            pages that did not split carry through unchanged
        """
        out = Document(source_uri=doc.source_uri, meta=dict(doc.meta),
                       warnings=list(doc.warnings))
        index = 0
        for page in doc.pages:
            for part in split_multi_bill_page(page, **kwargs):
                out.pages.append(_reindex_page(part, index))
                index += 1
        out.meta["page_count"] = len(out.pages)
        return out


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Ops from the inside.
# ``Ops.x`` is the documented path; ``x`` is the one that already works.
to_grayscale           = Ops.to_grayscale
invert_if_dark         = Ops.invert_if_dark
autocontrast           = Ops.autocontrast
gamma                  = Ops.gamma
clahe                  = Ops.clahe
normalize_illumination = Ops.normalize_illumination
remove_shadow          = Ops.remove_shadow
rescale                = Ops.rescale
ensure_dpi             = Ops.ensure_dpi
resize_max_side        = Ops.resize_max_side
deskew                 = Ops.deskew
rotate                 = Ops.rotate
auto_orient            = Ops.auto_orient
crop_to_content        = Ops.crop_to_content
remove_border          = Ops.remove_border
pad                    = Ops.pad
perspective_correct    = Ops.perspective_correct
denoise                = Ops.denoise
despeckle              = Ops.despeckle
unsharp                = Ops.unsharp
morphology             = Ops.morphology
binarize_array         = Ops.binarize_array
binarize               = Ops.binarize
remove_lines           = Ops.remove_lines
remove_stamps          = Ops.remove_stamps
split_multi_bill_page  = Ops.split_multi_bill_page
split_document         = Ops.split_document


def _autocontrast_array(gray: GrayImage, low_pct: float = 1.0,
                        high_pct: float = 99.0) -> GrayImage:
    """Stretch ``gray`` so ``low_pct``..``high_pct`` spans the full 0-255 range.

    Percentiles rather than min/max, so one dust speck and one blown-out
    highlight cannot define the range for the whole page.
    """
    np = _np()
    lo = _percentile_np(gray, low_pct)
    hi = _percentile_np(gray, high_pct)
    if hi - lo < 1.0:
        return gray
    return np.clip((gray.astype(np.float64) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


# -- geometry ----------------------------------------------------------------

def _osd_rotation(img: ImageArray, min_confidence: float = 1.0) -> Optional[int]:
    """Ask Tesseract which way is up.  Returns clockwise degrees, or ``None``."""
    pt = _try_import("pytesseract")
    if pt is None:
        return None
    pil = _try_import("PIL.Image")
    if pil is None:
        return None
    try:
        osd = pt.image_to_osd(pil.fromarray(to_rgb(img)), output_type=pt.Output.DICT)
        if float(osd.get("orientation_conf", 0.0)) < min_confidence:
            return None
        return int(osd.get("rotate", 0)) % 360
    except Exception as exc:
        logger.debug("OSD failed: %s", exc)
        return None


# -- noise and sharpness -----------------------------------------------------

# -- binarisation ------------------------------------------------------------

# -- page splitting ----------------------------------------------------------

# -- policies ----------------------------------------------------------------

PolicyFn = Callable[[Page], List[Op]]


class Policies(object):
    """Policies -- functions from a measured page to the ops it needs.

    A policy is the adaptive half of preprocessing: it reads
    :class:`PageQuality` and reaches only for the ops that page warrants.
    Note what :func:`Policies.default_policy` deliberately omits -- unconditional
    binarisation, which reliably helps classical OCR and reliably hurts
    vision-language models that use greyscale gradient.
    """

    @staticmethod
    def default_policy(page: Page, thresholds: Optional[QualityThresholds] = None) -> List[Op]:
        """Correct the distortions this page actually has, and nothing else.

        Reads only from ``page.quality``, so it is pure, testable, and cheap to
        audit.  Every branch here corresponds to a measured degradation; there is
        no unconditional step, because a clean 400 DPI native render needs nothing
        done to it and every op applied to it can only remove signal.

        Reader-agnostic by design, which is why it omits binarisation -- that
        helps classical OCR and hurts VLMs, so it belongs in
        :func:`Policies.ocr_policy`, not here.

        :param page: a page whose quality :func:`Quality.measure_page` has
            already filled in.  On an unmeasured page every reading is zero and
            this returns an empty list.
        :param thresholds: the cut-offs each branch compares against; ``None``
            uses :data:`DEFAULT_THRESHOLDS`.  Pass a
            modified :class:`QualityThresholds` to retune the whole policy
            without rewriting it -- e.g. ``dataclasses.replace(
            DEFAULT_THRESHOLDS, min_dpi=400)`` to upsample more eagerly.
        :returns: the ops this page warrants, in application order; empty for a
            page that needs nothing
        """
        th = thresholds or DEFAULT_THRESHOLDS
        q = page.quality
        ops: List[Op] = []

        if q.ink_coverage > th.ink_saturated:
            ops.append(invert_if_dark())
        if q.illumination < th.illumination_floor:
            ops.append(normalize_illumination())
        if q.noise > th.noise_ceiling:
            ops.append(denoise("light"))
        if q.effective_dpi and q.effective_dpi < th.min_dpi:
            ops.append(ensure_dpi(th.min_dpi))
        if abs(q.skew_deg) > th.skew_correct_deg:
            ops.append(deskew(min_angle=th.skew_correct_deg))
        if q.blur < th.blur_floor:
            ops.append(unsharp(amount=1.2))
        if q.contrast < th.contrast_floor:
            ops.append(autocontrast())
        return ops

    @staticmethod
    def ocr_policy(page: Page, thresholds: Optional[QualityThresholds] = None,
                   binarization: str = "sauvola") -> List[Op]:
        """Policy for classical OCR engines (Tesseract, Paddle, docTR).

        Everything :func:`Policies.default_policy` does, plus the steps that only make sense
        when a threshold-based recogniser is downstream: line removal, ruled-form
        cleanup, and binarisation.

        Line removal is conditional on the page looking tabular, but greyscale
        conversion, binarisation, despeckling and padding are unconditional --
        every classical engine wants them, so there is no measurement to gate on.

        :param page: a page whose quality has been measured
        :param thresholds: as :func:`Policies.default_policy`; ``None``
            uses :data:`DEFAULT_THRESHOLDS`
        :param binarization: method passed to :func:`Ops.binarize` -- ``"sauvola"``
            (default), ``"otsu"``, ``"adaptive"``, ``"niblack"``, ``"wolf"``,
            ``"nick"``, or ``"bradley"``.  Try ``"otsu"`` on evenly-lit laser
            print, where it is faster and marginally cleaner; stay on
            ``"sauvola"`` for anything photographed or faded.
        :returns: the ops for this page, ending with the binarise-despeckle-pad
            sequence that classical OCR expects
        """
        ops = default_policy(page, thresholds)
        ops.append(to_grayscale())
        if page.layout.is_tabular or page.meta.get("ruled"):
            ops.append(remove_lines())
        ops.append(binarize(method=binarization))
        ops.append(despeckle())
        ops.append(pad(margin_px=16))
        return ops

    @staticmethod
    def vlm_policy(page: Page, thresholds: Optional[QualityThresholds] = None,
                   max_side: int = 2000) -> List[Op]:
        """Policy for vision-language models.

        Gentler on purpose.  No binarisation (it destroys the greyscale gradient
        the model reads faint strokes from), no aggressive denoising (it removes
        diacritics and thin Devanagari strokes), and a hard pixel cap because cost
        scales with image size while accuracy stops improving well before the
        provider's limit.

        Its skew tolerance is also twice :func:`Policies.default_policy`'s, since
        vision models read moderately tilted text without help and the resampling
        would cost more sharpness than the tilt costs accuracy.

        :param page: a page whose quality has been measured
        :param thresholds: as :func:`Policies.default_policy`; ``None``
            uses :data:`DEFAULT_THRESHOLDS`.  Note the skew branch here
            compares against *twice* ``skew_correct_deg``.
        :param max_side: pixel cap for the longer side, handed
            to :func:`Ops.resize_max_side`.  ``2000`` is comfortably above every
            major provider's internal downscale; ``1500`` cuts cost with no
            measurable accuracy loss on ordinary print.
        :returns: a short op list -- often empty for a clean page, which is the
            intended behaviour
        """
        th = thresholds or DEFAULT_THRESHOLDS
        q = page.quality
        ops: List[Op] = []
        if q.ink_coverage > th.ink_saturated:
            ops.append(invert_if_dark())
        if q.illumination < th.illumination_floor:
            ops.append(normalize_illumination(strength=0.7))
        if abs(q.skew_deg) > th.skew_correct_deg * 2:
            ops.append(deskew(min_angle=th.skew_correct_deg * 2))
        if q.blur < th.blur_floor:
            ops.append(unsharp(amount=0.8))
        if q.contrast < th.contrast_floor:
            ops.append(autocontrast())
        ops.append(resize_max_side(max_side))
        return ops

    @staticmethod
    def no_policy(page: Page) -> List[Op]:
        """Apply nothing.  Useful as an eval baseline.

        Distinct from ``policy=None`` in intent only -- both skip preprocessing.
        Use this one when you want a *named* policy in the eval report, so the
        baseline row reads ``no_policy`` rather than ``None``.

        :param page: ignored; the signature exists to satisfy ``PolicyFn``
        :returns: always an empty list
        """
        return []

    @staticmethod
    def fixed_policy(*ops: Op) -> PolicyFn:
        """Turn a fixed op list into a policy, for A/B against adaptive ones.

        The honest way to test the library's central claim.  Pit a fixed sequence
        against :func:`Policies.default_policy` in an :class:`EvalSuite` and see
        whether adapting to measured quality actually wins on your documents::

            fixed = fixed_policy(deskew(), binarize(), despeckle())

        :param ops: the ops to apply to every page, in order.  A fresh copy of
            the list is returned per page, so callers cannot mutate it.
        :returns: a ``PolicyFn`` ignoring page quality entirely
        """
        def policy(page: Page) -> List[Op]:
            """Return the same op list for every page."""
            return list(ops)
        return policy


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Policies from the inside.
# ``Policies.x`` is the documented path; ``x`` is the one that already works.
default_policy = Policies.default_policy
ocr_policy     = Policies.ocr_policy
vlm_policy     = Policies.vlm_policy
no_policy      = Policies.no_policy
fixed_policy   = Policies.fixed_policy


def preprocess(doc: Document, policy: Optional[PolicyFn] = default_policy,
               ops: Optional[Sequence[Op]] = None, dpi: Optional[int] = None,
               max_workers: int = 0, measure: bool = True,
               thresholds: Optional[QualityThresholds] = None,
               in_place: bool = False) -> Document:
    """Measure and correct every page in a document.

    Layer 2 in one call: measure each page, ask the policy what that page needs,
    apply it, then re-measure so downstream routing and the confidence prior see
    the page as it will actually be read.

    :param doc: an ingested document.  Rasters are pulled lazily per page, so a
        300-page bundle is not materialised all at once.
    :param policy: called per page to choose ops.  Defaults
        to :func:`default_policy`; also :func:`Policies.ocr_policy`, or
        else :func:`Policies.vlm_policy`, or any ``Callable[[Page], List[Op]]``.
        Ignored when ``ops`` is given, and ``None`` means apply nothing.
    :param ops: a fixed op list applied to every page, bypassing the policy
        entirely.  For debugging and ablations -- ``ops=[deskew(), binarize()]``
        -- not for production, where a fixed sequence destroys signal on the
        pages that did not need it.
    :param dpi: override the DPI assumed when measuring, for images whose
        metadata lies or is missing.  ``None`` trusts ``page.raster_dpi``.
    :param max_workers: threads for per-page work.  ``0`` runs serially, which is
        what you want while debugging, since op errors then surface in order.
        The ops are IO- and NumPy-bound, so threads do help despite the GIL.
    :param measure: measure quality before choosing ops.  Required for any
        adaptive policy -- with ``False`` every reading stays zero, and
        so :func:`default_policy` sees a perfect page and picks nothing.  Set
        it ``False`` only alongside an explicit ``ops`` list.
    :param thresholds: cut-offs for measurement and policy; ``None``
        uses :data:`DEFAULT_THRESHOLDS`.
    :param in_place: mutate ``doc`` rather than working on a copy.  The copy is
        the safe default; ``True`` saves memory on large bundles, at the cost of
        being unable to compare against the original.
    :returns: the processed :class:`Document`.  Pages carrying no raster -- an
        email body, a native-text page that never rendered -- pass through
        untouched, and every op applied is recorded in ``page.history``.

    Measure the no-preprocessing baseline, then the adaptive one::

        base = preprocess(doc, policy=None)
        tuned = preprocess(doc, policy=Policies.ocr_policy, max_workers=4)
    """
    target = doc if in_place else doc.copy()

    def handle(page: Page) -> Page:
        """Measure, choose ops, apply them, then re-measure one page."""
        if not page.has_raster():
            return page
        if measure:
            measure_page(page, dpi=dpi, thresholds=thresholds)
        chosen = list(ops) if ops is not None else (policy(page) if policy else [])
        if not chosen:
            return page
        result = apply_ops(page, chosen)
        if measure:
            # Re-measure: downstream routing and the confidence prior must see
            # the page as it will actually be read, not as it arrived.
            measure_page(result, thresholds=thresholds)
        return result

    processed = _map_maybe_parallel(handle, target.pages, max_workers, "preprocess")
    target.pages = list(processed)
    return target


# =============================================================================
# 8. Layer 3 -- pluggable reading backends
# =============================================================================
#
# Every backend consumes a Page and produces TextSpans.  That is the whole
# contract, and it is what makes them interchangeable: a router can pick one
# per page, an ensemble can run two and compare, and the eval harness can A/B
# entire pipelines because the types line up.
#
# Backends are registered by name, so a consuming project can add its own
# without a pull request to this file -- if anyone ever has to fork the library
# to get their engine in, that is a defect here, not there.


@dataclass
class ReadResult(object):
    """What a backend returns: spans, what they cost, and how long they took."""

    spans: List[TextSpan] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    latency_ms: float = 0.0
    backend: str = ""
    warnings: List[str] = field(default_factory=list)
    raw: Optional[Any] = field(default=None, repr=False)

    @property
    def text(self) -> str:
        """All non-empty spans joined by spaces, in the order returned."""
        return " ".join(s.text for s in self.spans if s.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict.  ``raw`` is deliberately excluded (unbounded size)."""
        return {"backend": self.backend, "spans": [s.to_dict() for s in self.spans],
                "cost": self.cost.to_dict(), "latency_ms": round(self.latency_ms, 1),
                "warnings": list(self.warnings)}


class TextBackend(Protocol):  # pragma: no cover - structural type only
    """Structural type for anything that can read text off a page."""

    name: str

    def supports(self, page: Page) -> bool:
        """Whether this backend can read ``page`` at all.

        Asked per page, so the answer may legitimately differ across a document:
        a text-layer backend supports the digital pages of a mixed PDF and not
        the scanned ones.

        :param page: the page in question
        :returns: ``False`` when this backend needs a raster the page lacks, when
            its dependencies are missing, or when the page is otherwise outside
            what it handles
        """
        ...

    def is_available(self) -> bool:
        """Whether this backend's dependencies are importable right now.

        Part of the protocol, not just of :class:`BaseBackend`: the routers call
        it to skip a backend whose engine is not installed, so a custom backend
        that omits it breaks routing rather than merely failing to read.
        """
        ...

    def estimate_cost(self, page: Page) -> Cost:
        """Predicted cost of reading ``page``, before doing it.

        Consulted by :class:`BudgetRouter` to decide whether a page can afford
        this backend, so it must not read the page or call the provider.

        :param page: the page that would be read
        :returns: a :class:`Cost`; ``Cost.zero()`` for local engines.  Money is
            ``0.00`` until :func:`Pricing.set_pricing` is called, though token
            counts are always real.
        """
        ...

    def read(self, page: Page) -> ReadResult:
        """Read ``page`` and return its spans, cost and timing.

        :param page: the page to read; implementations must not mutate it
        :returns: a :class:`ReadResult` whose spans carry text, geometry and
            per-span confidence.  A page the backend could not read yields an
            empty span list and a warning, not an exception -- one bad page must
            not cost the document.
        """
        ...


class BaseBackend(object):
    """Convenience base implementing the boring parts of the protocol."""

    name = "base"
    #: Whether this backend needs a rasterised page (as opposed to a text layer).
    needs_raster = True
    #: Preferred input DPI.  The reader upsamples toward this when cheap.
    preferred_dpi = DEFAULT_DPI

    def __init__(self, name: Optional[str] = None, **options: Any) -> None:
        """Configure the backend.

        :param name: overrides the class-level registry name, so the same engine
            can be registered twice with different options
        :param options: passed through to the underlying engine at construction
        """
        if name:
            self.name = name
        self.options = dict(options)
        self._lock = threading.Lock()

    # -- protocol ---------------------------------------------------------
    def supports(self, page: Page) -> bool:
        """Whether this backend can read ``page`` at all.

        :param page: the page in question
        :returns: ``True`` unless the page lacks a raster this backend needs,
            or :meth:`is_available` is ``False``.  Override to add
            engine-specific limits, such as a script or a page-size ceiling.
        """
        if self.needs_raster and not page.has_raster():
            return False
        return self.is_available()

    def is_available(self) -> bool:
        """Whether this backend's dependencies are importable right now."""
        return True

    def estimate_cost(self, page: Page) -> Cost:
        """Predicted cost of reading ``page``.  Free by default (local engines).

        :param page: the page that would be read
        :returns: ``Cost.zero()``.  Vision backends override this; see the
            per-token estimate in :meth:`VisionBackend.estimate_cost`.
        """
        return Cost.zero()

    def read(self, page: Page) -> ReadResult:
        """Read ``page``, timing the call and tagging spans with this backend.

        Do not override this; implement :meth:`_read` instead.  This wrapper is
        what fills in latency, stamps ``span.source`` with the backend name, and
        detects the script of each span -- three things every backend would
        otherwise have to remember.

        :param page: the page to read
        :returns: the :class:`ReadResult` from :meth:`_read`, with ``latency_ms``,
            ``backend``, and each span's ``source`` and ``script`` filled in
        """
        with Timer() as t:
            result = self._read(page)
        result.latency_ms = t.ms
        result.backend = result.backend or self.name
        for span in result.spans:
            if span.source == "unknown":
                span.source = self.name
            if span.script is None:
                span.script = detect_script(span.text)
        return result

    def _read(self, page: Page) -> ReadResult:
        """Do the actual reading.  Subclasses implement this, not :meth:`read`.

        Timing, span sourcing and script detection are handled by :meth:`read`,
        so an implementation only has to produce spans.
        """
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    def _image_for(self, page: Page) -> ImageArray:
        """The raster this backend should see, at its preferred DPI when free."""
        return page.raster()

    def _bbox(self, page: Page, x0: float, y0: float, x1: float, y1: float,
              dpi: Optional[float] = None) -> BBox:
        """Pixel coordinates from an engine -> canonical page points."""
        return BBox.from_pixels(page.index, x0, y0, x1, y1, dpi or page.raster_dpi)

    def __repr__(self) -> str:
        """``TesseractOCR(name='tesseract')``."""
        return "%s(name=%r)" % (type(self).__name__, self.name)


class BackendRegistry(object):
    """Name -> backend, with lazy construction so imports stay cheap.

    Registering a factory rather than an instance matters: constructing a
    PaddleOCR or Surya backend loads hundreds of megabytes of weights, and a
    document that never routes to it should never pay that.
    """

    def __init__(self) -> None:
        """Start empty.  Factories are registered, never instances."""
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, name: str, factory: Union[Callable[[], Any], Any],
                 replace: bool = False) -> None:
        """Register a backend under ``name``.

        Prefer a callable: instantiating PaddleOCR or Surya loads hundreds of
        megabytes of weights, and a document that never routes there must not
        pay for it.  Passing an instance is supported for cheap backends and
        test doubles, and constructs eagerly by definition::

            registry.register("myocr", lambda: MyOCRBackend(lang="hi"))

        :param name: the routing key, as used by ``backend="myocr"`` and by
            any :class:`RuleRouter` rule
        :param factory: a zero-argument callable returning a backend, or a
            backend instance (wrapped in a callable for you)
        :param replace: allow overwriting an existing registration.  Off by
            default so a name collision is an error rather than a backend that
            silently stops being used; any cached instance is dropped.
        :raises ConfigError: ``name`` is taken and ``replace`` is ``False``
        """
        if name in self._factories and not replace:
            raise ConfigError("backend %r is already registered (pass replace=True)" % name)
        with self._lock:
            self._factories[name] = factory if callable(factory) else (lambda f=factory: f)
            self._instances.pop(name, None)

    def get(self, name: str) -> Any:
        """Instantiate (once) and return the backend registered as ``name``.

        The instance is cached under the lock, so the weights load once even if
        several pages race for the same backend.

        :param name: a registered name, e.g. ``"tesseract"`` or ``"anthropic"``
        :returns: the shared backend instance
        :raises ConfigError: nothing is registered under ``name``; the message
            lists what is
        """
        if name in self._instances:
            return self._instances[name]
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            factory = self._factories.get(name)
            if factory is None:
                raise ConfigError("no backend named %r (registered: %s)"
                                  % (name, ", ".join(sorted(self._factories))))
            instance = factory()
            self._instances[name] = instance
            return instance

    def __contains__(self, name: str) -> bool:
        """``"tesseract" in registry`` -- true if registered, available or not.

        Registered is not the same as usable: use :meth:`available` to ask
        whether the engine's dependencies are actually importable.

        :param name: the name to look for
        :returns: whether a factory exists under that name
        """
        return name in self._factories

    def __getitem__(self, name: str) -> Any:
        """``registry["tesseract"]`` -- alias for :meth:`get`.

        :param name: a registered name
        :returns: the shared backend instance
        :raises ConfigError: nothing is registered under ``name``
        """
        return self.get(name)

    def names(self) -> List[str]:
        """Every registered name, sorted."""
        return sorted(self._factories)

    def available(self) -> List[str]:
        """Names whose dependencies are importable, without constructing them."""
        out = []
        for name in self.names():
            try:
                if self.get(name).is_available():
                    out.append(name)
            except Exception as exc:
                logger.debug("backend %r unavailable: %s", name, exc)
        return out

    def clear_instances(self) -> None:
        """Drop cached instances, keeping the factories.

        Next use reconstructs them -- which is how a test swaps a backend's
        configuration, and how a long-lived process releases model weights.
        """
        with self._lock:
            self._instances.clear()


#: The process-wide backend registry.  ``docpipe.registry.register("mine", ...)``
registry = BackendRegistry()


# -- native text layer -------------------------------------------------------

class PyMuPDFTextLayer(BaseBackend):
    """Return the PDF's own text layer.

    Exact by construction and free.  For a native-text page this is strictly
    better than any recogniser: a model reading a *picture* of text it could
    have read directly can only introduce errors, and will occasionally
    hallucinate a plausible number where the text layer has the real one.
    """

    name = "pymupdf"
    needs_raster = False

    def supports(self, page: Page) -> bool:
        """True only for a page that already carries a trustworthy text layer.

        Stricter than :meth:`BaseBackend.supports` on purpose: a scanned page in
        a PDF has a text layer of empty strings, and reading that would return
        nothing while looking like a success.

        :param page: the page in question
        :returns: ``True`` only when the page is ``DIGITAL_NATIVE`` or ``HYBRID``
            *and* already carries spans -- see :func:`Ingest.classify_page_kind`
        """
        return page.kind in (PageKind.DIGITAL_NATIVE, PageKind.HYBRID) and bool(page.spans)

    def _read(self, page: Page) -> ReadResult:
        """Return the page's existing native spans, re-tagged as this backend's."""
        spans = [dataclasses.replace(s, source=self.name)
                 for s in page.spans if s.source in ("pymupdf", "unknown") and s.text.strip()]
        return ReadResult(spans=spans, cost=Cost.zero(), backend=self.name)


# -- Tesseract ---------------------------------------------------------------

class TesseractOCR(BaseBackend):
    """Tesseract 5 via ``pytesseract``, or the ``tesseract`` binary directly.

    Still the strongest accuracy-per-millisecond option for clean, well-framed
    machine print, and the only one of these engines that runs anywhere without
    a model download.  It degrades sharply on degraded scans, handwriting and
    dense tables -- which is precisely what the router is for.

    Word-level confidences are reported, which matters: they are the
    ``backend_conf`` term in confidence fusion, and Tesseract's are reasonably
    well behaved (unlike a language model's self-report).
    """

    name = "tesseract"
    preferred_dpi = 300

    def __init__(self, lang: str = "eng", psm: int = 3, oem: int = 3, config: str = "",
                 min_confidence: float = 0.0, name: Optional[str] = None,
                 **options: Any) -> None:
        """Configure the engine.

        :param lang: Tesseract language code(s), ``+``-joined, e.g. ``"eng"``,
            ``"eng+hin"``, ``"eng+deu"``.  Each needs its traineddata installed;
            naming a missing one fails at read time, not here.
        :param psm: page segmentation mode.  ``3`` (fully automatic) is right for
            a whole page; ``6`` ("a single uniform block") is markedly better on
            a cropped table cell or a form field; ``7`` reads one line, ``11``
            finds sparse text in any order.
        :param oem: OCR engine mode.  ``3`` picks the best available, ``1``
            forces the LSTM engine, ``0`` the legacy one.  Leave it at ``3``
            unless you are reproducing an old result.
        :param config: extra flags appended to the command line, e.g.
            ``"-c tessedit_char_whitelist=0123456789.,"`` to read an
            amounts-only column.
        :param min_confidence: drop words below this confidence, in ``0.0..1.0``.
            ``0.0`` keeps everything, which is the right default -- confidence
            is a fusion input, and dropping words here hides evidence
            from :func:`locate_value` rather than improving accuracy.
        :param name: overrides the registry name, so the same engine can be
            registered twice with different options
        :param options: forwarded to :class:`BaseBackend`
        """
        BaseBackend.__init__(self, name=name, **options)
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.config = config
        self.min_confidence = min_confidence

    def is_available(self) -> bool:
        """True when either ``pytesseract`` or the ``tesseract`` binary is present."""
        return have("pytesseract") or shutil.which("tesseract") is not None

    def _config_string(self) -> str:
        """The ``--psm``/``--oem``/extra flags as one command-line string."""
        return ("--psm %d --oem %d %s" % (self.psm, self.oem, self.config)).strip()

    def _read(self, page: Page) -> ReadResult:
        """Read via ``pytesseract`` when importable, else via the binary."""
        img = self._image_for(page)
        pt = _try_import("pytesseract")
        if pt is not None:
            return self._read_pytesseract(page, img, pt)
        return self._read_cli(page, img)

    def _read_pytesseract(self, page: Page, img: ImageArray, pt: Any) -> ReadResult:
        """Read through ``pytesseract``'s word-level TSV output."""
        pil = require("PIL.Image", "Tesseract input conversion")
        data = pt.image_to_data(pil.fromarray(to_rgb(img)), lang=self.lang,
                                config=self._config_string(),
                                output_type=pt.Output.DICT)
        spans = []
        count = len(data.get("text", []))
        for i in range(count):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            # Tesseract signals "no confidence" with -1; keep that as None
            # rather than pretending it is zero.
            confidence = None if conf < 0 else conf / 100.0
            if confidence is not None and confidence < self.min_confidence:
                continue
            x, y = float(data["left"][i]), float(data["top"][i])
            w, h = float(data["width"][i]), float(data["height"][i])
            spans.append(TextSpan(
                text=text, bbox=self._bbox(page, x, y, x + w, y + h),
                source=self.name, confidence=confidence,
                line=int(data.get("line_num", [0] * count)[i]),
                block=int(data.get("block_num", [0] * count)[i])))
        return ReadResult(spans=spans, cost=Cost.zero(), backend=self.name)

    def _read_cli(self, page: Page, img: ImageArray) -> ReadResult:
        """Shell out to ``tesseract``, parsing TSV.  Used when pytesseract is absent."""
        binary = shutil.which("tesseract")
        if binary is None:
            raise MissingDependency("pytesseract", "Tesseract OCR", "pytesseract")
        tmpdir = tempfile.mkdtemp(prefix="docpipe-tess-")
        try:
            src = os.path.join(tmpdir, "page.png")
            save_image(img, src)
            cmd = [binary, src, os.path.join(tmpdir, "out"), "-l", self.lang,
                   "--psm", str(self.psm), "--oem", str(self.oem), "tsv"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                raise BackendError("tesseract failed: %s"
                                   % proc.stderr.decode("utf-8", "replace")[:400])
            with open(os.path.join(tmpdir, "out.tsv"), "r", encoding="utf-8") as fh:
                rows = [line.rstrip("\n").split("\t") for line in fh]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if not rows:
            return ReadResult(backend=self.name)
        header = rows[0]
        index = dict((name, i) for i, name in enumerate(header))
        spans = []
        for row in rows[1:]:
            if len(row) < len(header):
                continue
            text = row[index["text"]].strip()
            if not text:
                continue
            conf = float(row[index["conf"]])
            confidence = None if conf < 0 else conf / 100.0
            if confidence is not None and confidence < self.min_confidence:
                continue
            x, y = float(row[index["left"]]), float(row[index["top"]])
            w, h = float(row[index["width"]]), float(row[index["height"]])
            spans.append(TextSpan(text=text, bbox=self._bbox(page, x, y, x + w, y + h),
                                  source=self.name, confidence=confidence,
                                  line=int(row[index["line_num"]]),
                                  block=int(row[index["block_num"]])))
        return ReadResult(spans=spans, cost=Cost.zero(), backend=self.name)


# -- PP-OCR family (PaddleOCR and its ONNX repack, RapidOCR) -----------------

class PaddleOCRBackend(BaseBackend):
    """PaddleOCR (PP-OCR).

    Strong on ruled tables, dense layouts and CJK, and considerably more
    robust than Tesseract on moderately degraded scans.  Detection is
    quadrilateral, so rotated text is found correctly; we keep the axis-aligned
    hull for the IR and stash the original quad in ``span.meta``, because a
    reviewer highlight box wants the hull but a dewarper wants the quad.
    """

    name = "paddle"
    preferred_dpi = 300

    def __init__(self, lang: str = "en", use_angle_cls: bool = True,
                 name: Optional[str] = None, **options: Any) -> None:
        """Configure the engine.

        :param lang: PaddleOCR language code, e.g. ``"en"``, ``"ch"``,
            ``"devanagari"``.  One per engine instance -- register the backend
            twice under different names to read two scripts.
        :param use_angle_cls: run the text-direction classifier, needed for pages
            with rotated or vertical text.  Costs a little latency per page and
            is worth keeping on for anything photographed.
        :param name: overrides the registry name, so the same engine can be
            registered twice with different options
        :param options: forwarded to :class:`BaseBackend`
        """
        BaseBackend.__init__(self, name=name, **options)
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._engine = None

    def is_available(self) -> bool:
        """True when ``paddleocr`` is importable."""
        return have("paddleocr")

    def _get_engine(self) -> Any:
        """Construct the engine once, under the lock.  Loads model weights."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    module = require("paddleocr", "PaddleOCR backend")
                    self._engine = module.PaddleOCR(lang=self.lang,
                                                    use_angle_cls=self.use_angle_cls,
                                                    show_log=False, **self.options)
        return self._engine

    def _read(self, page: Page) -> ReadResult:
        """Run PP-OCR and convert its quads into spans."""
        np = _np()
        engine = self._get_engine()
        img = to_rgb(self._image_for(page))
        try:
            raw = engine.ocr(np.asarray(img), cls=self.use_angle_cls)
        except TypeError:  # PaddleOCR 3.x dropped the `cls` argument
            raw = engine.ocr(np.asarray(img))
        spans = []
        for line in (raw or []):
            for entry in (line or []):
                try:
                    quad, (text, score) = entry[0], entry[1]
                except (TypeError, ValueError, IndexError):
                    continue
                if not str(text).strip():
                    continue
                bbox = BBox.from_quad(page.index, quad)
                spans.append(TextSpan(
                    text=str(text),
                    bbox=BBox.from_pixels(page.index, bbox.x0, bbox.y0, bbox.x1,
                                          bbox.y1, page.raster_dpi),
                    source=self.name, confidence=float(score),
                    meta={"quad": [[float(p[0]), float(p[1])] for p in quad]}))
        return ReadResult(spans=spans, cost=Cost.zero(), backend=self.name, raw=raw)


class RapidOCRBackend(BaseBackend):
    """RapidOCR -- the PP-OCR models repacked for ONNX Runtime.

    Practically the same accuracy as PaddleOCR with a fraction of the install
    weight and no PaddlePaddle dependency, which makes it the pragmatic default
    for on-premise deployments where documents cannot leave the estate.
    """

    name = "rapidocr"
    preferred_dpi = 300

    def __init__(self, name: Optional[str] = None, **options: Any) -> None:
        """Configure the engine.

        :param name: overrides the registry name, so the same engine can be
            registered twice with different options
        :param options: passed straight to ``RapidOCR`` at construction, e.g.
            ``det_model_path=...`` to point at your own ONNX weights
        """
        BaseBackend.__init__(self, name=name, **options)
        self._engine = None

    def is_available(self) -> bool:
        """True when either RapidOCR distribution is importable."""
        return have("rapidocr_onnxruntime") or have("rapidocr")

    def _get_engine(self) -> Any:
        """Construct the engine once, under the lock.  Loads ONNX weights."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    module = _try_import("rapidocr_onnxruntime") or _try_import("rapidocr")
                    if module is None:
                        raise MissingDependency("rapidocr_onnxruntime", "RapidOCR backend",
                                                "rapidocr-onnxruntime")
                    self._engine = module.RapidOCR(**self.options)
        return self._engine

    def _read(self, page: Page) -> ReadResult:
        """Run RapidOCR and convert its quads into spans."""
        np = _np()
        engine = self._get_engine()
        result = engine(np.asarray(to_rgb(self._image_for(page))))
        # 1.x returns (results, elapsed); 2.x returns a RapidOCROutput object.
        entries = result[0] if isinstance(result, tuple) else getattr(result, "boxes", None)
        if entries is None:
            entries = result
        spans = []
        for entry in (entries or []):
            try:
                quad, text, score = entry[0], entry[1], entry[2]
            except (TypeError, IndexError):
                continue
            if not str(text).strip():
                continue
            hull = BBox.from_quad(page.index, quad)
            spans.append(TextSpan(
                text=str(text),
                bbox=BBox.from_pixels(page.index, hull.x0, hull.y0, hull.x1, hull.y1,
                                      page.raster_dpi),
                source=self.name, confidence=float(score)))
        return ReadResult(spans=spans, cost=Cost.zero(), backend=self.name, raw=result)


class EasyOCRBackend(BaseBackend):
    """EasyOCR.  Broad script coverage and easy setup; slower than PP-OCR."""

    name = "easyocr"
    preferred_dpi = 300

    def __init__(self, languages: Sequence[str] = ("en",), gpu: bool = False,
                 name: Optional[str] = None, **options: Any) -> None:
        """Configure the engine.

        :param languages: EasyOCR language codes loaded together, e.g.
            ``("en",)`` or ``("en", "hi")``.  EasyOCR restricts which codes may
            be combined -- Latin scripts mix freely, but most non-Latin scripts
            pair only with English.
        :param gpu: use CUDA if EasyOCR finds it.  ``False`` is the safe default;
            on CPU this engine is several times slower than PP-OCR.
        :param name: overrides the registry name, so the same engine can be
            registered twice with different options
        :param options: forwarded to :class:`BaseBackend`
        """
        BaseBackend.__init__(self, name=name, **options)
        self.languages = list(languages)
        self.gpu = gpu
        self._engine = None

    def is_available(self) -> bool:
        """True when ``easyocr`` is importable."""
        return have("easyocr")

    def _get_engine(self) -> Any:
        """Construct the reader once, under the lock.  Loads model weights."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    module = require("easyocr", "EasyOCR backend")
                    self._engine = module.Reader(self.languages, gpu=self.gpu, **self.options)
        return self._engine

    def _read(self, page: Page) -> ReadResult:
        """Run EasyOCR and convert its quads into spans."""
        np = _np()
        engine = self._get_engine()
        raw = engine.readtext(np.asarray(to_rgb(self._image_for(page))))
        spans = []
        for quad, text, score in raw or []:
            if not str(text).strip():
                continue
            hull = BBox.from_quad(page.index, quad)
            spans.append(TextSpan(
                text=str(text),
                bbox=BBox.from_pixels(page.index, hull.x0, hull.y0, hull.x1, hull.y1,
                                      page.raster_dpi),
                source=self.name, confidence=float(score)))
        return ReadResult(spans=spans, cost=Cost.zero(), backend=self.name, raw=raw)


class DocTRBackend(BaseBackend):
    """docTR (Mindee).  Two-stage detection + recognition, both swappable.

    Worth having when you want to tune the accuracy/latency trade-off yourself
    rather than accept an engine's fixed choice.  Geometry comes back relative
    to the page, so it is scaled here rather than left for callers to get wrong.
    """

    name = "doctr"
    preferred_dpi = 300

    def __init__(self, det_arch: str = "db_resnet50",
                 reco_arch: str = "crnn_vgg16_bn", pretrained: bool = True,
                 name: Optional[str] = None, **options: Any) -> None:
        """Configure the two-stage predictor.

        :param det_arch: text-detection architecture.  ``"db_resnet50"`` is the
            accurate default; ``"db_mobilenet_v3_large"`` is markedly faster and
            a little worse on small print.
        :param reco_arch: recognition architecture.  ``"crnn_vgg16_bn"`` is the
            balanced default; ``"crnn_mobilenet_v3_small"`` trades accuracy for
            speed, and ``"master"`` or ``"parseq"`` cost more for better results
            on irregular text.
        :param pretrained: download pretrained weights rather than start cold.
            ``False`` is only useful when loading your own fine-tuned weights --
            an untrained model reads nothing.
        :param name: overrides the registry name, so the same engine can be
            registered twice with different options
        :param options: forwarded to :class:`BaseBackend`
        """
        BaseBackend.__init__(self, name=name, **options)
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained
        self._engine = None

    def is_available(self) -> bool:
        """True when ``doctr`` is importable."""
        return have("doctr")

    def _get_engine(self) -> Any:
        """Build the OCR predictor once, under the lock.  Loads model weights."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    models = require("doctr.models", "docTR backend")
                    self._engine = models.ocr_predictor(
                        det_arch=self.det_arch, reco_arch=self.reco_arch,
                        pretrained=self.pretrained, **self.options)
        return self._engine

    def _read(self, page: Page) -> ReadResult:
        """Run docTR and scale its relative geometry back into pixels."""
        np = _np()
        engine = self._get_engine()
        img = to_rgb(self._image_for(page))
        height, width = image_shape(img)
        export = engine([np.asarray(img)]).export()
        spans = []
        for doc_page in export.get("pages", []):
            for block_no, block in enumerate(doc_page.get("blocks", [])):
                for line_no, line in enumerate(block.get("lines", [])):
                    for word in line.get("words", []):
                        text = str(word.get("value", "")).strip()
                        if not text:
                            continue
                        (rx0, ry0), (rx1, ry1) = word["geometry"]
                        spans.append(TextSpan(
                            text=text,
                            bbox=BBox.from_pixels(page.index, rx0 * width, ry0 * height,
                                                  rx1 * width, ry1 * height,
                                                  page.raster_dpi),
                            source=self.name,
                            confidence=float(word.get("confidence", 0.0)),
                            line=line_no, block=block_no))
        return ReadResult(spans=spans, cost=Cost.zero(), backend=self.name, raw=export)


class SuryaBackend(BaseBackend):
    """Surya -- detection, recognition and layout across 90+ scripts.

    The best open option when a page mixes scripts, which is the normal case
    for Indian documents (a Devanagari letterhead over a Latin table of
    amounts).  Heavy: it wants a GPU to be quick.

    Surya's API has moved several times; the loader below tries the current
    shape first and falls back through the two previous ones rather than
    pinning consumers to one release.
    """

    name = "surya"
    preferred_dpi = 300

    def __init__(self, languages: Sequence[str] = ("en",),
                 name: Optional[str] = None, **options: Any) -> None:
        """Configure the engine.

        :param languages: script/language hints passed to the recogniser, e.g.
            ``("en",)`` or ``("en", "hi")``.  Surya detects script itself, so
            these are hints rather than a restriction -- which is why it handles
            a Devanagari letterhead over a Latin amounts table without being
            told the page is mixed.
        :param name: overrides the registry name, so the same engine can be
            registered twice with different options
        :param options: forwarded to :class:`BaseBackend`
        """
        BaseBackend.__init__(self, name=name, **options)
        self.languages = list(languages)
        self._predictors = None

    def is_available(self) -> bool:
        """True when ``surya`` is importable."""
        return have("surya")

    def _get_predictors(self) -> Tuple[Any, Any]:
        """Build ``(detection, recognition)`` predictors once, under the lock.

        Surya's constructor signature has changed across releases, so several
        known variants are tried before giving up with a clear error.
        """
        if self._predictors is not None:
            return self._predictors
        with self._lock:
            if self._predictors is not None:
                return self._predictors
            recognition = require("surya.recognition", "Surya backend")
            detection = require("surya.detection", "Surya backend")
            det = detection.DetectionPredictor()
            rec = None
            for build in (self._build_with_manager, self._build_with_foundation,
                          lambda cls: cls()):
                try:
                    rec = build(recognition.RecognitionPredictor)
                    break
                except Exception as exc:
                    logger.debug("Surya init variant failed: %s", exc)
            if rec is None:
                raise BackendError("could not construct a Surya RecognitionPredictor; "
                                   "the installed surya-ocr API is unrecognised")
            self._predictors = (det, rec)
            return self._predictors

    @staticmethod
    def _build_with_manager(predictor_cls: Any) -> Any:
        """Construct a predictor via the older ``SuryaInferenceManager`` API."""
        inference = require("surya.inference", "Surya backend")
        return predictor_cls(inference.SuryaInferenceManager())

    @staticmethod
    def _build_with_foundation(predictor_cls: Any) -> Any:
        """Construct a predictor via the newer ``FoundationPredictor`` API."""
        foundation = require("surya.foundation", "Surya backend")
        return predictor_cls(foundation.FoundationPredictor())

    def _read(self, page: Page) -> ReadResult:
        """Run Surya and convert its text lines into spans."""
        pil = require("PIL.Image", "Surya input conversion")
        det, rec = self._get_predictors()
        image = pil.fromarray(to_rgb(self._image_for(page)))
        try:
            predictions = rec([image], [self.languages], det)
        except TypeError:
            predictions = rec([image])
        spans = []
        for prediction in predictions or []:
            for line in getattr(prediction, "text_lines", []) or []:
                text = str(getattr(line, "text", "")).strip()
                if not text:
                    continue
                box = list(getattr(line, "bbox", []) or [])
                if len(box) != 4:
                    continue
                confidence = getattr(line, "confidence", None)
                spans.append(TextSpan(
                    text=text,
                    bbox=BBox.from_pixels(page.index, box[0], box[1], box[2], box[3],
                                          page.raster_dpi),
                    source=self.name,
                    confidence=float(confidence) if confidence is not None else None))
        return ReadResult(spans=spans, cost=Cost.zero(), backend=self.name, raw=predictions)


# -- cost model --------------------------------------------------------------
#
# Cost accounting is only useful if it is honest, so this table starts empty of
# guesses.  Prices change without notice and a *wrong* number is worse than a
# missing one: it silently produces a plausible budget report that nobody
# rechecks.  Register the prices your account actually pays.

#: ``model id -> (input USD per million tokens, output USD per million tokens)``
PRICING: Dict[str, Tuple[float, float]] = {}

_UNPRICED_WARNED: Set[str] = set()


class Pricing(object):
    """Token accounting and the optional price table.

    Token counts are always tracked.  Money reads ``0.00`` and warns until
    a consumer calls :func:`Pricing.set_pricing`, because :data:`PRICING`
    ships empty on purpose: a stale hard-coded rate silently produces
    confidently wrong numbers, which is worse than an obvious zero.
    """

    @staticmethod
    def set_pricing(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
        """Register token prices (USD per million tokens) for ``model``.

        Until a model is priced, :class:`Cost` reports token counts with a zero
        amount and warns once, so a missing price is visible rather than silently
        reported as free.

        Register what your account actually pays, not the list price -- committed
        spend and enterprise agreements both move it, and the point of this table
        is that the number in the report is one you can defend::

            Pricing.set_pricing("claude-sonnet-4-5", 3.0, 15.0)

        :param model: the model id exactly as the backend reports it in
            ``Cost.model``; a mismatched string leaves the model unpriced
        :param input_per_mtok: USD per million input tokens, e.g. ``3.0``
        :param output_per_mtok: USD per million output tokens, e.g. ``15.0``,
            typically several times the input rate
        """
        PRICING[model] = (float(input_per_mtok), float(output_per_mtok))
        _UNPRICED_WARNED.discard(model)

    @staticmethod
    def price_tokens(model: str, input_tokens: int, output_tokens: int) -> Cost:
        """Convert token counts to a :class:`Cost` using the registered pricing.

        :param model: the model id to look up in :data:`PRICING`
        :param input_tokens: prompt tokens consumed
        :param output_tokens: completion tokens produced
        :returns: a :class:`Cost` for one call.  When the model is unpriced the
            token counts are still exact and ``amount`` is ``0.0``, with a warning
            logged once per model -- never an exception, since an unpriced model
            must not stop a document being read.
        """
        prices = PRICING.get(model)
        if prices is None:
            if model not in _UNPRICED_WARNED:
                _UNPRICED_WARNED.add(model)
                logger.warning(
                    "no pricing registered for model %r: token counts will be tracked "
                    "but cost will read 0.00.  Call docpipe.set_pricing(%r, in, out).",
                    model, model)
            return Cost(amount=0.0, input_tokens=input_tokens,
                        output_tokens=output_tokens, calls=1)
        amount = (input_tokens * prices[0] + output_tokens * prices[1]) / 1e6
        return Cost(amount=amount, input_tokens=input_tokens,
                    output_tokens=output_tokens, calls=1)

    @staticmethod
    def estimate_image_tokens(img: ImageArray, divisor: float = 750.0) -> int:
        """Approximate the token cost of an image for a vision model.

        Providers bill images roughly by area; ``width * height / 750`` is the
        documented approximation for Claude and is close enough for OpenAI's
        high-detail mode to be useful for *budgeting*.  It is an estimate, and the
        reconciled figure from the response's usage block always wins.

        :param img: the raster that would be sent
        :param divisor: pixels per token.  ``750`` is Anthropic's documented
            approximation and is close enough for OpenAI high-detail mode to be
            useful for budgeting.  Lower it to estimate more conservatively.
        :returns: an estimated token count, rounded up.  Use it for pre-flight
            budgeting only -- :class:`BudgetRouter` does, and then reconciles
            against the real usage once the call returns.
        """
        h, w = image_shape(img)
        return int(math.ceil(w * h / float(divisor)))


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Pricing from the inside.
# ``Pricing.x`` is the documented path; ``x`` is the one that already works.
set_pricing           = Pricing.set_pricing
price_tokens          = Pricing.price_tokens
estimate_image_tokens = Pricing.estimate_image_tokens


#: The instruction given to vision models when transcribing a page.  Written to
#: suppress the two failure modes that matter: summarising instead of
#: transcribing, and inventing plausible-looking values for illegible regions.
DEFAULT_OCR_PROMPT = (
    "Transcribe every character of text visible in this document image, exactly "
    "as printed, preserving reading order and line breaks.\n"
    "Rules:\n"
    "- Do not summarise, correct, translate or reformat anything.\n"
    "- Preserve original spelling, punctuation, casing and digit grouping.\n"
    "- Keep table rows on one line, separating cells with ' | '.\n"
    "- If a character or word is illegible, write [illegible] rather than "
    "guessing a plausible value.\n"
    "- Output only the transcription, with no preamble or commentary."
)


class VisionBackend(BaseBackend):
    """Base for vision-language model readers.

    A VLM is the right tool for degraded scans, handwriting, and multi-script
    pages -- the cases where classical OCR's character models have nothing to
    work with.  It is the wrong tool for a native text layer, and an expensive
    tool for a clean printed page, which is exactly what routing is for.

    The provenance a VLM returns is weak: it reads the whole page and gives
    back a wall of text with no coordinates.  Rather than fabricate bboxes, one
    span per line is emitted with a page-level box, and the line index is
    recorded.  :func:`locate_value` can then recover tighter boxes by matching
    the value against a *second*, geometry-bearing read -- which is also the
    cross-read agreement signal.  Fabricated coordinates would be worse than
    none, because a reviewer would be pointed at the wrong part of the page.
    """

    name = "vlm"
    preferred_dpi = 200
    #: Vision models see no benefit above this; the extra pixels are pure cost.
    max_side_px = 2000

    def __init__(self, model: str = "", prompt: str = DEFAULT_OCR_PROMPT,
                 max_tokens: int = 8192, temperature: float = 0.0,
                 name: Optional[str] = None, **options: Any) -> None:
        """Configure the model call.

        :param model: provider model id, e.g. ``"claude-sonnet-4-5"`` or
            ``"gpt-4o"``.  Used as the pricing key as well as in the request, so
            it must match what you passed to :func:`Pricing.set_pricing`.
        :param prompt: transcription instructions.  The default value
            is :data:`DEFAULT_OCR_PROMPT`, tuned to suppress the
            summarising and reordering a general prompt invites.  Override it to
            add domain vocabulary, never to ask for interpretation -- that
            belongs in :func:`extract`.
        :param max_tokens: ceiling on the transcription length.  ``8192`` covers
            a dense A4 page; raise it for large-format or multi-column pages,
            where a truncated read silently loses the bottom of the page.
        :param temperature: ``0.0``, because transcription is recall, not
            creativity.  Raising it produces plausible text that is not on the
            page, which is the one failure mode confidence cannot catch.
        :param name: overrides the registry name, so the same model can be
            registered twice with different prompts
        :param options: forwarded to :class:`BaseBackend`
        """
        BaseBackend.__init__(self, name=name, **options)
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _image_for(self, page: Page) -> ImageArray:
        """The page raster, downscaled to ``max_side_px`` so cost stays bounded."""
        img = page.raster()
        h, w = image_shape(img)
        longest = max(h, w)
        if longest > self.max_side_px:
            img = resize_image(img, scale=float(self.max_side_px) / longest,
                               interpolation="area")
        return img

    def estimate_cost(self, page: Page) -> Cost:
        """Price the call before making it, from image tokens plus prompt.

        Approximate by construction -- it is what :class:`BudgetRouter` decides
        on, so it errs toward the honest side rather than the flattering one.

        :param page: the page that would be read
        :returns: estimated :class:`Cost` from image tokens plus prompt length,
            with output guessed from the page's character count.
            ``Cost.zero()`` for a page with no raster.  Reads ``0.00`` until the
            model is priced.
        """
        if not page.has_raster():
            return Cost.zero()
        tokens = estimate_image_tokens(self._image_for(page)) + len(self.prompt) // 4
        # Assume the transcription is about as long as the page's text.
        output = max(256, page.char_count // 3 or 800)
        return price_tokens(self.model, tokens, output)

    def _spans_from_text(self, page: Page, text: str) -> List[TextSpan]:
        """Turn a transcription into one span per line, with honest geometry."""
        lines = [l for l in (text or "").splitlines()]
        spans = []
        kept = [l for l in lines if l.strip()]
        if not kept:
            return spans
        # Distribute lines down the page so ordering and rough position are
        # right even though exact boxes are unknown.  meta marks them approximate.
        step = page.height_pt / float(len(kept))
        for i, line in enumerate(kept):
            spans.append(TextSpan(
                text=line.strip(),
                bbox=BBox(page.index, 0.0, i * step, page.width_pt, (i + 1) * step),
                source=self.name, confidence=None, line=i,
                meta={"approximate_bbox": True}))
        return spans

    def _call_model(self, image_bytes: bytes, media_type: str,
                    prompt: str) -> Tuple[str, Cost]:
        """Send one image and prompt to the provider.

        Subclasses implement this; returns ``(transcription, cost)``.
        """
        raise NotImplementedError

    def _read(self, page: Page) -> ReadResult:
        """Encode the page as JPEG, transcribe it, and split it into line spans."""
        img = self._image_for(page)
        data = encode_jpeg(img, quality=90)
        text, cost = self._call_model(data, "image/jpeg", self.prompt)
        result = ReadResult(spans=self._spans_from_text(page, text), cost=cost,
                            backend=self.name, raw=text)
        if "[illegible]" in (text or ""):
            result.warnings.append("model reported illegible regions on page %d" % page.index)
        return result


class AnthropicVisionBackend(VisionBackend):
    """Read a page with a Claude vision model."""

    name = "vlm:claude"

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None,
                 client: Any = None, **options: Any) -> None:
        """Configure the Claude vision call.

        :param model: Claude model id, e.g. ``"claude-sonnet-5"`` (the default,
            the best accuracy-per-rupee for transcription) or
            ``"claude-opus-5"`` for the hardest handwriting.  Also the pricing
            key -- see :func:`Pricing.set_pricing`.
        :param api_key: falls back to the ``ANTHROPIC_API_KEY`` environment
            variable when ``None``
        :param client: a pre-built SDK client, e.g. one with custom transport,
            a proxy, or a Bedrock/Vertex wrapper.  When given, ``api_key`` is
            unused and :meth:`is_available` is ``True`` without the SDK check.
        :param options: forwarded to :class:`VisionBackend` -- ``prompt``,
            ``max_tokens``, ``temperature``, ``name``
        """
        VisionBackend.__init__(self, model=model, **options)
        self._api_key = api_key
        self._client = client

    def is_available(self) -> bool:
        """True with an injected client, or with the SDK plus an API key."""
        if self._client is not None:
            return True
        return have("anthropic") and bool(self._api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self) -> Any:
        """Construct the SDK client once, under the lock."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    module = require("anthropic", "Claude vision backend")
                    self._client = module.Anthropic(api_key=self._api_key) \
                        if self._api_key else module.Anthropic()
        return self._client

    def _call_model(self, image_bytes: bytes, media_type: str,
                    prompt: str) -> Tuple[str, Cost]:
        """One Messages API call; returns the text and its billed cost."""
        client = self._get_client()
        message = client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": base64.b64encode(image_bytes).decode("ascii")}},
                {"type": "text", "text": prompt},
            ]}])
        text = "".join(getattr(block, "text", "") for block in getattr(message, "content", []))
        usage = getattr(message, "usage", None)
        cost = price_tokens(self.model,
                            int(getattr(usage, "input_tokens", 0) or 0),
                            int(getattr(usage, "output_tokens", 0) or 0))
        return text, cost


class OpenAIVisionBackend(VisionBackend):
    """Read a page with an OpenAI vision model."""

    name = "vlm:openai"

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 client: Any = None, detail: str = "high",
                 **options: Any) -> None:
        """Configure the OpenAI vision call.

        :param model: OpenAI model id, e.g. ``"gpt-4o"``.  Also the pricing key.
        :param api_key: falls back to the ``OPENAI_API_KEY`` environment
            variable when ``None``
        :param client: a pre-built SDK client; when given, ``api_key`` is unused
        :param detail: image fidelity hint -- ``"high"`` tiles the image and
            reads small print, ``"low"`` sends a single low-resolution pass for
            far fewer tokens, and ``"auto"`` lets the provider choose.  Keep
            ``"high"`` for documents; ``"low"`` loses most printed detail.
        :param options: forwarded to :class:`VisionBackend` -- ``prompt``,
            ``max_tokens``, ``temperature``, ``name``
        """
        VisionBackend.__init__(self, model=model, **options)
        self._api_key = api_key
        self._client = client
        self.detail = detail

    def is_available(self) -> bool:
        """True with an injected client, or with the SDK plus an API key."""
        if self._client is not None:
            return True
        return have("openai") and bool(self._api_key or os.environ.get("OPENAI_API_KEY"))

    def _get_client(self) -> Any:
        """Construct the SDK client once, under the lock."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    module = require("openai", "OpenAI vision backend")
                    self._client = module.OpenAI(api_key=self._api_key) \
                        if self._api_key else module.OpenAI()
        return self._client

    def _call_model(self, image_bytes: bytes, media_type: str,
                    prompt: str) -> Tuple[str, Cost]:
        """One chat-completions call; returns the text and its billed cost."""
        client = self._get_client()
        url = "data:%s;base64,%s" % (media_type, base64.b64encode(image_bytes).decode("ascii"))
        response = client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url, "detail": self.detail}},
            ]}])
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        cost = price_tokens(self.model,
                            int(getattr(usage, "prompt_tokens", 0) or 0),
                            int(getattr(usage, "completion_tokens", 0) or 0))
        return text, cost


class GeminiVisionBackend(VisionBackend):
    """Read a page with a Google Gemini vision model.

    Uses the unified ``google-genai`` SDK (``from google import genai``), which
    reaches both the Gemini Developer API and Vertex AI.  The legacy
    ``google-generativeai`` package is a different, incompatible API and is not
    supported; for Vertex, build the client yourself and pass it as ``client``,
    since Vertex authenticates by project and location rather than by API key.

    Gemini bills images by tile rather than by area, which is why
    :meth:`estimate_cost` is overridden -- see there.
    """

    name = "vlm:gemini"
    #: Gemini crops a large image into tiles of this size, billing each one.
    tile_px = 768
    #: Tokens per tile, and the flat cost of an image small enough not to be tiled.
    tokens_per_tile = 258
    #: An image with both sides under this is sent whole, at one tile's price.
    untiled_max_px = 384

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None,
                 client: Any = None, thinking_budget: Optional[int] = 0,
                 **options: Any) -> None:
        """Configure the Gemini vision call.

        :param model: Gemini model id, e.g. ``"gemini-2.5-flash"`` (fast and
            cheap, ample for printed pages) or ``"gemini-2.5-pro"`` for harder
            reads.  Also the pricing key.
        :param api_key: falls back to ``GOOGLE_API_KEY``, then ``GEMINI_API_KEY``
        :param client: a pre-built ``genai.Client``, e.g. one pointed at Vertex AI
        :param thinking_budget: reasoning tokens allowed before the answer.
            Defaults to 0, which disables thinking: transcription is recall, not
            reasoning, and thinking tokens bill at the output rate for output the
            page never needed.  Pass ``None`` to omit the setting altogether,
            which is required for models that do not support thinking at all.
        :param options: merged into the generation config, so ``safety_settings``
            or ``top_p`` need no subclass
        """
        VisionBackend.__init__(self, model=model, **options)
        self._api_key = api_key
        self._client = client
        self.thinking_budget = thinking_budget

    def is_available(self) -> bool:
        """True with an injected client, or with the SDK plus an API key.

        Probes ``google.genai`` rather than ``google``: the latter is a namespace
        package that a dozen unrelated Google libraries populate, so its
        importability would say nothing about whether this SDK is installed.
        """
        if self._client is not None:
            return True
        return have("google.genai") and bool(
            self._api_key or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY"))

    def _get_client(self) -> Any:
        """Construct the SDK client once, under the lock."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    module = require("google.genai", "Gemini vision backend")
                    self._client = module.Client(api_key=self._api_key) \
                        if self._api_key else module.Client()
        return self._client

    def estimate_cost(self, page: Page) -> Cost:
        """Price the call before making it, using Gemini's tiling rule.

        The base class's area/750 approximation is Claude's rule and is wrong
        here by enough to matter to :class:`BudgetRouter`: Gemini charges a flat
        :attr:`tokens_per_tile` for a small image and the same again for every
        :attr:`tile_px` tile of a large one, so cost is a step function of size,
        not a line.  Still an estimate -- the response's usage block always wins.

        :param page: the page that would be read
        :returns: estimated :class:`Cost`, or ``Cost.zero()`` for a page with no
            raster.  Reads ``0.00`` until the model is priced.
        """
        if not page.has_raster():
            return Cost.zero()
        h, w = image_shape(self._image_for(page))
        if max(h, w) <= self.untiled_max_px:
            tiles = 1
        else:
            tiles = (int(math.ceil(w / float(self.tile_px)))
                     * int(math.ceil(h / float(self.tile_px))))
        tokens = self.tokens_per_tile * max(1, tiles) + len(self.prompt) // 4
        output = max(256, page.char_count // 3 or 800)
        return price_tokens(self.model, tokens, output)

    def _config(self) -> Dict[str, Any]:
        """The generation config for one transcription call.

        ``self.options`` is applied last, so a consumer can override anything
        here -- including the thinking budget -- without subclassing.
        """
        config: Dict[str, Any] = {"temperature": self.temperature,
                                  "max_output_tokens": self.max_tokens}
        if self.thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": self.thinking_budget}
        config.update(self.options)
        return config

    @staticmethod
    def _text_of(response: Any) -> str:
        """The transcription, or :class:`BackendError` explaining its absence.

        Gemini reports a refused or truncated generation as an empty candidate
        with a reason attached, not as an exception, so ``response.text`` is
        simply ``None``.  Reading that as "this page had no text" would put a
        silently blank page into the document, which is the one outcome nobody
        catches in review.  Raising instead is safe at document scale:
        :func:`read` records the failure against the page and keeps going, so
        one refusal costs one page rather than the whole bundle.

        An empty response with *no* reason given is passed through as empty --
        a genuinely blank page is a legitimate answer, and inventing a failure
        for it would be as wrong in the other direction.
        """
        text = getattr(response, "text", None)
        candidates = list(getattr(response, "candidates", None) or [])
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        if text:
            if finish is not None and "MAX_TOKENS" in str(finish):
                logger.warning("Gemini hit max_output_tokens; the transcription is "
                               "truncated.  Raise max_tokens or split the page.")
            return text
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None)
        if reason is None and finish is not None and "STOP" not in str(finish):
            reason = finish
        if reason is not None:
            raise BackendError("Gemini returned no transcription (%s)" % (reason,))
        return ""

    def _call_model(self, image_bytes: bytes, media_type: str,
                    prompt: str) -> Tuple[str, Cost]:
        """One ``generate_content`` call; returns the text and its billed cost."""
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [
                {"inline_data": {"mime_type": media_type, "data": image_bytes}},
                {"text": prompt},
            ]}],
            config=self._config())
        usage = getattr(response, "usage_metadata", None)
        # Thinking tokens bill at the output rate but are counted separately from
        # the candidate, so charging candidates_token_count alone understates it.
        output = (int(getattr(usage, "candidates_token_count", 0) or 0)
                  + int(getattr(usage, "thoughts_token_count", 0) or 0))
        cost = price_tokens(self.model,
                            int(getattr(usage, "prompt_token_count", 0) or 0),
                            output)
        return self._text_of(response), cost


# -- backend wrappers --------------------------------------------------------

class CachingBackend(BaseBackend):
    """Memoise reads by raster content hash.

    Keyed on the pixels, not the file: after a preprocessing change the cache
    correctly misses, and two identical pages in different documents correctly
    hit.  This is what makes re-running an eval suite over a hundred documents
    cheap enough to do on every commit.
    """

    def __init__(self, inner: Any, cache: Optional["DiskCache"] = None,
                 namespace: str = "") -> None:
        """Wrap ``inner`` with a content-addressed read cache.

        :param inner: the backend to memoise; its ``needs_raster`` is inherited
        :param cache: where reads are stored; ``None`` builds
            a :class:`DiskCache` in the standard location.  Point it at a
            per-branch directory to keep experiments from sharing results.
        :param namespace: mixed into every key, to separate unrelated runs that
            would otherwise collide -- e.g. ``"v2-prompt"`` after changing the
            OCR prompt, since the prompt is not part of the raster hash.
        """
        BaseBackend.__init__(self, name="cached:%s" % inner.name)
        self.inner = inner
        self.cache = cache if cache is not None else DiskCache()
        self.namespace = namespace
        self.needs_raster = getattr(inner, "needs_raster", True)
        self.hits = 0
        self.misses = 0

    def is_available(self) -> bool:
        """Delegates to the wrapped backend."""
        return self.inner.is_available()

    def supports(self, page: Page) -> bool:
        """Delegates to the wrapped backend.

        :param page: the page in question
        :returns: whatever the wrapped backend answers -- caching never widens
            what can be read
        """
        return self.inner.supports(page)

    def estimate_cost(self, page: Page) -> Cost:
        """Zero on a cache hit, otherwise the wrapped backend's estimate.

        This is what lets :class:`BudgetRouter` send a repeated page to an
        expensive backend it could not otherwise afford.

        :param page: the page that would be read
        :returns: ``Cost.zero()`` when the key is already cached, else the
            wrapped backend's estimate
        """
        return Cost.zero() if self._key(page) in self.cache else self.inner.estimate_cost(page)

    def _key(self, page: Page) -> str:
        """Cache key: the raster's pixels, or the span text when there is no raster.

        Hashing the content rather than the filename is what makes the cache
        correct across preprocessing changes.
        """
        raster = array_hash(page.raster()) if page.has_raster() else ""
        spans = stable_hash([s.to_dict() for s in page.spans]) if not raster else ""
        return stable_hash(self.namespace, self.inner.name, raster, spans)

    def read(self, page: Page) -> ReadResult:
        """Return the cached read if there is one, else read and store it.

        Hits and misses accumulate on ``self.hits`` and ``self.misses``, which is
        how you check the cache is actually working rather than quietly missing
        on every page.

        :param page: the page to read
        :returns: a :class:`ReadResult`.  On a hit, ``cost`` is zero and
            ``latency_ms`` is ``0.0`` -- the cached run's figures would
            misreport this run.  Spans and warnings are restored; ``raw`` is not,
            being unbounded in size.
        """
        key = self._key(page)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            spans = [TextSpan.from_dict(s) for s in cached.get("spans", [])]
            return ReadResult(spans=spans, cost=Cost.zero(),
                              latency_ms=0.0, backend=self.inner.name,
                              warnings=list(cached.get("warnings", [])))
        self.misses += 1
        result = self.inner.read(page)
        self.cache.set(key, {"spans": [s.to_dict() for s in result.spans],
                             "warnings": result.warnings})
        return result


class RetryingBackend(BaseBackend):
    """Retry a flaky backend with exponential backoff and honest accounting.

    The subtlety this exists for: a provider that times out *after* generating
    tokens has already billed for them.  Retrying is right, but reporting the
    retry as free is not -- so failed attempts are counted in
    ``Cost.wasted_calls`` and surfaced, which is how a pipeline that quietly
    doubled its bill gets noticed.
    """

    def __init__(self, inner: TextBackend, attempts: int = 3,
                 base_delay: float = 0.5,
                 retry_on: Tuple[type, ...] = (Exception,),
                 give_up_on: Tuple[type, ...] = (MissingDependency,
                                                 ConfigError)) -> None:
        """Wrap ``inner`` with bounded retries.

        :param inner: the backend to wrap
        :param attempts: total tries, including the first.  ``3`` absorbs the
            usual transient provider failure; beyond ``5`` you are queueing on an
            outage rather than retrying a blip.
        :param base_delay: seconds before the first retry, doubling thereafter --
            ``0.5`` gives 0.5s, 1s, 2s.  Raise it when the provider rate-limits
            by window rather than by concurrency.
        :param retry_on: exception types worth retrying.  ``(Exception,)`` is
            deliberately broad, since provider SDKs raise their own hierarchies;
            narrow it if you would rather fail fast on unrecognised errors.
        :param give_up_on: exception types that never become successes -- these
            win over ``retry_on``, so a bad API key fails once, not three times
        """
        BaseBackend.__init__(self, name="retry:%s" % inner.name)
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay
        self.retry_on = retry_on
        self.give_up_on = give_up_on
        self.needs_raster = getattr(inner, "needs_raster", True)

    def is_available(self) -> bool:
        """Delegates to the wrapped backend."""
        return self.inner.is_available()

    def supports(self, page: Page) -> bool:
        """Delegates to the wrapped backend.

        :param page: the page in question
        :returns: whatever the wrapped backend answers
        """
        return self.inner.supports(page)

    def estimate_cost(self, page: Page) -> Cost:
        """Delegates to the wrapped backend (the estimate excludes retries).

        Deliberately optimistic: budgeting for the worst case would reserve
        ``attempts`` times the money for calls that usually succeed first time.
        Retries that do happen are charged to ``Cost.wasted_calls`` after the
        fact.

        :param page: the page that would be read
        :returns: the wrapped backend's estimate for a single attempt
        """
        return self.inner.estimate_cost(page)

    def read(self, page: Page) -> ReadResult:
        """Read with retries, recording any billed-but-failed attempts.

        :param page: the page to read
        :returns: the successful :class:`ReadResult`, with ``cost.wasted_calls``
            incremented per failed attempt and a warning naming the first few
            failures -- a provider that times out after generating tokens has
            already billed for them
        :raises Exception: whatever ``inner`` raised, once every attempt is spent
            or the error type is in ``give_up_on``
        """
        wasted = [0]
        failures: List[str] = []

        def note(attempt: int, exc: BaseException, delay: float) -> None:
            """Count a failed attempt and remember why it failed."""
            wasted[0] += 1
            failures.append("%s: %s" % (type(exc).__name__, exc))

        result = retry_call(lambda: self.inner.read(page), attempts=self.attempts,
                            base_delay=self.base_delay, retry_on=self.retry_on,
                            give_up_on=self.give_up_on, on_retry=note)
        if wasted[0]:
            result.cost = dataclasses.replace(
                result.cost, wasted_calls=result.cost.wasted_calls + wasted[0])
            result.warnings.append(
                "page %d succeeded after %d failed attempt(s); provider may have "
                "billed for them: %s" % (page.index, wasted[0], "; ".join(failures[:3])))
        return result


class EnsembleBackend(BaseBackend):
    """Read a page with two backends and keep both reads for cross-checking.

    Cross-read agreement is by a distance the strongest confidence signal
    available: two independent methods that produce the same string for the
    same region are very unlikely to have failed identically.  It is also the
    most annoying signal to implement, and therefore the one least likely to be
    built correctly three separate times under three separate deadlines --
    which is the argument for it living here.

    The primary backend's spans are returned (they carry the better geometry);
    the secondary's are attached to the result for :func:`Confidence.cross_read_agreement`.
    """

    def __init__(self, primary: TextBackend, secondary: TextBackend,
                 name: Optional[str] = None) -> None:
        """Pair two backends for cross-checked reading.

        :param primary: its spans are returned, so this should be the backend
            with the better geometry -- a classical OCR engine rather than a VLM,
            which reports no coordinates at all.
        :param secondary: read for agreement only; its spans are attached to the
            result rather than returned.  Pick something that fails
            *differently* -- pairing two Tesseract configurations agrees on its
            own mistakes and tells you nothing.
        :param name: overrides the generated ``"ensemble:a+b"`` registry name
        """
        BaseBackend.__init__(self, name=name or "ensemble:%s+%s"
                             % (primary.name, secondary.name))
        self.primary = primary
        self.secondary = secondary

    def is_available(self) -> bool:
        """True only when both backends are available."""
        return self.primary.is_available() and self.secondary.is_available()

    def supports(self, page: Page) -> bool:
        """True only when both backends support the page.

        :param page: the page in question
        :returns: ``False`` if either backend declines -- there is no agreement
            signal to be had from one read
        """
        return self.primary.supports(page) and self.secondary.supports(page)

    def estimate_cost(self, page: Page) -> Cost:
        """The sum of both backends' estimates -- an ensemble is not free.

        :param page: the page that would be read
        :returns: both estimates added together.  Worth reserving for the pages
            that need it: :class:`RuleRouter` can send only low-quality pages
            here and leave clean ones on a single cheap read.
        """
        return self.primary.estimate_cost(page) + self.secondary.estimate_cost(page)

    def read(self, page: Page) -> ReadResult:
        """Read with both backends and attach their agreement to every span.

        The agreement score lands in ``span.meta["cross_read_agreement"]``, where
        confidence fusion picks it up.

        :param page: the page to read; both backends see the same one
        :returns: a :class:`ReadResult` carrying the *primary's* spans, the summed
            cost, both sets of warnings, and both raw reads under
            ``raw["primary"]`` and ``raw["secondary"]`` -- which is how
            geometry is recovered for a VLM read by :func:`locate_value`
        """
        first = self.primary.read(page)
        second = self.secondary.read(page)
        agreement = cross_read_agreement(first.spans, second.spans)
        for span in first.spans:
            span.meta["cross_read_agreement"] = agreement
        result = ReadResult(
            spans=first.spans, cost=first.cost + second.cost,
            backend=self.name, warnings=first.warnings + second.warnings,
            raw={"primary": first, "secondary": second})
        result.warnings.append("cross-read agreement %.3f (%s vs %s)"
                               % (agreement, self.primary.name, self.secondary.name))
        return result


# -- routing -----------------------------------------------------------------

RouterFn = Callable[[Page], Optional[str]]


def default_router(page: Page) -> Optional[str]:
    """Choose a backend name for ``page``.

    The cost delta between "send every page to a vision model" and "send only
    the pages that need one" is large.  The accuracy delta on native-text pages
    runs the *other* way: a text layer read directly is exact, while a model
    reading a picture of it can hallucinate.  So both considerations point the
    same way and this function is worth writing down once, carefully.

    Order of preference:

    1. a trustworthy embedded text layer -- free and exact, never pay for it;
    2. a clean page, especially a ruled/tabular one -- a local OCR engine;
    3. anything degraded, handwritten or multi-script -- a vision model;
    4. whatever local engine is installed, if no vision model is configured.

    Returns ``None`` for blank pages, which the reader skips.

    Only ever names a backend that :meth:`BackendRegistry.available` reports as
    installed, so the same code path works on a machine with nothing but
    Tesseract and on one with every engine present.

    :param page: a page whose quality and layout have been measured.  Without
        measurement every reading is zero, so the "degraded" test never fires
        and clean-page routing is used throughout.
    :returns: a registered backend name such as ``"pymupdf"``, ``"paddle"`` or
        ``"vlm:claude"``; or ``None`` for a blank page, and also when no backend
        at all is installed
    """
    if page.kind is PageKind.BLANK or page.is_blank:
        return None
    if page.kind is PageKind.DIGITAL_NATIVE and page.spans:
        return "pymupdf"

    available = set(registry.available())
    local_order = ["paddle", "rapidocr", "doctr", "tesseract", "easyocr"]
    vision_order = ["vlm:claude", "vlm:openai", "vlm:gemini"]
    surya_ok = "surya" in available

    needs_vision = (page.quality.verdict is Verdict.UNREADABLE
                    or page.layout.has_handwriting
                    or page.quality.score < 0.45)
    if needs_vision:
        for candidate in vision_order:
            if candidate in available:
                return candidate
        if surya_ok:
            return "surya"  # best local option for hard pages

    if page.layout.is_tabular:
        for candidate in ["paddle", "rapidocr", "doctr", "tesseract"]:
            if candidate in available:
                return candidate

    for candidate in local_order:
        if candidate in available:
            return candidate
    if surya_ok:
        return "surya"
    for candidate in vision_order:
        if candidate in available:
            return candidate
    return None


class RuleRouter(object):
    """A router built from ``(predicate, backend_name)`` rules, first match wins.

    Rules are data, so a project can express "handwritten annexures go to the
    vision model, everything else to Paddle" without subclassing anything.
    """

    def __init__(self, rules: Optional[Sequence[Tuple[Callable[[Page], bool], str]]] = None,
                 fallback: Optional[str] = None) -> None:
        """Build a router from ordered rules.

        :param rules: ``(predicate, backend_name)`` pairs, evaluated in order
            with the first match winning.  The predicate takes a :class:`Page`,
            so any measured property is fair game::

                RuleRouter([(lambda p: p.layout.has_handwriting, "vlm:claude"),
                            (lambda p: p.layout.is_tabular, "paddle")],
                           fallback="tesseract")

        :param fallback: name returned when no rule matches.  ``None`` means the
            page is skipped entirely -- usually you want a cheap local engine
            here instead, or pass ``fallback=default_router(page)``-style logic
            as a final catch-all rule.
        """
        self.rules = list(rules or [])
        self.fallback = fallback

    def add(self, predicate: Callable[[Page], bool], backend_name: str) -> RuleRouter:
        """Append a rule.  Returns ``self``, so rules chain.

        Appends, so it is always lower priority than the rules already present.

        :param predicate: ``Callable[[Page], bool]``; one that raises is logged
            and skipped rather than failing the document
        :param backend_name: registered name to route to when it matches
        :returns: ``self``, so calls chain -- ``router.add(a, "x").add(b, "y")``
        """
        self.rules.append((predicate, backend_name))
        return self

    def __call__(self, page: Page) -> Optional[str]:
        """The first matching rule's backend, else the fallback.

        A predicate that raises is logged and skipped rather than failing the
        document: a broken rule should not cost you the other 200 pages.

        :param page: the page to route
        :returns: the first matching rule's backend name, else ``fallback``,
            which may be ``None`` to skip the page
        """
        for predicate, name in self.rules:
            try:
                if predicate(page):
                    return name
            except Exception as exc:
                logger.warning("router rule raised %s; skipping it", exc)
        return self.fallback


class BudgetRouter(object):
    """Wrap a router and downgrade to a cheaper backend once a budget is spent.

    Long bundles are where cost surprises happen: page 3 of a 300-page court
    filing looks like every other page, and nothing in a per-page decision knows
    that 297 more are coming.  This makes the ceiling explicit.
    """

    def __init__(self, inner: RouterFn, budget: float, cheap_backend: str = "tesseract",
                 currency: str = "USD") -> None:
        """Wrap ``inner`` with a spending ceiling.

        :param inner: the router whose choice is second-guessed -- typically the
            built-in :func:`default_router`, or a :class:`RuleRouter`
        :param budget: total spend allowed across the whole document, in
            ``currency``.  Meaningless until :func:`Pricing.set_pricing` is
            called -- an unpriced model estimates ``0.00`` and never triggers a
            downgrade.
        :param cheap_backend: registered name used once the budget would be
            exceeded.  ``"tesseract"`` needs no model download and no network;
            if it is not registered the page is skipped instead.
        :param currency: label for messages and reports only; no conversion is
            performed, so it must match the units of your registered prices
        """
        self.inner = inner
        self.budget = float(budget)
        self.cheap_backend = cheap_backend
        self.currency = currency
        self.spent = 0.0
        self.downgrades = 0

    def record(self, cost: Cost) -> None:
        """Add an actual cost to the running total.  Called by the reader.

        Actual, not estimated: the running total tracks what the provider really
        billed, so an estimate that was optimistic still shows up here.

        :param cost: the :class:`Cost` of a completed read
        """
        self.spent += cost.amount

    def remaining(self) -> float:
        """Budget left, never negative."""
        return max(0.0, self.budget - self.spent)

    def __call__(self, page: Page) -> Optional[str]:
        """The inner router's choice, downgraded when it would break the budget.

        Returns ``None`` if even the cheap backend is unavailable -- skipping a
        page is preferable to silently blowing through a ceiling.

        Downgrades are counted on ``self.downgrades`` and logged, so a run that
        quietly fell back to cheap OCR halfway through is visible afterwards
        rather than showing up as an unexplained accuracy drop.

        :param page: the page to route
        :returns: the inner router's choice when it fits the remaining budget,
            otherwise ``cheap_backend``; ``None`` when the inner router skipped
            the page or the cheap backend is not registered
        """
        name = self.inner(page)
        if name is None:
            return None
        try:
            backend = registry.get(name)
        except ConfigError:
            return name
        estimate = backend.estimate_cost(page)
        if estimate.amount and self.spent + estimate.amount > self.budget:
            self.downgrades += 1
            logger.warning("budget %.4f %s would be exceeded by page %d; "
                           "downgrading %s -> %s", self.budget, self.currency,
                           page.index, name, self.cheap_backend)
            return self.cheap_backend if self.cheap_backend in registry else None
        return name


def _register_default_backends() -> None:
    """Register the built-in backends as lazy factories."""
    registry.register("pymupdf", PyMuPDFTextLayer, replace=True)
    registry.register("tesseract", TesseractOCR, replace=True)
    registry.register("paddle", PaddleOCRBackend, replace=True)
    registry.register("rapidocr", RapidOCRBackend, replace=True)
    registry.register("easyocr", EasyOCRBackend, replace=True)
    registry.register("doctr", DocTRBackend, replace=True)
    registry.register("surya", SuryaBackend, replace=True)
    registry.register("vlm:claude", AnthropicVisionBackend, replace=True)
    registry.register("vlm:openai", OpenAIVisionBackend, replace=True)
    registry.register("vlm:gemini", GeminiVisionBackend, replace=True)


_register_default_backends()


def read_page(page: Page, backend: Optional[Union[str, TextBackend]] = None,
              router: Optional[RouterFn] = default_router,
              replace_spans: bool = True) -> ReadResult:
    """Read one page with an explicit backend, or one chosen by ``router``.

    The single-page entry point, useful for debugging a page that came out wrong
    without running the whole document.

    :param page: the page to read.  Its ``history`` gains a ``"read"`` entry
        either way, including when no backend would take it.
    :param backend: force a specific reader, as a registered name
        (``"pymupdf"``, ``"tesseract"``, ``"paddle"``, ``"vlm:claude"``) or an
        instance of :class:`TextBackend`.  Overrides ``router`` when both given.
    :param router: chooses the backend when ``backend`` is ``None``.  Defaults
        to :func:`default_router`; ``None`` with no ``backend`` reads nothing.
    :param replace_spans: overwrite ``page.spans`` with the result.  Set
        ``False`` to read without disturbing the page -- which is how you compare
        two backends on the same page, or keep a native text layer while reading
        it again with OCR.
    :returns: a :class:`ReadResult`.  When no backend was selected or the chosen
        one declined the page, the result is empty with an explanatory warning
        rather than an exception -- one unreadable page must not cost the
        document.

    Compare two engines on a page that read badly::

        a = read_page(page, backend="tesseract", replace_spans=False)
        b = read_page(page, backend="paddle", replace_spans=False)
        print(a.text(), "|", b.text())
    """
    # Resolved into its own name rather than reassigning the parameter: the
    # parameter admits a name or an instance, the local is always an instance.
    chosen: TextBackend
    if backend is None:
        name = router(page) if router else None
        if name is None:
            return ReadResult(backend="none",
                              warnings=["no backend selected for page %d" % page.index])
        chosen = registry.get(name) if isinstance(name, str) else name
    elif isinstance(backend, str):
        chosen = registry.get(backend)
    else:
        chosen = backend

    if not chosen.supports(page):
        return ReadResult(backend=chosen.name,
                          warnings=["%s cannot read page %d" % (chosen.name, page.index)])
    result = chosen.read(page)
    if replace_spans:
        page.spans = result.spans
    page.record("read", {"backend": result.backend, "spans": len(result.spans)},
                result.latency_ms)
    return result


def read(doc: Document, backend: Optional[Union[str, TextBackend]] = None,
         router: Optional[RouterFn] = default_router, max_workers: int = 0,
         in_place: bool = False, budget: Optional[float] = None,
         on_page: Optional[Callable[[Page, ReadResult], None]] = None) -> Document:
    """Read every page of ``doc``, choosing a backend per page.

    :param doc: a document, normally already preprocessed.  Reading an
        unmeasured document works, but :func:`default_router` then sees a
        perfect page everywhere and routes accordingly.
    :param backend: force one reader for all pages, as a registered name or an
        instance of :class:`TextBackend`.  Overrides ``router``.
    :param router: per-page backend chooser, ignored when ``backend`` is set.
        Defaults to :func:`default_router`; see also :class:`RuleRouter` and
        the ceiling-aware :class:`BudgetRouter`.
    :param max_workers: threads for concurrent page reads.  ``0`` runs serially.
        Safe to raise -- backends hold their own locks -- and the win is large
        for network-bound vision backends.  Local engines that already use every
        core benefit far less.
    :param in_place: mutate ``doc`` rather than working on a copy
    :param budget: hard ceiling in USD across the document, checked after each
        page.  Distinct from :class:`BudgetRouter`, which downgrades to stay
        under; this one stops.  Requires :func:`Pricing.set_pricing` to mean
        anything.
    :param on_page: called as ``on_page(page, result)`` after each page, for
        progress reporting.  Runs outside the internal lock, so it may be slow,
        but it is called from worker threads when ``max_workers`` is set.
    :returns: the read :class:`Document`.  Costs, warnings and the per-page
        backend choices accumulate onto ``meta["read_cost"]`` and
        ``meta["read_backends"]`` -- the latter being how you check the router
        did what you expected.
    :raises BudgetExceeded: the running total passed ``budget``

    A page whose read raises is logged, recorded as a document warning, and
    skipped; the other pages still get read.
    """
    target = doc if in_place else doc.copy()
    total = Cost.zero()
    lock = threading.Lock()
    choices: Dict[int, str] = {}

    def handle(page: Page) -> None:
        """Read one page, folding its cost and warnings into the document."""
        try:
            result = read_page(page, backend=backend, router=router)
        except Exception as exc:
            logger.exception("read failed on page %d", page.index)
            with lock:
                target.warn("page %d: read failed: %s" % (page.index, exc))
            return
        with lock:
            choices[page.index] = result.backend
            if result.cost.calls or result.cost.amount:
                merged = total + result.cost
                total.amount = merged.amount
                total.input_tokens = merged.input_tokens
                total.output_tokens = merged.output_tokens
                total.calls = merged.calls
                total.wasted_calls = merged.wasted_calls
            for warning in result.warnings:
                target.warn(warning)
            if budget is not None and total.amount > budget:
                raise BudgetExceeded(
                    "spent %.4f USD reading %s, over the %.4f budget"
                    % (total.amount, target.source_uri, budget))
        if on_page is not None:
            on_page(page, result)

    _map_maybe_parallel(handle, target.pages, max_workers, "read")
    target.meta["read_cost"] = total.to_dict()
    target.meta["read_backends"] = choices
    return target


# =============================================================================
# 9. Normalisation -- scripts, digits, dates, amounts
# =============================================================================
#
# Domain-independent, but not culture-independent: date order, digit grouping
# and numeral systems are properties of the document's origin, not of what it
# means.  These live here so that "1,23,456.78 is 123456.78, not 1.23" is
# decided once.


#: Unicode block starts for the scripts these documents actually contain.
_SCRIPT_RANGES = [
    ("Latn", 0x0041, 0x024F),
    ("Grek", 0x0370, 0x03FF),
    ("Cyrl", 0x0400, 0x04FF),
    ("Arab", 0x0600, 0x06FF),
    ("Deva", 0x0900, 0x097F),
    ("Beng", 0x0980, 0x09FF),
    ("Guru", 0x0A00, 0x0A7F),
    ("Gujr", 0x0A80, 0x0AFF),
    ("Orya", 0x0B00, 0x0B7F),
    ("Taml", 0x0B80, 0x0BFF),
    ("Telu", 0x0C00, 0x0C7F),
    ("Knda", 0x0C80, 0x0CFF),
    ("Mlym", 0x0D00, 0x0D7F),
    ("Sinh", 0x0D80, 0x0DFF),
    ("Thai", 0x0E00, 0x0E7F),
    ("Hans", 0x4E00, 0x9FFF),
    ("Hang", 0xAC00, 0xD7AF),
    ("Hira", 0x3040, 0x309F),
    ("Kana", 0x30A0, 0x30FF),
]


class Text(object):
    """Script detection, text normalisation and value parsing.

    These run *after* reading and before extraction, on the theory that an
    OCR engine's mistakes are systematic: full-width digits, Devanagari
    numerals, ``O`` for ``0`` in a numeric field.  Parsing is locale-aware
    by necessity -- ``1,23,456.78`` and ``1.234,56`` are both real.
    """

    @staticmethod
    def detect_script(text: str, minimum: int = 1) -> Optional[str]:
        """Dominant ISO 15924 script code of ``text``, or ``None``.

        Per-span rather than per-document, because a hospital letterhead in
        Devanagari over a table of Latin numerals is one page with two scripts, and
        routing a recogniser at the page level would get one of them wrong.

        :param text: the text to inspect; digits and whitespace are ignored, as
            they carry no script information
        :param minimum: least character count for a script to be reported.
            ``1`` reports whatever dominates, which is right for a single span;
            raise it to ignore a stray character in a long string.
        :returns: an ISO 15924 code -- ``"Latn"``, ``"Deva"``, ``"Arab"``,
            ``"Beng"``, ``"Taml"``, ``"Hani"`` and the rest -- or ``None`` for
            empty text or text that is all digits and punctuation
        """
        if not text:
            return None
        counts: Dict[str, int] = {}
        for ch in text:
            code = ord(ch)
            if code < 0x0041 or ch.isspace() or ch.isdigit():
                continue
            for name, lo, hi in _SCRIPT_RANGES:
                if lo <= code <= hi:
                    counts[name] = counts.get(name, 0) + 1
                    break
        if not counts:
            return None
        best = max(counts.items(), key=lambda kv: kv[1])
        return best[0] if best[1] >= minimum else None

    @staticmethod
    def normalize_digits(text: str) -> str:
        """Convert every Unicode decimal digit to ASCII ``0-9``.

        A Devanagari ``४८२५०`` and an ASCII ``48250`` are the same amount, and every
        validator downstream should only ever have to handle one of them.

        :param text: any text; non-digit characters pass through untouched
        :returns: the text with every Unicode decimal digit -- Devanagari,
            Arabic-Indic, Bengali, full-width and the rest -- mapped to ASCII
        """
        return text.translate(_DIGIT_MAP) if text else text

    @staticmethod
    def normalize_unicode(text: str,
                          form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC") -> str:
        """Normalise Unicode form, collapsing ligatures and compatibility variants.

        The four forms are spelled out in the annotation because they are the
        only values :mod:`unicodedata` accepts; anything else is a ``ValueError``
        from the standard library rather than a docpipe error.

        :param text: the text to normalise
        :param form: ``"NFKC"`` (default), ``"NFC"``, ``"NFD"``, or ``"NFKD"``.
            The compatibility forms (``NFKC``/``NFKD``) are what fold ``ﬁ`` to
            ``fi`` and full-width to ASCII, which is what OCR output needs.  Use
            ``"NFC"`` when the exact original characters must survive.
        :returns: the normalised text
        """
        return unicodedata.normalize(form, text) if text else text

    @staticmethod
    def normalize_whitespace(text: str, collapse_newlines: bool = False) -> str:
        """Collapse runs of whitespace, preserving line structure by default.

        :param text: the text to tidy
        :param collapse_newlines: also fold line breaks into spaces.  Leave it
            ``False`` for page text, where line structure is what
            label-and-value parsing depends on; set it ``True`` for a single
            field value that OCR split across lines.
        :returns: the text with whitespace runs collapsed, leading and trailing
            whitespace stripped, and runs of three or more blank lines reduced
            to one
        """
        if not text:
            return text
        out = _WHITESPACE_RE.sub(" ", text)
        if collapse_newlines:
            out = re.sub(r"\s*\n\s*", " ", out)
        else:
            out = re.sub(r"[ \t]*\n[ \t]*", "\n", out)
            out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    @staticmethod
    def normalize_text(text: str) -> str:
        """The standard cleanup: Unicode form, ASCII digits, tidy whitespace.

        Applies :func:`Text.normalize_unicode`, then
        :func:`Text.normalize_digits`, then :func:`Text.normalize_whitespace`, in
        that order -- digit mapping has to follow the compatibility fold, or
        full-width digits are missed.

        :param text: the text to clean; ``None`` is treated as empty
        :returns: the normalised text.  Safe for free text: nothing here guesses
            at characters, unlike :func:`Text.fix_ocr_confusions`.
        """
        return normalize_whitespace(normalize_digits(normalize_unicode(text or "")))

    @staticmethod
    def fix_ocr_confusions(text: str, expect: str = "numeric") -> str:
        """Repair classic OCR character confusions for a *known-type* field.

        Never apply this to free text.  ``expect="numeric"`` on a field declared as
        an amount is safe and recovers a lot of otherwise-lost values; the same
        substitution on a patient's name turns "Sonia" into "50nia".

        :param text: the field value to repair
        :param expect: ``"numeric"`` maps letters onto digits (``O`` to ``0``,
            ``l`` and ``I`` to ``1``, ``S`` to ``5``, ``B`` to ``8``);
            ``"alpha"`` maps the other way for a known-alphabetic field.  Any
            other value leaves the text untouched rather than guessing.
        :returns: the repaired text
        """
        if not text:
            return text
        table = _NUMERIC_CONFUSIONS if expect == "numeric" else _ALPHA_CONFUSIONS
        if expect not in ("numeric", "alpha"):
            raise ConfigError("expect must be 'numeric' or 'alpha'")
        return "".join(table.get(ch, ch) for ch in text)

    @staticmethod
    def parse_amount(text: Optional[str], decimal_separator: str = "auto",
                     allow_negative: bool = True) -> Optional[decimal.Decimal]:
        """Parse a monetary amount, handling Indian and Western digit grouping.

        Indian grouping (``1,23,456.78`` -- two digits per group above the
        hundreds) breaks any parser that assumes groups of three, and it is the
        default on every document this library was written for.  Grouping is
        therefore *inferred* rather than assumed:

        * ``1,234.56`` -> 1234.56    (comma groups, dot decimal)
        * ``1,23,456.78`` -> 123456.78  (Indian grouping)
        * ``1.234,56`` -> 1234.56    (European convention, detected by position)
        * ``(1,234)`` -> -1234       (accounting negative)
        * ``1,234 CR`` / ``1,234 DR`` -> credit/debit sign

        Returns a :class:`decimal.Decimal` -- never a float, because money and
        binary floating point should not meet.  Returns ``None`` when no number is
        present.

        :param text: the text to parse, currency symbols and labels included.
            ``None`` returns ``None``.
        :param decimal_separator: ``"auto"`` (default) infers the convention from
            separator positions, which is what handles Indian grouping.  Force
            it with ``"dot"`` or ``"comma"`` when a batch is known to be one
            convention and short values like ``"1,50"`` are genuinely ambiguous.
        :param allow_negative: honour accounting negatives -- a leading minus,
            parentheses, and a ``DR`` marker.  ``False`` returns the magnitude,
            for a field that cannot meaningfully be negative.
        :returns: the amount as a :class:`decimal.Decimal`, or ``None`` when the
            text holds no number
        """
        if text is None:
            return None
        # Strip typographic group separators *before* NFKC, which would otherwise
        # fold them into ordinary spaces and make them indistinguishable from the
        # space between two unrelated numbers in a table row.
        raw = re.sub("[%s]" % _GROUP_SPACES, "", str(text))
        raw = normalize_digits(normalize_unicode(raw)).strip()
        if not raw:
            return None

        negative = False
        if allow_negative and raw.startswith("(") and raw.endswith(")"):
            negative = True
            raw = raw[1:-1]
        upper = raw.upper()
        if re.search(r"\bCR\b", upper):
            negative = False
        elif re.search(r"\bDR\b", upper):
            negative = True

        match = _AMOUNT_RE.search(raw)
        if not match:
            return None
        body = match.group(0)
        if body.startswith("-"):
            negative = True
        body = body.lstrip("+-")
        body = re.sub(r"['%s]" % _GROUP_SPACES, "", body)

        if decimal_separator == "comma":
            body = body.replace(".", "").replace(",", ".")
        elif decimal_separator == "dot":
            body = body.replace(",", "")
        else:
            has_dot = "." in body
            has_comma = "," in body
            if has_dot and has_comma:
                # Whichever separator comes last is the decimal one.
                body = (body.replace(",", "") if body.rfind(".") > body.rfind(",")
                        else body.replace(".", "").replace(",", "."))
            elif has_comma:
                tail = body.split(",")[-1]
                # A trailing group of exactly 2 digits with a single comma is
                # ambiguous ("1,50" is 1.50 in Europe, 150 in India); grouping wins,
                # because these documents group and do not use comma decimals.
                body = body.replace(",", "") if len(tail) in (2, 3) else body.replace(",", ".")
            # A lone dot is already the decimal separator.

        try:
            value = decimal.Decimal(body)
        except (decimal.InvalidOperation, ValueError):
            return None
        return -value if negative else value

    @staticmethod
    def detect_currency(text: str) -> Optional[str]:
        """ISO currency code implied by ``text``, or ``None``.

        :param text: text that may carry a symbol (``₹``, ``$``, ``€``) or a code
            (``INR``, ``USD``)
        :returns: the ISO 4217 code, or ``None`` when nothing identifies one.
            Ambiguous symbols resolve to the most likely code for these document
            classes, so treat the answer as a hint -- pass the currency in
            ``context`` when you know it.
        """
        if not text:
            return None
        lowered = normalize_unicode(text).lower()
        for token, code in _CURRENCY_SYMBOLS.items():
            if token in lowered:
                return code
        return None

    @staticmethod
    def parse_date(text: Optional[str], dayfirst: bool = True, min_year: int = 1900,
                   max_year: int = 2100) -> Optional[_dt.date]:
        """Parse a date from noisy OCR text.

        ``dayfirst=True`` by default because DD-MM-YYYY is the convention in the
        document classes this library targets, and an ambiguous ``03/04/2024``
        silently read as March 4th is the kind of error that reaches a claims
        reviewer looking entirely plausible.  Unambiguous inputs (a day above 12, a
        named month, an ISO date) always win over the flag.

        Returns ``None`` rather than guessing when the text holds no date.

        :param text: the text to parse; ``None`` returns ``None``
        :param dayfirst: how to read a genuinely ambiguous ``03/04/2024``.
            ``True`` (default) reads it as 3 April.  Only consulted when both
            numbers are 12 or below -- a day above 12, a named month, or an ISO
            date decides for itself.
        :param min_year: reject dates before this year.  Guards against an OCR
            misread producing a plausible but absurd date.
        :param max_year: reject dates after this year
        :returns: a :class:`datetime.date`, or ``None`` when no date is present
            or every candidate falls outside the year range
        """
        if not text:
            return None
        cleaned = normalize_digits(normalize_unicode(str(text)))
        for pattern, order in _DATE_PATTERNS:
            match = pattern.search(cleaned)
            if not match:
                continue
            groups = match.groups()
            try:
                if order == "ymd":
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                elif order == "ddmmyyyy":
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                elif order == "dMy":
                    month = _MONTHS.get(groups[1].lower()[:9]) or _MONTHS.get(groups[1].lower()[:3])
                    if not month:
                        continue
                    day, year = int(groups[0]), _expand_year(int(groups[2]))
                elif order == "Mdy":
                    month = _MONTHS.get(groups[0].lower()[:9]) or _MONTHS.get(groups[0].lower()[:3])
                    if not month:
                        continue
                    day, year = int(groups[1]), _expand_year(int(groups[2]))
                else:  # dmy_or_mdy
                    first, second = int(groups[0]), int(groups[1])
                    year = _expand_year(int(groups[2]))
                    if first > 12 >= second:
                        day, month = first, second
                    elif second > 12 >= first:
                        day, month = second, first
                    else:
                        day, month = (first, second) if dayfirst else (second, first)
                if not (min_year <= year <= max_year):
                    continue
                return _dt.date(year, month, day)
            except (ValueError, TypeError):
                continue
        return None

    @staticmethod
    def parse_bool(text: Optional[str]) -> Optional[bool]:
        """Parse a checkbox-ish value from a form.

        :param text: the cell's text.  ``None`` returns ``None``; so does
            anything unrecognised, which keeps "the box was unreadable" distinct
            from "the box was unticked".
        :returns: ``True`` for ``"yes"``, ``"y"``, ``"true"``, ``"1"``, ``"x"``,
            ``"✓"``, ``"✔"``, ``"checked"``; ``False`` for ``"no"``, ``"n"``,
            ``"false"``, ``"0"``, ``""``, ``"-"``, ``"unchecked"``; ``None``
            otherwise.  Matching is case-insensitive.
        """
        if text is None:
            return None
        token = str(text).strip().lower()
        if token in ("yes", "y", "true", "1", "x", "✓", "✔", "checked"):
            return True
        if token in ("no", "n", "false", "0", "", "-", "unchecked"):
            return False
        return None

    @staticmethod
    def normalize_spans(spans: Sequence[TextSpan], digits: bool = True,
                        unicode_form: Optional[Literal["NFC", "NFD", "NFKC", "NFKD"]]
                        = "NFKC") -> List[TextSpan]:
        """Return normalised copies of ``spans``, tagging each with its script.

        Copies rather than mutating, so the original read stays available for
        comparison -- which matters when a normalisation is suspected of losing
        something.

        :param spans: the spans to normalise
        :param digits: map every Unicode digit to ASCII
        :param unicode_form: Unicode normalisation form, or ``None`` to skip it.
            ``"NFKC"`` is the default and the right choice for OCR output.
        :returns: new spans with normalised text and ``script`` filled in where
            it was missing.  Geometry, confidence and meta carry through
            untouched.
        """
        out = []
        for span in spans:
            text = span.text
            if unicode_form:
                text = normalize_unicode(text, unicode_form)
            if digits:
                text = normalize_digits(text)
            out.append(dataclasses.replace(span, text=text,
                                           script=span.script or detect_script(text)))
        return out

    @staticmethod
    def normalize_document(doc: Document, in_place: bool = False, **kwargs: Any) -> Document:
        """Normalise every span in a document.

        Run automatically at the end of :func:`Pipeline.run_document` unless
        ``normalize=False``.

        :param doc: the document to normalise
        :param in_place: mutate ``doc`` rather than working on a copy
        :param kwargs: forwarded to :func:`Text.normalize_spans` -- ``digits``,
            ``unicode_form``
        :returns: the normalised document
        """
        target = doc if in_place else doc.copy()
        for page in target.pages:
            page.spans = normalize_spans(page.spans, **kwargs)
        return target


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Text from the inside.
# ``Text.x`` is the documented path; ``x`` is the one that already works.
detect_script        = Text.detect_script
normalize_digits     = Text.normalize_digits
normalize_unicode    = Text.normalize_unicode
normalize_whitespace = Text.normalize_whitespace
normalize_text       = Text.normalize_text
fix_ocr_confusions   = Text.fix_ocr_confusions
parse_amount         = Text.parse_amount
detect_currency      = Text.detect_currency
parse_date           = Text.parse_date
parse_bool           = Text.parse_bool
normalize_spans      = Text.normalize_spans
normalize_document   = Text.normalize_document


#: Non-ASCII decimal digits seen on Indian and Arabic documents.
_DIGIT_BLOCKS = [
    0x0660,  # Arabic-Indic
    0x06F0,  # Extended Arabic-Indic (Persian/Urdu)
    0x0966,  # Devanagari
    0x09E6,  # Bengali
    0x0A66,  # Gurmukhi
    0x0AE6,  # Gujarati
    0x0B66,  # Oriya
    0x0BE6,  # Tamil
    0x0C66,  # Telugu
    0x0CE6,  # Kannada
    0x0D66,  # Malayalam
    0x0E50,  # Thai
    0xFF10,  # Fullwidth
]

_DIGIT_MAP: Dict[int, str] = dict(
    (base + offset, str(offset))
    for base in _DIGIT_BLOCKS for offset in range(10))


_WHITESPACE_RE = re.compile(r"[ \t  -​]+")


#: Character confusions that classical OCR makes constantly.  Applied only
#: within a field whose expected type is known, never globally -- "SO" is a
#: word and "S0" is not, but only the caller knows which was expected.
_NUMERIC_CONFUSIONS = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "i": "1", "|": "1", "!": "1",
    "S": "5", "s": "5", "Z": "2", "z": "2",
    "B": "8", "b": "6", "G": "6", "g": "9", "T": "7",
}
_ALPHA_CONFUSIONS = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


_CURRENCY_SYMBOLS = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "₨": "INR",
    "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP", "¥": "JPY", "jpy": "JPY",
}

#: Typographic group separators that are unambiguously *inside* a number.
#: A plain ASCII space is deliberately excluded: "12 500" is as likely to be
#: two numbers in a table row as one grouped amount, and gluing them together
#: silently invents a value.
_GROUP_SPACES = "\u2009\u202f\u00a0"

#: Matches a number with any mixture of grouping and decimal separators, so
#: that "1.234,56" is captured whole rather than truncated at the first dot.
_AMOUNT_RE = re.compile(r"[-+]?\d[\d.,'%s]*\d|[-+]?\d" % _GROUP_SPACES)


_MONTHS: Dict[str, int] = dict(
    (name, number)
    for number, names in enumerate([
        ("jan", "january"), ("feb", "february"), ("mar", "march"),
        ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
        ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
        ("nov", "november"), ("dec", "december")], start=1)
    for name in names)

_DATE_PATTERNS = [
    # (regex, field order)
    (re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b"), "dmy_or_mdy"),
    (re.compile(r"\b(\d{1,2})[\s-]*([A-Za-z]{3,9})[\s,-]*(\d{2,4})\b"), "dMy"),
    (re.compile(r"\b([A-Za-z]{3,9})[\s-]*(\d{1,2})[\s,-]*(\d{2,4})\b"), "Mdy"),
    (re.compile(r"\b(\d{2})(\d{2})(\d{4})\b"), "ddmmyyyy"),
]


def _expand_year(value: int) -> int:
    """Expand a two-digit year.  Documents in scope are recent, not Victorian."""
    if value >= 100:
        return value
    current = _dt.date.today().year
    century = current - current % 100
    candidate = century + value
    # Allow a few years into the future (policy expiry dates) but not decades.
    return candidate - 100 if candidate > current + 5 else candidate


#: Text that *looks* like a date: two separators, or a named month.
_DATE_SHAPED_RE = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\d{1,2}[\s-]*[A-Za-z]{3,9}[\s,-]*\d{2,4}"
    r"|[A-Za-z]{3,9}[\s-]*\d{1,2}[\s,-]*\d{2,4}")


# =============================================================================
# 10. Cross-cutting -- caching, tracing, budgets
# =============================================================================


class DiskCache(object):
    """A tiny content-addressed JSON cache on the filesystem.

    Deliberately not sqlite: it must be safe under concurrent readers and
    writers across processes, and "write a temp file then rename" gives that
    for free on every POSIX filesystem, with no locking and no schema to
    migrate.
    """

    def __init__(self, directory: Optional[str] = None, enabled: bool = True,
                 max_entries: int = 0) -> None:
        """Open (and create) the cache directory.

        :param directory: defaults to ``$DOCPIPE_CACHE_DIR`` or a temp directory
        :param enabled: False makes every read a miss and every write a no-op
        :param max_entries: reserved; entries are not evicted yet

        An unusable directory is not fatal -- the cache degrades to memory-only
        and logs a warning, because a read-only filesystem should not stop a run.
        """
        self.directory = directory or os.path.join(
            os.environ.get("DOCPIPE_CACHE_DIR")
            or os.path.join(tempfile.gettempdir(), "docpipe-cache"))
        self.enabled = enabled
        self.max_entries = max_entries
        self._memory: Dict[str, Any] = {}
        if enabled:
            try:
                os.makedirs(self.directory, exist_ok=True)
            except OSError as exc:
                logger.warning("cache directory %s unusable (%s); caching in memory only",
                               self.directory, exc)
                self.directory = ""

    def _path(self, key: str) -> str:
        """Filesystem path holding ``key``."""
        return os.path.join(self.directory, "%s.json" % key)

    def get(self, key: str) -> Optional[Any]:
        """The cached value, or ``None`` when absent, disabled or unreadable.

        A corrupt or truncated entry counts as a miss: a cache is an
        optimisation, and it must never be able to fail a run.

        :param key: the cache key, normally a content hash produced
            by :func:`Util.stable_hash`
        :returns: the stored value, or ``None`` when the key is absent, the
            cache is disabled, or the entry could not be read.  All four cases
            look the same to a caller by design -- there is nothing useful to do
            differently about a corrupt entry.
        """
        if not self.enabled:
            return None
        if key in self._memory:
            return self._memory[key]
        if not self.directory:
            return None
        try:
            with open(self._path(key), "r", encoding="utf-8") as fh:
                value = json.load(fh)
        except (IOError, OSError, ValueError):
            return None
        self._memory[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``.

        Written to a temp file and renamed, which is atomic on POSIX -- so a
        concurrent reader sees either the old entry or the new one, never half.

        :param key: the cache key
        :param value: any JSON-serialisable value.  A value that will not
            serialise is logged and dropped rather than raised: failing a read
            because its result would not cache is the wrong trade.
        """
        if not self.enabled:
            return
        self._memory[key] = value
        if not self.directory:
            return
        try:
            handle, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump(_as_jsonable(value), fh)
            os.replace(tmp, self._path(key))
        except (IOError, OSError, TypeError, ValueError) as exc:
            logger.debug("cache write failed for %s: %s", key, exc)

    def __contains__(self, key: str) -> bool:
        """``key in cache`` -- true when a value is stored for it.

        Implemented as a full :meth:`get`, so it costs a read.  When you intend
        to fetch the value anyway, call :meth:`get` once and test for ``None``.

        :param key: the cache key
        :returns: whether a readable value is stored
        """
        return self.get(key) is not None

    def clear(self) -> None:
        """Drop every entry, in memory and on disk."""
        self._memory.clear()
        if self.directory and os.path.isdir(self.directory):
            for name in os.listdir(self.directory):
                if name.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.directory, name))
                    except OSError:
                        pass


@dataclass
class TraceSpan(object):
    """One timed step in a pipeline run."""

    name: str
    start: float
    end: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["TraceSpan"] = field(default_factory=list)

    @property
    def ms(self) -> float:
        """Duration in milliseconds -- so far, if the span is still open."""
        return ((self.end or time.time()) - self.start) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict, including nested children."""
        return {"name": self.name, "ms": round(self.ms, 2),
                "attributes": _as_jsonable(self.attributes),
                "children": [c.to_dict() for c in self.children]}


class Tracer(object):
    """Nested timing spans, so "why did this document take 40 seconds" has an answer.

    Deliberately minimal and dependency-free.  Emit to OpenTelemetry from the
    consuming service if you have it; this exists so the library is debuggable
    without one.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Start a trace.  ``enabled=False`` makes every span a no-op.

        :param enabled: ``False`` makes :meth:`span` return null context
            managers, so instrumentation can stay in the code with no
            measurable cost when tracing is off
        """
        self.enabled = enabled
        self.root = TraceSpan("root", time.time())
        self._stack = [self.root]
        self._lock = threading.Lock()

    class _Span(object):
        """Context manager that closes its span and pops the tracer's stack."""
        def __init__(self, tracer: Tracer, span: TraceSpan) -> None:
            """Bind a span to the tracer that will close it.

            :param tracer: the tracer whose stack this span was pushed onto
            :param span: the open :class:`TraceSpan` to close on exit
            """
            self.tracer = tracer
            self.span = span

        def __enter__(self) -> TraceSpan:
            """Give the block the :class:`TraceSpan` itself, for attributes."""
            return self.span

        def __exit__(self, *exc: Any) -> Literal[False]:
            """Close the span and pop it.  Never swallows an exception."""
            self.span.end = time.time()
            with self.tracer._lock:
                if len(self.tracer._stack) > 1:
                    self.tracer._stack.pop()
            return False

    def span(self, name: str, **attributes: Any) -> Any:
        """``with tracer.span("read", pages=12): ...`` -- open a nested span.

        Returns a no-op context manager when tracing is disabled, so callers
        never need to branch.

        :param name: the span's label, e.g. ``"read"`` or ``"extract"``
        :param attributes: arbitrary key-value pairs recorded on the span, e.g.
            ``pages=12`` or ``backend="paddle"``.  They survive
            into :meth:`TraceSpan.to_dict`, so this is where to put whatever
            would make a slow trace explicable afterwards.
        :returns: a context manager yielding the :class:`TraceSpan`, so the block
            can add attributes it only learns partway through
        """
        if not self.enabled:
            return _NullContext()
        span = TraceSpan(name, time.time(), attributes=dict(attributes))
        with self._lock:
            self._stack[-1].children.append(span)
            self._stack.append(span)
        return Tracer._Span(self, span)

    def finish(self) -> TraceSpan:
        """Close the root span and return it."""
        self.root.end = time.time()
        return self.root

    def to_dict(self) -> Dict[str, Any]:
        """The finished trace tree as a JSON-ready dict."""
        return self.finish().to_dict()


class _NullContext(object):
    """A context manager that does nothing -- used when tracing is off."""
    def __enter__(self) -> None:
        """Yield nothing."""
        return None

    def __exit__(self, *exc: Any) -> Literal[False]:
        """Do nothing.  Never swallows an exception."""
        return False


class CostTracker(object):
    """Accumulate spend, optionally enforcing a hard ceiling."""

    def __init__(self, budget: Optional[float] = None, currency: str = "USD") -> None:
        """Start at zero.

        :param budget: hard ceiling; exceeding it raises :class:`BudgetExceeded`
        :param currency: the currency every added cost must be in
        """
        self.budget = budget
        self.total = Cost(currency=currency)
        self._lock = threading.Lock()
        self.by_backend: Dict[str, Cost] = {}

    def add(self, cost: Cost, backend: str = "") -> Cost:
        """Add ``cost`` to the total and return the new total.

        Raises :class:`BudgetExceeded` once the budget is passed.  The cost is
        recorded before the check, so the reported total is what was actually
        spent rather than the last figure under the ceiling.

        :param cost: the :class:`Cost` to accumulate
        :param backend: attributes the spend, for the per-backend breakdown.
            Empty means "unattributed", which still counts towards the total.
        :returns: the new running total
        :raises BudgetExceeded: the total passed the budget.  The spend is
            already recorded when this raises, so ``self.total`` afterwards is
            the true figure.
        """
        with self._lock:
            self.total = self.total + cost
            if backend:
                self.by_backend[backend] = self.by_backend.get(backend, Cost()) + cost
            if self.budget is not None and self.total.amount > self.budget:
                raise BudgetExceeded("spent %.4f %s, budget was %.4f"
                                     % (self.total.amount, self.total.currency, self.budget))
            return self.total

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict: total, per-backend breakdown, and the budget."""
        return {"total": self.total.to_dict(),
                "by_backend": dict((k, v.to_dict()) for k, v in self.by_backend.items()),
                "budget": self.budget}


# =============================================================================
# 11. Confidence -- fusion and calibration
# =============================================================================
#
# A model's self-reported confidence is not a probability of correctness.  It
# correlates with fluency.  A clean, confidently-formatted hallucination scores
# high; a correct reading of a smudged digit scores low.  Shipping that number
# to a claims reviewer labelled "confidence" is worse than shipping no number,
# because it will be trusted.
#
# A defensible score fuses signals chosen to be as uncorrelated as possible,
# and is then *calibrated* against a labelled set.  Confidence without a
# calibration set is decoration -- which is why Layer 5 exists.


class Confidence(object):
    """Confidence fusion and the metrics that check it.

    Fusion combines *independent* signals -- page quality, backend
    confidence, cross-read agreement, logprob, validation, format match --
    in log-odds space.  A model's self-reported confidence is one input
    among several and never the answer: it measures fluency, not
    correctness.  Calibrators live alongside as classes.
    """

    @staticmethod
    def logit(p: float, eps: float = 1e-6) -> float:
        """Log-odds of ``p``, clamped away from the infinities.

        :param p: a probability in ``0.0..1.0``
        :param eps: clamp distance from each end, so ``0.0`` and ``1.0`` map to
            large finite values rather than infinities.  ``1e-6`` bounds the
            result to about +/-13.8.
        :returns: ``log(p / (1 - p))``
        """
        p = clamp(p, eps, 1.0 - eps)
        return math.log(p / (1.0 - p))

    @staticmethod
    def sigmoid(z: float) -> float:
        """Inverse of :func:`Confidence.logit`, numerically safe at both tails.

        Branches on the sign so neither tail overflows ``exp``, which the naive
        form does for ``z`` beyond about -700.

        :param z: a log-odds value, any real number
        :returns: the probability in ``0.0..1.0``
        """
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    @staticmethod
    def fuse_confidence(signals: Mapping[str, Optional[float]],
                        weights: Optional[Mapping[str, float]] = None,
                        prior: float = 0.5) -> float:
        """Fuse independent evidence into one calibrated-shaped score in ``[0, 1]``.

        Fusion happens in log-odds space (a log-linear opinion pool), not as a
        weighted average of probabilities.  That matters: averaging lets one
        confident-but-wrong signal of 0.99 drag a field up, whereas in log-odds a
        single strong disagreement pulls the result down hard, which is the
        behaviour you want when a validator says the line items do not sum to the
        total.

        Signals set to ``None`` are *absent*, not zero, and are dropped with their
        weight renormalised away.  Encoding "the backend reported no confidence" as
        0.0 would quietly punish every engine that declines to guess.

        Each signal is clamped to ``[0.02, 0.98]`` before its log-odds are taken,
        so no individual signal can dominate the pool outright -- see
        :data:`_SIGNAL_CLAMP`.

        Returns ``prior`` when no signal is available.

        :param signals: named signal values in ``0.0..1.0``, or ``None`` for
            "not measured".  The names the default weights recognise are
            ``page_quality``, ``backend_conf``, ``cross_read``, ``logprob``,
            ``validation`` and ``format_match``.  An unrecognised name is
            ignored, so adding a signal means adding a weight for it too.
        :param weights: relative influence per signal.  ``None`` uses the
            defaults in :data:`DEFAULT_CONFIDENCE_WEIGHTS`.  Only the ratios
            matter, since the pool renormalises.  Zero or negative drops a
            signal entirely.
        :param prior: returned when every signal is absent.  ``0.5`` is the
            honest "no evidence either way"; lower it if your documents are
            hard enough that silence should read as doubt.
        :returns: the fused score in ``0.0..1.0``, rounded to 4 places.
            *Uncalibrated* -- it ranks reliably, but is not a probability until
            a :class:`Calibrator` fitted on your own labelled data is applied.
        """
        weight_map = dict(weights or DEFAULT_CONFIDENCE_WEIGHTS)
        total_weight = 0.0
        accumulated = 0.0
        for name, value in signals.items():
            if value is None:
                continue
            weight = weight_map.get(name)
            if weight is None or weight <= 0:
                continue
            bounded = clamp(float(value), _SIGNAL_CLAMP, 1.0 - _SIGNAL_CLAMP)
            accumulated += weight * logit(bounded)
            total_weight += weight
        if total_weight <= 0:
            return round(float(prior), 4)
        return round(float(clamp(sigmoid(accumulated / total_weight), 0.0, 1.0)), 4)

    @staticmethod
    def page_quality_prior(quality: PageQuality, bbox: Optional[BBox] = None,
                           page: Optional[Page] = None) -> float:
        """Prior probability of a correct read, from measured page degradation.

        When a ``bbox`` and its ``page`` are supplied the measurement is localised
        to the evidence region: a page can be pristine at the top and destroyed at
        the bottom, and a total read from the destroyed part should not inherit the
        page's average health.

        :param quality: the page's measured :class:`PageQuality`
        :param bbox: the evidence region to localise to.  ``None`` uses the
            page-wide measurement, which is the right fallback but a blunt one.
        :param page: the page ``bbox`` belongs to, needed to crop the raster.
            Both this and ``bbox`` must be given for localisation to happen.
        :returns: the prior in ``0.0..1.0``, ready to pass
            to :func:`Confidence.fuse_confidence` as its ``page_quality`` signal
        """
        base = quality.score
        if bbox is None or page is None or not page.has_raster():
            return round(float(clamp(base, 0.02, 0.99)), 4)
        try:
            img = page.raster()
            x0, y0, x1, y1 = bbox.to_pixels(page.raster_dpi)
            h, w = image_shape(img)
            x0 = int(clamp(x0, 0, w - 1)); x1 = int(clamp(x1, x0 + 1, w))
            y0 = int(clamp(y0, 0, h - 1)); y1 = int(clamp(y1, y0 + 1, h))
            crop = img[y0:y1, x0:x1]
            if min(image_shape(crop)) < 8:
                return round(float(clamp(base, 0.02, 0.99)), 4)
            local = PageQuality(blur=measure_blur(crop), contrast=measure_contrast(crop),
                                effective_dpi=quality.effective_dpi,
                                noise=measure_noise(crop),
                                illumination=quality.illumination)
            # Blend: the region's own measurement dominates, the page's grounds it.
            return round(float(clamp(0.7 * local.score + 0.3 * base, 0.02, 0.99)), 4)
        except Exception as exc:
            logger.debug("local quality prior failed: %s", exc)
            return round(float(clamp(base, 0.02, 0.99)), 4)

    @staticmethod
    def align_spans(a: Sequence[TextSpan], b: Sequence[TextSpan],
                    iou_threshold: float = 0.3
                    ) -> List[Tuple[Optional[TextSpan], Optional[TextSpan]]]:
        """Pair up spans from two reads of the same page by geometric overlap.

        Greedy best-IoU matching.  Unmatched spans are returned paired with
        ``None`` so that a backend which simply *missed* a field is penalised, not
        silently ignored.

        :param a: spans from the first read, normally the one with better geometry
        :param b: spans from the second read
        :param iou_threshold: least intersection-over-union for two spans to pair,
            in ``0.0..1.0``.  ``0.3`` is deliberately loose, because two engines
            word-segment differently and demanding tight overlap would report
            disagreement where there is only different tokenisation.
        :returns: ``(a_span, b_span)`` pairs.  An unmatched span from ``a``
            appears as ``(span, None)`` and one from ``b`` as ``(None, span)``,
            so both directions of miss are visible.
        """
        pairs: List[Tuple[Optional[TextSpan], Optional[TextSpan]]] = []
        remaining = list(b)
        for span in a:
            best = None
            best_iou = iou_threshold
            for candidate in remaining:
                score = span.bbox.iou(candidate.bbox)
                if score >= best_iou:
                    best, best_iou = candidate, score
            if best is not None:
                remaining.remove(best)
                pairs.append((span, best))
            else:
                pairs.append((span, None))
        for leftover in remaining:
            pairs.append((None, leftover))
        return pairs

    @staticmethod
    def cross_read_agreement(a: Sequence[TextSpan], b: Sequence[TextSpan],
                             iou_threshold: float = 0.3,
                             min_geometric_fraction: float = 0.2) -> float:
        """Agreement in ``[0, 1]`` between two independent reads of one page.

        Prefers geometric matching, which is precise.  When too few spans match
        geometrically -- the normal case when one of the reads came from a vision
        model, whose bboxes are approximate by construction -- it falls back to
        comparing the two transcriptions as token sequences, which is coarser but
        still discriminating.

        Returns ``0.0`` if either read is empty: one method finding nothing where
        the other found text is maximal disagreement, not a missing measurement.

        :param a: spans from the first read
        :param b: spans from the second read
        :param iou_threshold: geometric pairing threshold, with the meaning it
            has in :func:`Confidence.align_spans`
        :param min_geometric_fraction: fraction of spans that must pair
            geometrically before the geometric score is trusted, in ``0.0..1.0``.
            Below it the comparison falls back to token sequences.  ``0.2`` is
            low on purpose: a vision read has approximate boxes, so requiring
            more would discard the signal exactly when it is most valuable.
        :returns: agreement in ``0.0..1.0``.  Feed it in as the ``cross_read``
            signal of :func:`Confidence.fuse_confidence` -- it is the strongest
            single signal available, because two methods that fail identically
            are rare.
        """
        if not a or not b:
            return 0.0
        approximate = any(s.meta.get("approximate_bbox") for s in list(a) + list(b))
        if not approximate:
            pairs = align_spans(a, b, iou_threshold)
            matched = [(x, y) for x, y in pairs if x is not None and y is not None]
            if pairs and len(matched) >= max(1, int(min_geometric_fraction * len(pairs))):
                total_weight = 0.0
                score = 0.0
                for x, y in pairs:
                    if x is None or y is None:
                        present = x if x is not None else y
                        if present is None:   # align_spans never pairs None with None
                            continue
                        total_weight += max(1, len(_compare_key(present.text)))
                        continue
                    key_x, key_y = _compare_key(x.text), _compare_key(y.text)
                    weight = max(1, max(len(key_x), len(key_y)))
                    score += weight * similarity(key_x, key_y)
                    total_weight += weight
                return round(float(_safe_div(score, total_weight)), 4)

        tokens_a = [_compare_key(t) for t in " ".join(s.text for s in a).split() ]
        tokens_b = [_compare_key(t) for t in " ".join(s.text for s in b).split()]
        tokens_a = [t for t in tokens_a if t]
        tokens_b = [t for t in tokens_b if t]
        if not tokens_a or not tokens_b:
            return 0.0
        return round(float(difflib.SequenceMatcher(None, tokens_a, tokens_b).ratio()), 4)

    @staticmethod
    def value_agreement(a: Any, b: Any, numeric_tolerance: float = 0.0) -> float:
        """Agreement between two *extracted values* rather than two transcriptions.

        Numbers compare numerically (``48250`` and ``48,250.00`` agree), dates
        compare as dates, and everything else compares as normalised strings.

        :param a: the first value, of any type
        :param b: the second value
        :param numeric_tolerance: relative slack for numbers, so ``0.001`` allows
            a tenth of a percent.  ``0.0`` demands exact equality, which is right
            for amounts -- two engines reading the same digits should agree
            exactly, and a near miss is a misread, not a rounding difference.
        :returns: ``1.0`` or ``0.0`` for numbers, booleans and dates, which either
            match or do not; a graded string similarity otherwise.  ``0.0`` when
            either value is ``None``.
        """
        if a is None or b is None:
            return 0.0
        if isinstance(a, bool) or isinstance(b, bool):
            return 1.0 if bool(a) == bool(b) else 0.0
        numbers = (int, float, decimal.Decimal)
        if isinstance(a, numbers) and isinstance(b, numbers):
            x, y = float(a), float(b)
            if x == y:
                return 1.0
            scale = max(abs(x), abs(y), 1e-9)
            return 1.0 if abs(x - y) <= numeric_tolerance * scale else 0.0
        if isinstance(a, (_dt.date, _dt.datetime)) and isinstance(b, (_dt.date, _dt.datetime)):
            return 1.0 if str(a)[:10] == str(b)[:10] else 0.0
        return round(similarity(_compare_key(str(a)), _compare_key(str(b))), 4)

    @staticmethod
    def format_match_score(value: Any, expected_type: str) -> Optional[float]:
        """Does ``value`` look like a well-formed instance of ``expected_type``?

        A weak signal on its own, but a genuinely independent one: it catches the
        field-shifted-by-one-row failure that every other signal misses, because a
        date sitting in an amount field is fluent, high-confidence and wrong.

        :param value: the extracted value to check
        :param expected_type: the declared type -- ``"number"``, ``"integer"``,
            ``"amount"``, ``"date"``, ``"boolean"``, ``"string"``, and their
            aliases.  Matching is by name, so an unrecognised type returns
            ``None`` rather than a guess.
        :returns: ``1.0`` when the value is well formed for the type, ``0.0``
            when it is not, or ``None`` when the type carries no format
            expectation -- a free-text field cannot be malformed.  ``None`` is
            dropped by the fusion pool rather than scored as failure.
        """
        if value is None:
            return 0.0
        text = str(value).strip()
        if not text:
            return 0.0
        kind = (expected_type or "").lower()
        if kind in ("int", "integer", "float", "number", "decimal", "amount", "money"):
            # "12-04-2024" parses as the amount 12, so a date landing in an amount
            # field would otherwise score a confident 1.0.  That off-by-one-row
            # failure is exactly what this signal exists to catch, and it is the
            # one no other signal sees: the value is fluent and the page is clean.
            if _DATE_SHAPED_RE.search(text) and parse_date(text) is not None:
                return 0.0
            return 1.0 if parse_amount(text) is not None else 0.0
        if kind in ("date", "datetime"):
            return 1.0 if parse_date(text) is not None else 0.0
        if kind in ("bool", "boolean"):
            return 1.0 if parse_bool(text) is not None else 0.0
        if kind in ("str", "string", "text"):
            # Long runs of OCR garbage characters are the failure mode here.
            letters = sum(1 for ch in text if ch.isalnum() or ch.isspace())
            return round(float(clamp(letters / float(len(text)), 0.0, 1.0)), 4)
        return None

    @staticmethod
    def reliability_curve(scores: Sequence[float], labels: Sequence[bool],
                          bins: int = 10) -> List[Dict[str, float]]:
        """Bin predictions and report empirical accuracy per bin.

        This is the plot that answers "is a 0.9 right 90% of the time?".  Until it
        is drawn, a confidence number is an assertion, not a measurement.

        :param scores: predicted confidences in ``0.0..1.0``
        :param labels: whether each prediction was actually correct
        :param bins: equal-width bins over the confidence range.  ``10`` is
            conventional.  Watch the ``count`` per bin -- a bin holding three
            samples says nothing, however tempting its accuracy looks.
        :returns: one dict per bin with ``bin_low``, ``bin_high``, ``count``,
            ``mean_confidence``, ``accuracy`` and ``gap``.  A positive ``gap``
            means underconfidence, negative means overconfidence, which is the
            usual direction.  Empty bins are included with zero counts so the
            list always has ``bins`` entries.
        :raises ConfigError: ``scores`` and ``labels`` differ in length
        """
        if len(scores) != len(labels):
            raise ConfigError("scores and labels differ in length")
        buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(max(1, bins))]
        for score, label in zip(scores, labels):
            index = int(clamp(float(score), 0.0, 0.999999) * bins)
            buckets[index].append((float(score), bool(label)))
        out = []
        for i, bucket in enumerate(buckets):
            if not bucket:
                out.append({"bin_low": i / float(bins), "bin_high": (i + 1) / float(bins),
                            "count": 0, "mean_confidence": 0.0, "accuracy": 0.0, "gap": 0.0})
                continue
            mean_conf = sum(s for s, _ in bucket) / len(bucket)
            accuracy = sum(1.0 for _, l in bucket if l) / len(bucket)
            out.append({"bin_low": i / float(bins), "bin_high": (i + 1) / float(bins),
                        "count": len(bucket), "mean_confidence": round(mean_conf, 4),
                        "accuracy": round(accuracy, 4), "gap": round(accuracy - mean_conf, 4)})
        return out

    @staticmethod
    def expected_calibration_error(scores: Sequence[float], labels: Sequence[bool],
                                   bins: int = 10) -> float:
        """Expected calibration error: the bin-count-weighted mean |accuracy - confidence|.

        The single number to track per release.  Under 0.05 is a defensible target
        for a shipped extraction pipeline.

        :param scores: predicted confidences in ``0.0..1.0``
        :param labels: whether each prediction was actually correct
        :param bins: histogram bins over the confidence range.  ``10`` is
            conventional; more bins resolve finer structure but need far more
            samples per bin to mean anything.
        :returns: the error in ``0.0..1.0``; ``0.0`` is perfect calibration, and
            also what an empty input returns.  Pair it with the sharpness-aware
            Brier score, which a constant predictor cannot game; that lives
            at :func:`Confidence.brier_score`.
        """
        curve = reliability_curve(scores, labels, bins)
        total = sum(b["count"] for b in curve)
        if not total:
            return 0.0
        return round(sum(b["count"] * abs(b["gap"]) for b in curve) / float(total), 4)

    @staticmethod
    def brier_score(scores: Sequence[float], labels: Sequence[bool]) -> float:
        """Mean squared error of the probabilities.  Rewards sharpness *and*
        calibration, unlike ECE, which a constant predictor can game.

        :param scores: predicted confidences in ``0.0..1.0``
        :param labels: whether each prediction was actually correct
        :returns: the mean squared error in ``0.0..1.0``, lower being better.
            A model that always predicts the base rate scores a perfect ECE and
            a poor Brier, which is exactly the difference worth watching.
        """
        if not scores:
            return 0.0
        return round(sum((float(s) - (1.0 if l else 0.0)) ** 2
                         for s, l in zip(scores, labels)) / float(len(scores)), 4)


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Confidence from the inside.
# ``Confidence.x`` is the documented path; ``x`` is the one that already works.
logit                      = Confidence.logit
sigmoid                    = Confidence.sigmoid
fuse_confidence            = Confidence.fuse_confidence
page_quality_prior         = Confidence.page_quality_prior
align_spans                = Confidence.align_spans
cross_read_agreement       = Confidence.cross_read_agreement
value_agreement            = Confidence.value_agreement
format_match_score         = Confidence.format_match_score
reliability_curve          = Confidence.reliability_curve
expected_calibration_error = Confidence.expected_calibration_error
brier_score                = Confidence.brier_score


#: Default fusion weights.  These are *priors*, chosen by how much independent
#: evidence each signal carries, and they are meant to be replaced by weights
#: fitted against a labelled set for your document class.  Ship the fitted ones.
DEFAULT_CONFIDENCE_WEIGHTS = {
    # Two independent recognisers agreeing on the same region is the strongest
    # signal available, and the hardest to fake: independent failures rarely
    # coincide on the same wrong string.  Absent unless a page was read twice.
    "cross_read_agree": 0.25,
    # Arithmetic and cross-field checks are near-deterministic and completely
    # independent of how the page was read.  Note the asymmetry encoded in
    # _build_field_result: a *failure* here is strong evidence, a *pass* is
    # weak, because plenty of wrong values still satisfy a sum check.
    "validation": 0.25,
    # Was the extracted value actually found in the recognised text?  Kept
    # separate from cross_read_agree, because they are independent: one says
    # two engines saw the same pixels the same way, the other says the
    # extractor did not invent a value that no engine ever read.  Folding them
    # into one slot means an ensemble read adds no confidence at all.
    "text_support": 0.20,
    # The OCR engine's own confidence over the evidence region.  Meaningful for
    # classical engines; deliberately absent for language models.
    "backend_conf": 0.15,
    # Prior from measured page degradation over the evidence region.
    "page_quality": 0.10,
    # Does the value match the expected shape for its declared type?
    "format_match": 0.05,
    # Token-level probability, where the provider exposes it.
    "logprob": 0.05,
}

#: No single signal may contribute more than ``logit(0.98)`` (~3.9) to the
#: pool.  Without this, a binary signal reporting exactly 1.0 lands at
#: ``logit(1 - 1e-6)`` -- nearly 14 log-odds -- and one "the format is valid"
#: check outvotes every genuine measurement on the page.
_SIGNAL_CLAMP = 0.02


def _compare_key(text: str) -> str:
    """Aggressive normalisation for agreement comparison only."""
    cleaned = normalize_digits(normalize_unicode(text or "")).lower()
    return re.sub(r"[^a-z0-9]+", "", cleaned)


# -- calibration -------------------------------------------------------------

class Calibrator(object):
    """Base for post-hoc probability calibration."""

    def fit(self, scores: Sequence[float], labels: Sequence[bool]) -> Calibrator:
        """Fit on scores and their correctness labels.  Returns ``self``.

        :param scores: raw fused confidences in ``[0, 1]``
        :param labels: True where that field was actually correct
        """
        raise NotImplementedError

    def predict(self, score: float) -> float:
        """Map one raw score onto a calibrated probability.

        Implementations must be monotone: calibration changes what a number
        means, never the ranking between two fields.

        :param score: a raw fused confidence in ``0.0..1.0``
        :returns: the calibrated probability in ``0.0..1.0``
        :raises NotImplementedError: on the base class
        """
        raise NotImplementedError

    def predict_many(self, scores: Sequence[float]) -> List[float]:
        """Calibrate a sequence of scores.

        :param scores: raw fused confidences
        :returns: the calibrated probabilities, in the same order
        """
        return [self.predict(s) for s in scores]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict, tagged with ``kind`` for :meth:`from_dict`."""
        raise NotImplementedError

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Calibrator:
        """Rebuild whichever calibrator ``d`` describes.

        The dispatcher that makes a fitted calibrator round-trip through an eval
        report -- which is how a calibration fitted last month gets applied to
        this month's run.

        :param d: a mapping from any calibrator's ``to_dict``, tagged with
            ``kind``: ``"platt"``, ``"isotonic"``, or ``"identity"``
        :returns: the reconstructed calibrator
        :raises ConfigError: ``kind`` is missing or unrecognised
        """
        kind = d.get("kind")
        if kind == "platt":
            return PlattCalibrator.from_dict(d)
        if kind == "isotonic":
            return IsotonicCalibrator.from_dict(d)
        if kind == "identity":
            return IdentityCalibrator()
        raise ConfigError("unknown calibrator kind %r" % kind)


class IdentityCalibrator(Calibrator):
    """Pass scores through unchanged.  The honest default before fitting."""

    def fit(self, scores: Sequence[float], labels: Sequence[bool]) -> IdentityCalibrator:
        """Nothing to fit.  Returns ``self``.

        :param scores: ignored
        :param labels: ignored
        :returns: ``self``, unchanged.  The point of this class is to be a
            drop-in that documents "no calibration was applied" instead of
            leaving ``calibrator=None`` ambiguous.
        """
        return self

    def predict(self, score: float) -> float:
        """The score itself, clamped to ``[0, 1]``.

        :param score: a fused confidence
        :returns: the same value, clamped
        """
        return float(clamp(score, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return {"kind": "identity"}


class PlattCalibrator(Calibrator):
    """Platt scaling: fit ``sigmoid(a * logit(s) + b)`` by log-loss.

    Two parameters, so it works from a few hundred labelled fields -- which is
    realistic for a first calibration set.  Fitted on the logit of the raw
    score rather than the score itself, so a perfectly calibrated input is
    recovered as the identity (``a=1, b=0``) instead of being squashed.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        """Start from ``a``, ``b`` -- the defaults are the identity mapping.

        :param a: slope applied to the logit of the raw score.  ``1.0`` with
            ``b=0.0`` is the identity, so an unfitted calibrator changes nothing.
        :param b: intercept in logit space; negative shifts scores down, which is
            what fitting produces when the raw scores are overconfident.
        """
        self.a = float(a)
        self.b = float(b)
        self.n = 0

    def fit(self, scores: Sequence[float], labels: Sequence[bool], iterations: int = 400,
            learning_rate: float = 0.25) -> PlattCalibrator:
        """Fit ``a`` and ``b`` by gradient descent on the log-loss.

        Targets are smoothed by Platt's correction so a perfectly separable set
        cannot drive the weights to infinity.

        :param scores: raw fused confidences in ``0.0..1.0``
        :param labels: whether each prediction was actually correct
        :param iterations: full-batch gradient steps.  ``400`` converges on the
            few-hundred-sample sets this is meant for; raising it is cheap but
            rarely changes the fit.
        :param learning_rate: step size.  ``0.25`` is stable for this loss --
            larger values oscillate, and the fit silently ends up worse rather
            than failing.
        :returns: ``self``, fitted
        """
        xs = [logit(clamp(float(s), 0.0, 1.0)) for s in scores]
        ys = [1.0 if bool(l) else 0.0 for l in labels]
        if len(xs) != len(ys):
            raise ConfigError("scores and labels differ in length")
        if not xs:
            return self
        self.n = len(xs)
        # Smoothed targets (Platt's correction) so that a separable set does
        # not drive the weights to infinity.
        positives = sum(ys)
        negatives = len(ys) - positives
        hi = (positives + 1.0) / (positives + 2.0)
        lo = 1.0 / (negatives + 2.0)
        targets = [hi if y > 0.5 else lo for y in ys]

        a, b = self.a, self.b
        for _ in range(iterations):
            grad_a = grad_b = 0.0
            for x, t in zip(xs, targets):
                p = sigmoid(a * x + b)
                error = p - t
                grad_a += error * x
                grad_b += error
            a -= learning_rate * grad_a / len(xs)
            b -= learning_rate * grad_b / len(xs)
        self.a, self.b = a, b
        return self

    def predict(self, score: float) -> float:
        """``sigmoid(a * logit(score) + b)``.

        :param score: a fused confidence in ``0.0..1.0``
        :returns: the calibrated probability, rounded to 4 places.  Monotone in
            ``score``, so calibration never reorders two fields -- it only
            changes what the numbers mean.
        """
        return round(float(sigmoid(self.a * logit(clamp(score, 0.0, 1.0)) + self.b)), 4)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict, including the sample count it was fitted on."""
        return {"kind": "platt", "a": self.a, "b": self.b, "n": self.n}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> PlattCalibrator:
        """Rebuild from :meth:`to_dict` output.

        A staticmethod, not a classmethod, so that it matches
        :meth:`Calibrator.from_dict` -- the base declares the dispatcher every
        calibrator must answer to, and a classmethod would take an extra
        positional parameter it does not have.

        :param d: a mapping from :meth:`to_dict`, with ``a``, ``b`` and ``n``
        :returns: the reconstructed calibrator; missing keys fall back to the
            identity mapping rather than raising
        """
        obj = PlattCalibrator(float(d.get("a", 1.0)), float(d.get("b", 0.0)))
        obj.n = int(d.get("n", 0))
        return obj

    def __repr__(self) -> str:
        """Parameters and the size of the fitting set."""
        return "PlattCalibrator(a=%.4f, b=%.4f, n=%d)" % (self.a, self.b, self.n)


class IsotonicCalibrator(Calibrator):
    """Isotonic regression via pool-adjacent-violators.

    Non-parametric, so it fixes arbitrary miscalibration shapes -- including
    the common one where a pipeline is well calibrated in the middle and wildly
    overconfident at the top.  Needs more labelled data than Platt scaling and
    will overfit a small set, so :meth:`fit` requires a minimum sample size
    rather than silently producing a step function.
    """

    def __init__(self, thresholds: Optional[Sequence[float]] = None,
                 values: Optional[Sequence[float]] = None) -> None:
        """Start from an optional fitted step function.

        Normally built by :meth:`fit` or :meth:`from_dict` rather than directly.

        :param thresholds: ascending score breakpoints of the step function
        :param values: the calibrated probability for each breakpoint, one per
            entry in ``thresholds``.  Both empty gives an unfitted calibrator
            that passes scores through.
        """
        self.thresholds = list(thresholds or [])
        self.values = list(values or [])
        self.n = 0

    def fit(self, scores: Sequence[float], labels: Sequence[bool],
            min_samples: int = 50) -> IsotonicCalibrator:
        """Fit a monotone step function by pool-adjacent-violators.

        :param scores: predicted confidences in ``0.0..1.0``
        :param labels: whether each prediction was actually correct
        :param min_samples: refuse to fit below this many samples rather than
            overfit.  ``50`` is already optimistic for a free-form step
            function; a few hundred is where it starts beating Platt scaling.
        :returns: ``self``, fitted
        :raises ConfigError: fewer than ``min_samples`` samples were supplied.
            The message points at :class:`PlattCalibrator`, whose two parameters
            survive a small set.
        """
        points = sorted(zip([clamp(float(s), 0.0, 1.0) for s in scores],
                            [1.0 if bool(l) else 0.0 for l in labels]))
        if len(points) < min_samples:
            raise ConfigError(
                "isotonic calibration needs at least %d labelled samples, got %d; "
                "use PlattCalibrator for smaller sets" % (min_samples, len(points)))
        self.n = len(points)
        # Pool adjacent violators.
        blocks = [[x, y, 1.0] for x, y in points]  # [score, mean_label, weight]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][1] > blocks[i + 1][1]:
                total = blocks[i][2] + blocks[i + 1][2]
                merged = [blocks[i][0],
                          (blocks[i][1] * blocks[i][2] + blocks[i + 1][1] * blocks[i + 1][2]) / total,
                          total]
                blocks[i:i + 2] = [merged]
                if i > 0:
                    i -= 1
            else:
                i += 1
        self.thresholds = [b[0] for b in blocks]
        self.values = [b[1] for b in blocks]
        return self

    def predict(self, score: float) -> float:
        """The fitted step at ``score``; the raw score when unfitted.

        :param score: a raw fused confidence in ``0.0..1.0``
        :returns: the calibrated probability, rounded to 4 places.  Piecewise
            constant, so nearby raw scores can map to the same output -- the
            price of not assuming a parametric shape.
        """
        if not self.thresholds:
            return float(clamp(score, 0.0, 1.0))
        x = clamp(float(score), 0.0, 1.0)
        # Piecewise-constant step, taking the last block at or below x.
        import bisect
        index = bisect.bisect_right(self.thresholds, x) - 1
        index = int(clamp(index, 0, len(self.values) - 1))
        return round(float(self.values[index]), 4)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict holding the full step function."""
        return {"kind": "isotonic", "thresholds": self.thresholds,
                "values": self.values, "n": self.n}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> IsotonicCalibrator:
        """Rebuild from :meth:`to_dict` output.

        A staticmethod for the same reason as :meth:`PlattCalibrator.from_dict`:
        it has to match the dispatcher declared on :class:`Calibrator`.

        :param d: a mapping from :meth:`to_dict`, with ``thresholds``, ``values``
            and ``n``
        :returns: the reconstructed calibrator; without ``thresholds`` it is
            unfitted and passes scores through
        """
        obj = IsotonicCalibrator(d.get("thresholds"), d.get("values"))
        obj.n = int(d.get("n", 0))
        return obj

    def __repr__(self) -> str:
        """Number of steps and the size of the fitting set."""
        return "IsotonicCalibrator(blocks=%d, n=%d)" % (len(self.values), self.n)


# =============================================================================
# 12. Layer 4 -- schema-driven extraction with confidence and provenance
# =============================================================================
#
# A domain project declares a schema and its business rules.  Everything from
# here to the typed result -- prompt assembly, chunking, lenient parsing,
# coercion, provenance recovery, confidence fusion -- is substrate.


@dataclass
class FieldSpec(object):
    """One extractable field: its name, declared type, and whether it is required."""

    name: str
    type_name: str = "string"
    description: str = ""
    required: bool = True
    item_type: Optional[str] = None
    path: str = ""

    def __post_init__(self) -> None:
        """Default the lookup ``path`` to the field's own name."""
        if not self.path:
            self.path = self.name

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return {"name": self.name, "type": self.type_name, "required": self.required,
                "description": self.description, "path": self.path}


_PY_TO_JSON = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    decimal.Decimal: "number", _dt.date: "string", _dt.datetime: "string",
    type(None): "null",
}


def _json_type_for(annotation: Any) -> Tuple[str, bool, Optional[str]]:
    """``(json_type, required, item_type)`` for a Python annotation."""
    try:
        import typing as _typing
        origin = getattr(_typing, "get_origin", lambda t: None)(annotation)
        args = getattr(_typing, "get_args", lambda t: ())(annotation)
    except Exception:  # pragma: no cover
        origin, args = None, ()

    if origin is Union:
        non_null = [a for a in args if a is not type(None)]
        required = len(non_null) == len(args)
        if len(non_null) == 1:
            inner, _, item = _json_type_for(non_null[0])
            return inner, required, item
        return "string", required, None
    if origin in (list, List, set, frozenset, tuple):
        item = _json_type_for(args[0])[0] if args else "string"
        return "array", True, item
    if origin in (dict, Dict):
        return "object", True, None
    if annotation in _PY_TO_JSON:
        return _PY_TO_JSON[annotation], True, None
    if isinstance(annotation, type):
        if issubclass(annotation, bool):
            return "boolean", True, None
        if issubclass(annotation, int):
            return "integer", True, None
        if issubclass(annotation, float):
            return "number", True, None
        if issubclass(annotation, str):
            return "string", True, None
        if dataclasses.is_dataclass(annotation) or hasattr(annotation, "__fields__") \
                or hasattr(annotation, "model_fields"):
            return "object", True, None
    return "string", True, None


class SchemaAdapter(object):
    """Bridge between a project's schema and this library's extraction machinery.

    Pydantic v1, Pydantic v2, dataclasses and plain dict specifications are all
    accepted, and all reduce to JSON Schema plus a flat field list.  Supporting
    four shapes is not indulgence: the whole point is that a consuming project
    should not have to adopt this library's idea of a model in order to use its
    substrate.
    """

    def __init__(self, schema: SchemaLike, json_schema: Dict[str, Any],
                 fields: List[FieldSpec], name: str = "Extraction") -> None:
        """Hold a schema alongside its normalised form.

        Built by :meth:`from_schema` rather than directly -- it is the classmethod
        that knows how to reduce each supported schema shape to these arguments.

        :param schema: the project's original schema object, retained so
            that :meth:`build` can instantiate it.  ``None`` means "dict only",
            and building then returns a plain dict.
        :param json_schema: the JSON Schema sent to the model in the prompt
        :param fields: the flat :class:`FieldSpec` list used for coercion and
            per-field confidence
        :param name: label for the schema, used in prompts and reports
        """
        self.schema = schema
        self.json_schema = json_schema
        self.fields = fields
        self.name = name

    # -- construction -----------------------------------------------------
    @classmethod
    def from_schema(cls, schema: SchemaLike) -> SchemaAdapter:
        """Adapt any supported schema shape.  The one entry point.

        Accepts a pydantic v1 or v2 model, a dataclass, a ``{field: type}`` dict,
        or an existing adapter (returned unchanged).  Anything else raises
        :class:`ConfigError` naming the shapes that do work.

        :param schema: one of --

            * a pydantic model class (v1 or v2), which carries descriptions and
              required-ness and so gives the model the most to work with;
            * a dataclass, whose annotations become the field types;
            * a ``{field: type}`` dict such as
              ``{"invoice_no": str, "total": float, "items": list}``, the
              quickest way to try something;
            * an existing :class:`SchemaAdapter`, returned unchanged so callers
              can pass either without checking.

        :returns: an adapter exposing ``json_schema`` and ``fields``
        :raises ConfigError: the object is none of the above
        """
        if isinstance(schema, SchemaAdapter):
            return schema
        if isinstance(schema, dict):
            return cls._from_dict_spec(schema)
        if hasattr(schema, "model_json_schema"):           # pydantic v2
            return cls._from_json_schema(schema, schema.model_json_schema(),
                                         getattr(schema, "__name__", "Extraction"))
        if hasattr(schema, "schema") and hasattr(schema, "__fields__"):  # pydantic v1
            return cls._from_json_schema(schema, schema.schema(),
                                         getattr(schema, "__name__", "Extraction"))
        if dataclasses.is_dataclass(schema):
            return cls._from_dataclass(schema)
        raise ConfigError(
            "unsupported schema %r: pass a pydantic model, a dataclass, or a dict "
            "of {field: type}" % schema)

    @classmethod
    def _from_json_schema(cls, schema: SchemaLike, json_schema: Dict[str, Any],
                          name: str) -> SchemaAdapter:
        """Adapt a model that can already emit JSON Schema (pydantic v1/v2).

        ``anyOf``/``oneOf`` branches -- how pydantic renders ``Optional[X]`` --
        are collapsed to their first non-null type, and date formats are lifted
        into a ``date`` type so coercion knows to parse them.
        """
        properties = json_schema.get("properties") or {}
        required = set(json_schema.get("required") or [])
        fields = []
        for field_name, spec in properties.items():
            type_name = spec.get("type")
            if not type_name:
                # anyOf/oneOf: pick the first non-null branch.
                for branch in (spec.get("anyOf") or spec.get("oneOf") or []):
                    if branch.get("type") and branch["type"] != "null":
                        type_name = branch["type"]
                        break
            if spec.get("format") in ("date", "date-time"):
                type_name = "date"
            fields.append(FieldSpec(
                name=field_name, type_name=type_name or "string",
                description=spec.get("description", "") or spec.get("title", ""),
                required=field_name in required,
                item_type=(spec.get("items") or {}).get("type")))
        return cls(schema, json_schema, fields, name)

    @classmethod
    def _from_dataclass(cls, schema: SchemaLike) -> SchemaAdapter:
        """Adapt a dataclass, deriving JSON Schema from its annotations.

        A field with a default (or a default factory) is treated as optional even
        when its annotation is not ``Optional``, matching what the author meant.
        """
        try:
            import typing as _typing
            hints = _typing.get_type_hints(schema)
        except Exception:
            hints = dict((f.name, f.type) for f in dataclasses.fields(schema))
        properties = {}
        required = []
        fields = []
        for f in dataclasses.fields(schema):
            annotation = hints.get(f.name, f.type)
            type_name, is_required, item_type = _json_type_for(annotation)
            has_default = (f.default is not dataclasses.MISSING
                           or f.default_factory is not dataclasses.MISSING)  # type: ignore
            is_required = is_required and not has_default
            entry: Dict[str, Any] = {"type": type_name}
            if item_type:
                entry["items"] = {"type": item_type}
            if annotation in (_dt.date, _dt.datetime):
                entry["format"] = "date"
                type_name = "date"
            properties[f.name] = entry
            if is_required:
                required.append(f.name)
            fields.append(FieldSpec(name=f.name, type_name=type_name,
                                    required=is_required, item_type=item_type))
        json_schema = {"title": schema.__name__, "type": "object",
                       "properties": properties, "required": required}
        return cls(schema, json_schema, fields, schema.__name__)

    @classmethod
    def _from_dict_spec(cls, spec: Dict[str, Any]) -> SchemaAdapter:
        """``{"total": "number", "patient": {"type": "string", "description": ...}}``"""
        properties = {}
        required = []
        fields = []
        for key, value in spec.items():
            if isinstance(value, dict):
                type_name = value.get("type", "string")
                description = value.get("description", "")
                is_required = bool(value.get("required", True))
                item_type = (value.get("items") or {}).get("type")
            else:
                type_name = str(value)
                description, is_required, item_type = "", True, None
            entry = {"type": "string" if type_name == "date" else type_name}
            if type_name == "date":
                entry["format"] = "date"
            if description:
                entry["description"] = description
            if item_type:
                entry["items"] = {"type": item_type}
            properties[key] = entry
            if is_required:
                required.append(key)
            fields.append(FieldSpec(name=key, type_name=type_name, description=description,
                                    required=is_required, item_type=item_type))
        return cls(None, {"title": "Extraction", "type": "object",
                          "properties": properties, "required": required},
                   fields, "Extraction")

    # -- use --------------------------------------------------------------
    def field(self, name: str) -> Optional[FieldSpec]:
        """The spec for ``name``, or ``None`` when the schema has no such field.

        :param name: the field name to look up
        :returns: its :class:`FieldSpec`, or ``None``.  ``None`` is normal, not
            an error: a model may return a field the schema never asked for, and
            that value is coerced as a string rather than dropped.
        """
        for spec in self.fields:
            if spec.name == name:
                return spec
        return None

    def build(self, data: Mapping[str, Any]) -> Any:
        """Instantiate the project's schema from raw extracted data.

        Falls back to returning the coerced dict when the schema cannot be
        instantiated -- a validation failure must not lose the extraction, it
        must be *reported alongside* it, because a reviewer can still use
        eleven correct fields out of twelve.

        :param data: already-coerced field values
        :returns: an instance of the project's schema -- a pydantic model, a
            dataclass instance, or a plain dict when the adapter was built from
            a dict spec.  Dataclass construction drops keys the dataclass does
            not declare.
        :raises ExtractionError: the schema rejected the data.  :func:`extract`
            catches this and records it as a warning, so the per-field results
            survive even when the object cannot be built.
        """
        if self.schema is None:
            return dict(data)
        try:
            if hasattr(self.schema, "model_validate"):     # pydantic v2
                return self.schema.model_validate(dict(data))
            if hasattr(self.schema, "parse_obj"):          # pydantic v1
                return self.schema.parse_obj(dict(data))
            if dataclasses.is_dataclass(self.schema):
                names = set(f.name for f in dataclasses.fields(self.schema))
                # is_dataclass() narrows to "a dataclass instance *or* class";
                # here it is the class, which is what we call.
                ctor = cast(Callable[..., Any], self.schema)
                return ctor(**dict((k, v) for k, v in data.items() if k in names))
        except Exception as exc:
            raise ExtractionError("schema validation failed: %s" % exc)
        return dict(data)

    def coerce(self, name: str, value: Any) -> Any:
        """Convert a raw JSON value to the declared type, tolerantly.

        :param name: the field the value belongs to; an unknown one is coerced
            as a string rather than rejected
        :param value: the raw value as the model returned it
        :returns: the coerced value, or ``None`` when it could not be coerced --
            which confidence fusion then scores as a format mismatch
        """
        spec = self.field(name)
        return coerce_value(value, spec.type_name if spec else "string",
                            spec.item_type if spec else None)


def coerce_value(value: Any, type_name: str, item_type: Optional[str] = None) -> Any:
    """Coerce a model's JSON output into the declared type.

    Tolerant on purpose: a model asked for a number will sometimes answer
    ``"Rs. 48,250.00"``, and rejecting that loses a correct extraction over a
    formatting preference.  Returns ``None`` when the value cannot be coerced,
    which the confidence machinery then scores as a format mismatch.

    :param value: the raw JSON value.  ``None`` passes straight through, since
        "the field is absent" is different from "the field would not parse".
    :param type_name: the JSON Schema type to coerce to -- ``"string"``,
        ``"number"``, ``"integer"``, ``"boolean"``, ``"date"``, ``"array"``, or
        ``"object"``.  Unknown names are treated as ``"string"``.
    :param item_type: element type for ``"array"``, applied to each entry.  A
        non-list value under ``"array"`` is wrapped in a one-element list, since
        a model asked for a list of one item routinely returns the item.
    :returns: the coerced value, or ``None`` when coercion failed

    What tolerance buys::

        coerce_value("Rs. 48,250.00", "number")   # -> 48250.0
        coerce_value("1,20,000", "number")        # -> 120000.0  (lakh grouping)
        coerce_value("15/03/2024", "date")        # -> "2024-03-15"
        coerce_value("yes", "boolean")            # -> True
        coerce_value("not a number", "number")    # -> None
    """
    if value is None:
        return None
    kind = (type_name or "string").lower()
    if kind == "array":
        if not isinstance(value, (list, tuple)):
            return [coerce_value(value, item_type or "string")]
        return [coerce_value(v, item_type or "string") for v in value]
    if kind == "object":
        return dict(value) if isinstance(value, Mapping) else value
    if kind in ("number", "float", "decimal", "amount", "money"):
        if isinstance(value, (int, float, decimal.Decimal)) and not isinstance(value, bool):
            return decimal.Decimal(str(value))
        return parse_amount(str(value))
    if kind in ("integer", "int"):
        amount = value if isinstance(value, int) and not isinstance(value, bool) \
            else parse_amount(str(value))
        if amount is None:
            return None
        try:
            return int(amount)
        except (ValueError, TypeError, decimal.InvalidOperation):
            return None
    if kind in ("boolean", "bool"):
        return value if isinstance(value, bool) else parse_bool(str(value))
    if kind in ("date", "datetime"):
        if isinstance(value, (_dt.date, _dt.datetime)):
            return value
        return parse_date(str(value))
    text = str(value).strip()
    return text or None


# -- prompt assembly ---------------------------------------------------------

#: The system prompt for extraction.  Every line here exists to suppress an
#: observed failure: inventing values for absent fields, silently normalising
#: numbers, translating Indian date order, and answering in prose.
DEFAULT_EXTRACTION_SYSTEM = (
    "You extract structured data from documents that have been scanned and read "
    "by OCR, so the text you receive may contain recognition errors.\n"
    "Rules:\n"
    "- Return a single JSON object matching the provided schema, and nothing else.\n"
    "- Copy values exactly as they appear in the document. Do not convert "
    "currencies, reformat numbers, or translate.\n"
    "- If a field is not present in the document, return null for it. Never "
    "invent, infer or estimate a value that is not written down.\n"
    "- If a value is present but partly illegible, return what is legible.\n"
    "- Dates: return ISO format (YYYY-MM-DD). Documents commonly use DD-MM-YYYY; "
    "read them accordingly.\n"
    "- Amounts: return the numeric value only, without currency symbols or "
    "thousands separators."
)


def format_document_text(doc: Document, include_page_markers: bool = True,
                         max_chars: int = 0) -> str:
    """Render a document's text for a prompt, with page markers.

    Page markers are not decoration: they are what lets an extracted value be
    traced back to a page when the model is asked to cite one, and what keeps
    a 40-page bundle's fields from being attributed to page 1.

    :param doc: a document that has been read.  Pages with no text are skipped
        entirely rather than emitting an empty marker.
    :param include_page_markers: prefix each page with ``--- page N ---``,
        numbered from 1 to match what a human sees.  Turn it off only when
        feeding a consumer that cannot tolerate the markers.
    :param max_chars: truncate at this many characters; ``0`` means no limit.
        Truncation prefers a whole-page boundary, falling back to a partial page
        only when the remaining room is large enough to be worth using.
    :returns: the rendered text, pages separated by blank lines
    """
    parts = []
    used = 0
    for page in doc.pages:
        text = page.text()
        if not text.strip():
            continue
        header = "--- page %d ---\n" % (page.index + 1) if include_page_markers else ""
        block = header + text
        if max_chars and used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > len(header) + 40:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_extraction_prompt(adapter: SchemaAdapter, document_text: str, context: str = "",
                            extra_instructions: str = "") -> str:
    """Assemble the user prompt: schema, domain context, then document text.

    The order is deliberate.  The schema comes first so the model knows what it
    is looking for before it reads, and the document goes last inside a fence so
    that instructions embedded in a scanned page read as data rather than as
    something to obey.

    :param adapter: the schema, whose ``json_schema`` is embedded verbatim
    :param document_text: the page text, normally produced
        by :func:`format_document_text`
    :param context: domain hints, e.g. ``"Indian private hospital bill, INR"``.
        Omitted from the prompt entirely when empty.
    :param extra_instructions: run-specific notes, appended after the context
        and also omitted when empty
    :returns: the complete user prompt, ending with the instruction to reply
        with JSON only
    """
    sections = [
        "Extract the following fields from the document below.",
        "",
        "JSON schema:",
        json.dumps(adapter.json_schema, indent=2, ensure_ascii=False),
    ]
    if context:
        sections += ["", "Document context:", context.strip()]
    if extra_instructions:
        sections += ["", "Additional instructions:", extra_instructions.strip()]
    sections += ["", "Document text (OCR output):", "```", document_text, "```", "",
                 "Respond with the JSON object only."]
    return "\n".join(sections)


def chunk_pages(doc: Document, max_chars: int = 60000,
                overlap_pages: int = 1) -> List[Document]:
    """Split a document into page-aligned chunks that fit a context window.

    Splitting on page boundaries rather than character counts keeps tables and
    totals intact, and the one-page overlap catches the field that straddles a
    page break -- which is exactly where discharge dates and totals live.

    :param doc: the document to split
    :param max_chars: character budget per chunk.  ``60000`` fits current context
        windows with room for the schema and the reply.  ``0`` or less disables
        chunking and returns the document whole.
    :param overlap_pages: pages repeated at the start of each subsequent chunk.
        ``1`` catches the straddling field; ``0`` disables the overlap and is
        right only when pages are genuinely independent.  Overlap costs tokens
        at every boundary, which is why it is one page and not three.
    :returns: one or more :class:`Document` chunks sharing the original's
        ``source_uri`` and ``meta``.  Always at least one, so callers need no
        empty-case branch.  A single page longer than ``max_chars`` is never
        split, since splitting mid-table is worse than one oversized call.
    """
    if max_chars <= 0:
        return [doc]
    chunks: List[Document] = []
    current: List[Page] = []
    size = 0
    for page in doc.pages:
        length = len(page.text()) + 32
        if current and size + length > max_chars:
            chunks.append(Document(source_uri=doc.source_uri, pages=list(current),
                                   meta=dict(doc.meta)))
            current = current[-overlap_pages:] if overlap_pages else []
            size = sum(len(p.text()) + 32 for p in current)
        current.append(page)
        size += length
    if current:
        chunks.append(Document(source_uri=doc.source_uri, pages=list(current),
                               meta=dict(doc.meta)))
    return chunks or [doc]


# -- model clients -----------------------------------------------------------

class LLMClient(Protocol):  # pragma: no cover - structural type
    """Anything that can turn a prompt into text and report what it cost."""

    name = ""

    def complete(self, prompt: str, system: Optional[str] = None,
                 images: Optional[Sequence[bytes]] = None) -> Tuple[str, Cost]:
        """Complete ``prompt`` and report what it cost.

        :param prompt: the fully assembled user prompt
        :param system: system instructions, when the provider supports them
        :param images: JPEG page images to send alongside the prompt
        :returns: ``(text, cost)``
        """
        ...


class BaseLLMClient(object):
    """Shared plumbing for model clients."""

    name = "llm"

    def __init__(self, model: str = "", max_tokens: int = 4096,
                 temperature: float = 0.0, **options: Any) -> None:
        """Configure the model call.

        :param model: provider model id -- also the pricing key
        :param max_tokens: ceiling on the response
        :param temperature: 0.0 -- extraction is not a creative task
        :param options: passed through to the underlying SDK
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.options = dict(options)
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Whether this client can be used right now.  True by default."""
        return True

    def complete(self, prompt: str, system: Optional[str] = None,
                 images: Optional[Sequence[bytes]] = None) -> Tuple[str, Cost]:
        """Complete ``prompt``.  Subclasses implement this.

        :param prompt: the fully assembled user prompt
        :param system: system instructions, when the provider supports them
        :param images: JPEG page images to send alongside the prompt
        :returns: ``(text, cost)``
        :raises NotImplementedError: always -- this is the extension point
        """
        raise NotImplementedError


class EchoClient(BaseLLMClient):
    """A deterministic test double.

    Returns a canned response, or one produced by a callable given the prompt.
    Having this in the library rather than in a test file is deliberate:
    consuming projects need to test their schemas and validators without
    hitting a paid API, and every one of them would otherwise write it again.
    """

    name = "echo"

    def __init__(self, response: Union[str, Mapping[str, Any], Callable[[str], Any]] = "{}",
                 cost: Optional[Cost] = None, record: bool = True, **options: Any) -> None:
        """Configure the double.

        :param response: what to return -- a JSON string (``'{"total": 100}'``),
            a mapping (``{"total": 100}``, encoded for you), or a callable
            ``fn(prompt) -> str | mapping``, which is how you make the double
            answer differently per chunk or assert on what it was asked.
        :param cost: the :class:`Cost` to report per call, for exercising budget
            and pricing logic.  Defaults to one call at zero money.
        :param record: keep every prompt in ``self.prompts``, so a test can assert
            on what the pipeline actually asked for
        """
        BaseLLMClient.__init__(self, model="echo", **options)
        self.response = response
        self.cost = cost or Cost(calls=1)
        self.record = record
        self.prompts: List[str] = []

    def complete(self, prompt: str, system: Optional[str] = None,
                 images: Optional[Sequence[bytes]] = None) -> Tuple[str, Cost]:
        """Return the canned response (JSON-encoded if it is not a string).

        :param prompt: recorded on ``self.prompts`` when ``record`` is set, and
            passed to ``response`` when that is a callable
        :param system: accepted and ignored -- the double does not model system
            instructions
        :param images: accepted and ignored
        :returns: ``(text, cost)`` with the configured cost.  Deterministic by
            construction, which is what makes accuracy assertions in the test
            suite exact rather than plausible.
        """
        if self.record:
            self.prompts.append(prompt)
        response = self.response(prompt) if callable(self.response) else self.response
        if not isinstance(response, str):
            response = json.dumps(_as_jsonable(response))
        return response, self.cost


class AnthropicClient(BaseLLMClient):
    """Claude, via the ``anthropic`` SDK.  Text and/or page images."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5",
                 api_key: Optional[str] = None, client: Any = None,
                 **options: Any) -> None:
        """Configure the Claude client.

        :param model: Claude model id, e.g. ``"claude-sonnet-5"``.  Also the
            pricing key -- see :func:`Pricing.set_pricing`.
        :param api_key: falls back to the ``ANTHROPIC_API_KEY`` environment
            variable when ``None``
        :param client: a pre-built SDK client, injected instead of constructed
        """
        BaseLLMClient.__init__(self, model=model, **options)
        self._api_key = api_key
        self._client = client

    def is_available(self) -> bool:
        """True with an injected client, or with the SDK plus an API key."""
        if self._client is not None:
            return True
        return have("anthropic") and bool(self._api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self) -> Any:
        """Construct the SDK client once, under the lock."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    module = require("anthropic", "Claude extraction client")
                    self._client = (module.Anthropic(api_key=self._api_key)
                                    if self._api_key else module.Anthropic())
        return self._client

    def complete(self, prompt: str, system: Optional[str] = None,
                 images: Optional[Sequence[bytes]] = None) -> Tuple[str, Cost]:
        """One Messages API call, with any page images attached first.

        :param prompt: the fully assembled user prompt
        :param system: system instructions, when the provider supports them
        :param images: JPEG page images sent alongside the prompt, for pages OCR
            handled badly.  Ignored by providers without vision.
        :returns: ``(text, cost)`` -- the raw completion, and what it billed as
            a :class:`Cost`.  The text goes through :func:`parse_json_lenient`,
            so a fenced code block or a prose preamble is tolerated.
        """
        content: List[Dict[str, Any]] = []
        for blob in (images or []):
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(blob).decode("ascii")}})
        content.append({"type": "text", "text": prompt})
        kwargs = {"model": self.model, "max_tokens": self.max_tokens,
                  "temperature": self.temperature,
                  "messages": [{"role": "user", "content": content}]}
        if system:
            kwargs["system"] = system
        message = self._get_client().messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in getattr(message, "content", []))
        usage = getattr(message, "usage", None)
        return text, price_tokens(self.model,
                                  int(getattr(usage, "input_tokens", 0) or 0),
                                  int(getattr(usage, "output_tokens", 0) or 0))


class OpenAIClient(BaseLLMClient):
    """GPT models via the ``openai`` SDK."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 client: Any = None, **options: Any) -> None:
        """Configure the OpenAI client.

        :param model: OpenAI model id, e.g. ``"gpt-4o"``.  Also the pricing key.
        :param api_key: falls back to the ``OPENAI_API_KEY`` environment
            variable when ``None``
        :param client: a pre-built SDK client, injected instead of constructed --
            for a proxy, custom transport, or a test double
        :param options: forwarded to :class:`BaseLLMClient` -- ``max_tokens``,
            ``temperature``
        """
        BaseLLMClient.__init__(self, model=model, **options)
        self._api_key = api_key
        self._client = client

    def is_available(self) -> bool:
        """True with an injected client, or with the SDK plus an API key."""
        if self._client is not None:
            return True
        return have("openai") and bool(self._api_key or os.environ.get("OPENAI_API_KEY"))

    def _get_client(self) -> Any:
        """Construct the SDK client once, under the lock."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    module = require("openai", "OpenAI extraction client")
                    self._client = (module.OpenAI(api_key=self._api_key)
                                    if self._api_key else module.OpenAI())
        return self._client

    def complete(self, prompt: str, system: Optional[str] = None,
                 images: Optional[Sequence[bytes]] = None) -> Tuple[str, Cost]:
        """One chat-completions call, with any page images attached.

        :param prompt: the fully assembled user prompt
        :param system: system instructions, when the provider supports them
        :param images: JPEG page images sent alongside the prompt, for pages OCR
            handled badly
        :returns: ``(text, cost)`` -- the raw completion, and what it billed as
            a :class:`Cost`.  The text goes through :func:`parse_json_lenient`,
            so a fenced code block or a prose preamble is tolerated.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for blob in (images or []):
            url = "data:image/jpeg;base64,%s" % base64.b64encode(blob).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": content}]
        response = self._get_client().chat.completions.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature, messages=messages)
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return text, price_tokens(self.model,
                                  int(getattr(usage, "prompt_tokens", 0) or 0),
                                  int(getattr(usage, "completion_tokens", 0) or 0))


# -- lenient JSON ------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_lenient(text: Optional[str]) -> Any:
    """Recover a JSON object from a model's reply.

    Models wrap JSON in fences, prefix it with "Here is the extracted data:",
    leave trailing commas, and occasionally emit ``NaN``.  Failing the whole
    extraction over any of that would be throwing away a correct answer on a
    presentation detail, so this walks through progressively more forgiving
    strategies and raises only when nothing parses.

    :param text: the model's raw reply.  ``None`` and empty both raise, since
        there is a real difference between "the model said nothing" and "the
        model said something unparseable".
    :returns: the decoded object -- usually a ``dict``, sometimes a ``list``
        when the model wrapped its answer in one
    :raises ExtractionError: no strategy recovered valid JSON.  :func:`extract`
        catches this per chunk and records a warning, so one unparseable chunk
        does not lose the others.
    """
    if text is None:
        raise ExtractionError("model returned no text")
    raw = text.strip()
    if not raw:
        raise ExtractionError("model returned empty text")

    candidates = [raw]
    fenced = _FENCE_RE.search(raw)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    balanced = _first_balanced_json(raw)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        for attempt in (candidate, _repair_json(candidate)):
            if not attempt:
                continue
            try:
                return _drop_non_finite(json.loads(attempt))
            except ValueError:
                continue
    raise ExtractionError("could not parse JSON from model output: %r"
                          % (raw[:300] + ("..." if len(raw) > 300 else "")))


def _drop_non_finite(value: Any) -> Any:
    """Replace NaN and +/-Infinity with ``None``, recursively.

    Python's ``json`` accepts these as float literals, so they survive parsing
    and then poison everything downstream: ``Decimal(nan)`` is a valid object,
    ``nan != nan`` breaks every comparison, and the value serialises back out
    as invalid JSON.  A missing value is the honest reading of ``NaN`` anyway.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return dict((k, _drop_non_finite(v)) for k, v in value.items())
    if isinstance(value, list):
        return [_drop_non_finite(v) for v in value]
    return value


def _first_balanced_json(text: str) -> Optional[str]:
    """Slice out the first balanced ``{...}`` or ``[...]``, ignoring braces in strings."""
    start = None
    opener = closer = ""
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener, closer = ch, ("}" if ch == "{" else "]")
            break
    if start is None:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair_json(text: str) -> str:
    """Fix the small syntax errors models actually make."""
    if not text:
        return text
    out = re.sub(r",\s*([}\]])", r"\1", text)            # trailing commas
    out = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "null", out)
    out = re.sub(r"//[^\n\"]*$", "", out, flags=re.MULTILINE)  # line comments
    out = out.replace("“", '"').replace("”", '"')     # smart quotes
    return out.strip()


# -- provenance --------------------------------------------------------------

@dataclass
class Evidence(object):
    """Where a value was found, and how well it matched."""

    bbox: BBox
    score: float
    text: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return {"bbox": self.bbox.to_dict(), "score": round(self.score, 4),
                "text": self.text, "source": self.source}


#: Renderings a date may plausibly have on the page.  After :func:`_compare_key`
#: strips separators these collapse to a handful of distinct keys.
_DATE_RENDERINGS = ("%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%b-%Y", "%b-%d-%Y",
                    "%d-%B-%Y", "%d-%m-%y")


def _comparison_keys(value: Any) -> List[str]:
    """Every form ``value`` might plausibly take on the page.

    A value's extracted representation legitimately differs from its printed
    one for exactly two types.  Numbers are printed with currency symbols and
    grouping (``Rs 1,18,250.00``) but extracted as ``118250.00``.  Dates are
    printed ``12-04-2024`` but -- because the extraction prompt asks for it --
    come back as ``2024-04-12``.  Without this, the pipeline instructs the
    model to normalise and then penalises it for having done so, scoring every
    correctly-read date as though it were a hallucination.
    """
    primary = _compare_key(str(value))
    if not primary:
        return []
    keys = [primary]

    text = str(value)
    is_date = isinstance(value, (_dt.date, _dt.datetime)) or bool(
        _DATE_SHAPED_RE.search(text))
    if is_date:
        parsed = value if isinstance(value, (_dt.date, _dt.datetime)) else parse_date(text)
        if parsed is not None:
            for pattern in _DATE_RENDERINGS:
                try:
                    key = _compare_key(parsed.strftime(pattern))
                except ValueError:  # pragma: no cover - platform strftime quirks
                    continue
                if key and key not in keys:
                    keys.append(key)
        return keys

    # Not a date, so a numeric reading is safe: "2024-04-12" would otherwise
    # be matched against the bare number 2024 and find spurious evidence.
    numeric = parse_amount(text)
    if numeric is not None:
        for rendering in (format(numeric.normalize(), "f"), format(numeric, "f")):
            key = _compare_key(rendering)
            if key and key not in keys:
                keys.append(key)
    return keys


def locate_value(doc: Document, value: Any, min_score: float = 0.72,
                 max_span_window: int = 8,
                 pages: Optional[Sequence[int]] = None) -> List[Evidence]:
    """Find where on which page a value appears, returning bounding boxes.

    Every extracted fact came from a specific rectangle on a specific page.  If
    that link is dropped during extraction it cannot be rebuilt later, and with
    it goes the ability to show a reviewer or an auditor *why* we believe
    something -- so it is recovered here even when the reading backend gave
    only page-level geometry.

    Matching is fuzzy and window-based: a value is compared against every run of
    up to ``max_span_window`` consecutive spans on a line, because "1,23,456.78"
    may arrive as three separate word spans.  Numeric values are additionally
    matched on their digits alone, so "Rs 48,250.00" is found for ``48250``.

    Returns the best matches sorted by score, one per page at most.

    :param doc: a document that has been read, so its pages carry spans
    :param value: the extracted value to locate.  Numbers and dates are matched
        in every form they might be printed in, so ``48250`` finds
        ``"Rs 48,250.00"`` and ``"2024-04-12"`` finds ``"12-04-2024"``.
        ``None`` returns no evidence.
    :param min_score: least similarity for a match to count, in ``0.0..1.0``.
        ``0.72`` tolerates the usual OCR damage while rejecting coincidence.
        Raise it towards ``0.85`` when values are long and distinctive, lower it
        only if you would rather have approximate evidence than none.
    :param max_span_window: how many consecutive spans on a line may be joined
        to form a candidate.  ``8`` covers ``"Rs"``, ``"1,23,456.78"`` split
        across word spans, and a label sharing the line.  Raising it costs time
        quadratically and invites spurious matches.
    :param pages: restrict the search to these page indices; ``None`` searches
        every page.  Worth passing when the model reported which page it read a
        value from -- it is faster and avoids matching a repeated header.
    :returns: the best :class:`Evidence` per page scoring at least
        ``min_score``, sorted best first.  Empty when nothing matched, which is
        itself a signal -- confidence fusion scores an unlocatable value lower.
    """
    if value is None:
        return []
    targets = _comparison_keys(value)
    if not targets:
        return []
    target = targets[0]

    results: List[Evidence] = []
    for page in doc.pages:
        if pages is not None and page.index not in pages:
            continue
        best: Optional[Evidence] = None
        for line_spans in _group_spans_into_lines(page):
            for start in range(len(line_spans)):
                accumulated = ""
                for end in range(start, min(start + max_span_window, len(line_spans))):
                    accumulated += _compare_key(line_spans[end].text)
                    if not accumulated:
                        continue
                    score = max(similarity(accumulated, t) for t in targets)
                    if best is None or score > best.score:
                        boxes = [s.bbox for s in line_spans[start:end + 1]]
                        merged = boxes[0]
                        for box in boxes[1:]:
                            merged = merged.union(box)
                        best = Evidence(bbox=merged, score=score,
                                        text=" ".join(s.text for s in line_spans[start:end + 1]),
                                        source=line_spans[start].source)
                    if len(accumulated) > 2 * len(target) + 4:
                        break
        if best is not None and best.score >= min_score:
            results.append(best)
    results.sort(key=lambda e: e.score, reverse=True)
    return results


def _group_spans_into_lines(page: Page,
                            y_tolerance: Optional[float] = None) -> List[List[TextSpan]]:
    """Spans grouped into visual lines, left to right, top to bottom.

    Backend-reported line ids are used when available, but keyed on
    ``(block, line)`` rather than ``line`` alone.  PyMuPDF -- and every other
    engine with a block/line/word hierarchy -- restarts its line counter inside
    each block, so grouping on the line index by itself silently welds the
    first line of every block on the page into one enormous line.  That
    corrupts reading order and defeats provenance lookup, which searches
    within lines.
    """
    spans = [s for s in page.spans if s.text.strip()]
    if not spans:
        return []
    if y_tolerance is None:
        heights = sorted(s.bbox.height for s in spans)
        y_tolerance = (heights[len(heights) // 2] or 10.0) * 0.6

    if all(s.line is not None for s in spans):
        groups: Dict[Tuple[int, int], List[TextSpan]] = {}
        for s in spans:
            groups.setdefault((int(s.block or 0), int(s.line or 0)), []).append(s)
        ordered = sorted(groups.values(),
                         key=lambda g: (min(x.bbox.y0 for x in g),
                                        min(x.bbox.x0 for x in g)))
    else:
        ordered = []
        for s in sorted(spans, key=lambda s: (s.bbox.y0, s.bbox.x0)):
            for group in reversed(ordered):
                if abs(s.bbox.y0 - group[0].bbox.y0) <= y_tolerance:
                    group.append(s)
                    break
            else:
                ordered.append([s])
    for group in ordered:
        group.sort(key=lambda s: s.bbox.x0)
    return ordered


def backend_confidence_for(doc: Document, evidence: Sequence[Evidence]) -> Optional[float]:
    """Mean backend-reported confidence over the spans supporting a value.

    Returns ``None`` when no supporting span carries a confidence, which keeps
    "the engine declined to say" distinct from "the engine said zero".

    :param doc: the document the evidence points into
    :param evidence: spans of evidence, as returned by :func:`locate_value`
    :returns: the mean confidence over overlapping spans, rounded to 4 places;
        or ``None`` when no supporting span reported one.  Vision backends
        report nothing here, which is why cross-read agreement carries the
        weight for them.
    """
    values: List[float] = []
    for item in evidence:
        try:
            page = doc.page(item.bbox.page)
        except KeyError:
            continue
        for span in page.spans_in(item.bbox, min_overlap=0.3):
            if span.confidence is not None:
                values.append(float(span.confidence))
    if not values:
        return None
    return round(sum(values) / len(values), 4)


# -- validation --------------------------------------------------------------

@dataclass
class ValidationIssue(object):
    """A business-rule failure, optionally attributed to specific fields."""

    message: str
    fields: List[str] = field(default_factory=list)
    severity: str = "error"  #: "error" or "warning"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return {"message": self.message, "fields": list(self.fields),
                "severity": self.severity}

    def __str__(self) -> str:
        """``error: line_items sums to 4800 but total is 48250``."""
        return "%s: %s" % (self.severity, self.message)


Validator = Callable[[Any], Any]


class Validators(object):
    """Domain-independent validators, as factories.

    Each returns a ``Validator`` closure, so a schema declares its checks
    as data.  Only structural invariants that hold for any document type
    belong here -- arithmetic consistency, date ordering, a format match.
    Anything that knows what a field *means* belongs in the consuming
    project.
    """

    @staticmethod
    def run_validators(model: Any, validators: Sequence[Validator]) -> List[ValidationIssue]:
        """Run business rules, collecting issues.

        A validator may return ``None``, a string, a :class:`ValidationIssue`, or a
        list of those.  A validator that *raises* is itself reported as an issue
        rather than aborting extraction: a buggy rule must not cost you the
        document.

        :param model: the built schema object to validate
        :param validators: the rules to run.  Each takes the model and returns
            ``None`` (passed), a string (an error), a :class:`ValidationIssue`,
            or a list of those.
        :returns: every issue raised, in validator order.  A validator that
            raised contributes one issue of severity ``"warning"`` naming the
            exception.
        """
        issues: List[ValidationIssue] = []
        for validator in validators or []:
            name = getattr(validator, "__name__", repr(validator))
            try:
                outcome = validator(model)
            except Exception as exc:
                issues.append(ValidationIssue(
                    "validator %s raised %s: %s" % (name, type(exc).__name__, exc),
                    severity="warning"))
                continue
            for item in (outcome if isinstance(outcome, (list, tuple)) else [outcome]):
                if item is None:
                    continue
                if isinstance(item, ValidationIssue):
                    issues.append(item)
                else:
                    issues.append(ValidationIssue(str(item), fields=[], severity="error"))
        return issues

    @staticmethod
    def line_items_sum_to_total(items_field: str = "line_items", amount_field: str = "amount",
                                total_field: str = "total",
                                tolerance: decimal.Decimal = decimal.Decimal("0.02"),
                                relative_tolerance: float = 0.0) -> Validator:
        """Validator factory: line items must add up to the stated total.

        The single most valuable check on a bill, because it is *independent* of
        the reading: an OCR error in any one amount breaks the sum, which is why
        this feeds confidence fusion rather than just producing a warning.

        :param items_field: attribute or key holding the line items, e.g.
            ``"line_items"`` or ``"charges"``
        :param amount_field: the per-item amount attribute, e.g. ``"amount"``
        :param total_field: the stated document total, e.g. ``"total"`` or
            ``"net_payable"``
        :param tolerance: absolute slack, as a :class:`decimal.Decimal`.
            ``0.02`` absorbs two paise of rounding.  Decimal rather than float
            on purpose -- money compared in binary floating point produces
            failures nobody can reproduce.
        :param relative_tolerance: extra slack proportional to the total, so
            ``0.005`` allows half a percent.  Use it when the document rounds
            its own total, and keep it at ``0.0`` otherwise: a percentage
            tolerance on a large bill hides real errors.
        :returns: a ``Validator`` closure.  It passes quietly when either field
            is missing, since "this document has no line items" is not an
            arithmetic failure.
        """
        def validate(model: Any) -> Optional[ValidationIssue]:
            """Compare the summed line items against the stated total."""
            items = _get_attr(model, items_field)
            total = _get_attr(model, total_field)
            if not items or total is None:
                return None
            try:
                summed = sum(decimal.Decimal(str(_get_attr(i, amount_field) or 0)) for i in items)
                stated = decimal.Decimal(str(total))
            except (decimal.InvalidOperation, TypeError, ValueError):
                return ValidationIssue("could not sum %s" % items_field,
                                       fields=[items_field, total_field], severity="warning")
            allowed = tolerance + (abs(stated) * decimal.Decimal(str(relative_tolerance)))
            if abs(summed - stated) > allowed:
                return ValidationIssue(
                    "%s sums to %s but %s is %s" % (items_field, summed, total_field, stated),
                    fields=[items_field, total_field])
            return None
        validate.__name__ = "line_items_sum_to_total"
        return validate

    @staticmethod
    def date_order(earlier_field: str, later_field: str, allow_equal: bool = True) -> Validator:
        """Validator factory: one date must not precede another.

        Catches the digit-transposition errors OCR makes on dates, which are
        otherwise invisible -- ``2024`` misread as ``2074`` is still a valid date.

        :param earlier_field: the field that should come first, e.g.
            ``"admission_date"``
        :param later_field: the field that should come second, e.g.
            ``"discharge_date"``
        :param allow_equal: treat equal dates as valid.  ``True`` is right for
            same-day admission and discharge; set ``False`` when the two genuinely
            cannot coincide.
        :returns: a ``Validator`` closure, passing quietly when either date is
            missing.  Both values must be comparable -- coerce them to dates
            first via a ``date``-typed schema field.
        """
        def validate(model: Any) -> Optional[ValidationIssue]:
            """Check that the earlier field really is not after the later one."""
            a = _get_attr(model, earlier_field)
            b = _get_attr(model, later_field)
            if a is None or b is None:
                return None
            if a > b or (not allow_equal and a == b):
                return ValidationIssue("%s (%s) is after %s (%s)"
                                       % (earlier_field, a, later_field, b),
                                       fields=[earlier_field, later_field])
            return None
        validate.__name__ = "date_order(%s<=%s)" % (earlier_field, later_field)
        return validate

    @staticmethod
    def field_matches(field_name: str, pattern: str,
                      message: Optional[str] = None) -> Validator:
        """Validator factory: a field must match a regular expression.

        The cheapest way to catch a value the model invented in the right shape
        but the wrong place -- a GSTIN that is 14 characters, an invoice number
        missing its prefix.

        :param field_name: the field to check
        :param pattern: a regular expression, applied with ``re.search`` so it
            need not match the whole value.  Anchor it with ``^...$`` when it
            should, e.g. ``"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9]Z[A-Z0-9]$"`` for
            a GSTIN.
        :param message: the issue text on failure; ``None`` generates one naming
            the field and the pattern
        :returns: a ``Validator`` closure, passing quietly when the field is
            absent -- absence is a completeness question, not a format one
        """
        compiled = re.compile(pattern)

        def validate(model: Any) -> Optional[ValidationIssue]:
            """Check the field against the compiled pattern."""
            value = _get_attr(model, field_name)
            if value is None:
                return None
            if not compiled.search(str(value)):
                return ValidationIssue(message or "%s (%r) does not match %s"
                                       % (field_name, value, pattern), fields=[field_name])
            return None
        validate.__name__ = "field_matches(%s)" % field_name
        return validate

    @staticmethod
    def required_fields(*names: str) -> Validator:
        """Validator factory: these fields must be present and non-empty."""
        def validate(model: Any) -> List[ValidationIssue]:
            """Report one issue per missing or blank field."""
            issues = []
            for name in names:
                value = _get_attr(model, name)
                if value is None or (isinstance(value, str) and not value.strip()):
                    issues.append(ValidationIssue("%s is missing" % name, fields=[name]))
            return issues
        validate.__name__ = "required_fields"
        return validate


# Module-level aliases.  These are load-bearing, not merely back-compatible:
# the staticmethods above call one another through these flat names, resolved
# from module globals at call time.  Deleting them breaks Validators from the inside.
# ``Validators.x`` is the documented path; ``x`` is the one that already works.
run_validators          = Validators.run_validators
line_items_sum_to_total = Validators.line_items_sum_to_total
date_order              = Validators.date_order
field_matches           = Validators.field_matches
required_fields         = Validators.required_fields


def _get_attr(obj: Any, name: str) -> Any:
    """Attribute or key access, whichever the object supports."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


# -- results -----------------------------------------------------------------

@dataclass
class FieldResult(Generic[T]):
    """One extracted field, wrapped in everything needed to judge it."""

    name: str
    value: Optional[T] = None
    confidence: float = 0.0
    evidence: List[BBox] = field(default_factory=list)
    method: str = ""
    signals: Dict[str, Optional[float]] = field(default_factory=dict)
    raw: Any = None
    warnings: List[str] = field(default_factory=list)

    @property
    def pages(self) -> List[int]:
        """Page indices this value has evidence on, ascending."""
        return sorted(set(b.page for b in self.evidence))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict, including the individual confidence signals."""
        return {"name": self.name, "value": _as_jsonable(self.value),
                "confidence": round(self.confidence, 4),
                "evidence": [b.to_dict() for b in self.evidence],
                "method": self.method,
                "signals": dict((k, None if v is None else round(v, 4))
                                for k, v in self.signals.items()),
                "warnings": list(self.warnings)}

    def __repr__(self) -> str:
        """Name, value, confidence and how many evidence boxes back it."""
        return "FieldResult(%s=%r, confidence=%.2f, evidence=%d)" % (
            self.name, self.value, self.confidence, len(self.evidence))


@dataclass
class Extraction(Generic[T]):
    """The result of extracting a schema from a document."""

    model: Optional[T] = None
    fields: Dict[str, FieldResult] = field(default_factory=dict)
    document: Optional[Document] = None
    cost: Cost = field(default_factory=Cost)
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_response: str = ""
    latency_ms: float = 0.0

    def __getitem__(self, name: str) -> FieldResult:
        """``result["total"]`` -- the :class:`FieldResult`.  Raises on unknown names.

        :param name: a schema field name
        :returns: its :class:`FieldResult`, carrying value, confidence and
            evidence together
        :raises KeyError: no such field.  Use :meth:`value` when a missing field
            should be a default rather than an error.
        """
        return self.fields[name]

    def value(self, name: str, default: Any = None) -> Any:
        """The extracted value for ``name``, or ``default`` if missing or null.

        Deliberately conflates "absent" and "null": both mean the document did
        not state it.  Use ``name in result.fields`` to tell them apart.

        :param name: a schema field name
        :param default: returned when the field is absent or its value is
            ``None``
        :returns: the coerced value, already converted to the declared type
        """
        result = self.fields.get(name)
        return default if result is None or result.value is None else result.value

    def confidence(self, name: str) -> float:
        """Confidence for ``name``, or ``0.0`` when the field is absent.

        :param name: a schema field name
        :returns: the fused confidence in ``0.0..1.0``, or ``0.0`` for an unknown
            field.  Uncalibrated unless a ``calibrator`` was fitted on your own
            labelled data -- it ranks reliably, but it is not a probability.
        """
        result = self.fields.get(name)
        return result.confidence if result else 0.0

    @property
    def mean_confidence(self) -> float:
        """Mean confidence over the fields that actually got a value.

        Null fields are excluded rather than scored zero: "not present on this
        document" is a legitimate answer, not a low-confidence one.
        """
        values = [f.confidence for f in self.fields.values() if f.value is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    def low_confidence(self, threshold: float = 0.7) -> List[FieldResult]:
        """Fields a human should look at.  The point of the whole exercise.

        :param threshold: confidence below which a field is queued for review.
            ``0.7`` is a starting point, not a recommendation -- pick it from
            your own reliability curve, where :meth:`EvalReport.regression_vs`
            and :func:`Confidence.reliability_curve` tell you what a given score
            actually buys on your documents.
        :returns: the qualifying :class:`FieldResult` objects, least confident
            first, so a review queue is already in priority order
        """
        return sorted([f for f in self.fields.values() if f.confidence < threshold],
                      key=lambda f: f.confidence)

    @property
    def is_valid(self) -> bool:
        """True when no validator reported an ``error`` (warnings are allowed)."""
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict of the whole result, fields and all."""
        return {
            "model": _as_jsonable(self.model),
            "fields": dict((k, v.to_dict()) for k, v in self.fields.items()),
            "cost": self.cost.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "warnings": list(self.warnings),
            "mean_confidence": self.mean_confidence,
            "valid": self.is_valid,
            "latency_ms": round(self.latency_ms, 1),
            "source_uri": self.document.source_uri if self.document else "",
        }

    def __repr__(self) -> str:
        """Field count, mean confidence, validity and cost."""
        return "Extraction(fields=%d, mean_confidence=%.2f, valid=%s, cost=%s)" % (
            len(self.fields), self.mean_confidence, self.is_valid, self.cost)


def merge_extracted_data(chunks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge per-chunk extractions into one.

    First non-null wins for scalars (earlier pages hold headers and identifiers);
    lists are concatenated with duplicates dropped, because line items are
    spread across pages and each chunk sees only its own.

    :param chunks: the parsed object from each chunk, in document order.  Order
        matters: "first non-null wins" makes the earliest chunk authoritative
        for scalars, which is why chunks must not be reordered or run
        out-of-sequence.
    :returns: one merged dict.  Duplicate list entries are dropped by comparing
        their JSON encoding, so the page repeated by ``overlap_pages`` does not
        double every line item it carried.
    """
    merged: Dict[str, Any] = {}
    for chunk in chunks:
        for key, value in (chunk or {}).items():
            if value is None:
                continue
            if isinstance(value, list):
                existing = merged.setdefault(key, [])
                if not isinstance(existing, list):
                    continue
                for item in value:
                    marker = json.dumps(_as_jsonable(item), sort_keys=True)
                    if marker not in [json.dumps(_as_jsonable(e), sort_keys=True)
                                      for e in existing]:
                        existing.append(item)
            elif key not in merged or merged[key] is None:
                merged[key] = value
    return merged


def extract(doc: Document, schema: SchemaLike, context: str = "",
            client: Optional[LLMClient] = None,
            validators: Optional[Sequence[Validator]] = None,
            system: str = DEFAULT_EXTRACTION_SYSTEM, extra_instructions: str = "",
            max_chars: int = 60000, overlap_pages: int = 1, include_images: bool = False,
            weights: Optional[Mapping[str, float]] = None,
            calibrator: Optional[Calibrator] = None, locate_evidence: bool = True,
            min_evidence_score: float = 0.72, retries: int = 2) -> Extraction:
    """Extract ``schema`` from ``doc``, with per-field confidence and provenance.

    This is the whole domain-side integration surface.  A project supplies its
    schema, its validators and a context string; everything else -- prompting,
    chunking, parsing, coercion, evidence recovery, confidence fusion -- is
    library.

    :param doc: a document that has already been read, so its pages carry spans
        -- see :func:`read`.  An unread document yields empty prompt text and
        no evidence to locate values against.
    :param schema: a pydantic model, a dataclass, or a ``{field: type}`` dict
        such as ``{"invoice_no": str, "total": float}``.  A pydantic model gives
        the model the most to work with, since field descriptions reach the
        prompt.  See :meth:`SchemaAdapter.from_schema`.
    :param context: domain hints pasted into the prompt, e.g.
        ``"Indian private hospital bill, INR"``.  The cheapest accuracy in the
        library -- it is what disambiguates lakh grouping and local date order.
    :param client: an :class:`LLMClient` -- one of :class:`AnthropicClient`,
        or :class:`OpenAIClient`, or :class:`EchoClient` in tests.  Required.
    :param validators: business rules run over the built object.  Each outcome
        feeds the confidence of the fields it names, so a failing sum lowers
        confidence on the amounts rather than only logging a complaint.
    :param system: system prompt; defaults to :data:`DEFAULT_EXTRACTION_SYSTEM`.
        Replace it only if you know what you are giving up -- it is what keeps
        the model from inferring values that are not on the page.
    :param extra_instructions: appended to the user prompt, after the schema and
        the context.  For per-run notes ("this batch has two bills per page"),
        not for domain vocabulary, which belongs in ``context``.
    :param max_chars: characters of document text per model call.  ``60000`` is
        a safe fit for current context windows with room for the schema.  A
        document above it is split into page-aligned chunks and the results
        merged; a warning records that this happened.
    :param overlap_pages: pages repeated at each chunk boundary, so a field
        straddling a page break is not lost.  ``1`` is usually enough; ``0``
        disables it and is only right when pages are known to be independent.
    :param include_images: also send up to 8 page rasters alongside the text.
        Worth it for pages OCR handled badly, and it costs vision tokens on
        every call -- so it is off by default rather than "on just in case".
    :param weights: override the confidence fusion weights; ``None`` uses
        the defaults in :data:`DEFAULT_CONFIDENCE_WEIGHTS`.  The signals and
        what they mean are set out in :func:`Confidence.fuse_confidence`.
    :param calibrator: maps fused scores onto calibrated probabilities.  Without
        one, confidence ranks correctly but is not a probability -- ``0.9`` does
        not mean 90% right.  Fit a :class:`PlattCalibrator` on your own labelled
        set to change that.
    :param locate_evidence: search the document for each value to recover its
        bounding box.  Leave it on: provenance is unrecoverable afterwards, and
        it is also a confidence signal.  Turning it off saves time on very large
        documents at the cost of both.
    :param min_evidence_score: least match score for evidence to count, in
        ``0.0..1.0``, passed to :func:`locate_value`.
    :param retries: extra attempts on transport or parse failure, so ``2`` means
        up to three tries.  Configuration errors and missing dependencies are
        never retried.
    :returns: an :class:`Extraction` carrying the built ``model``, and one
        per-field :class:`FieldResult` with its confidence and evidence, plus
        the accumulated cost, validation issues and warnings
    :raises ConfigError: no ``client`` was given

    Confidence is fused from independent signals and is *uncalibrated* unless a
    ``calibrator`` fitted on your own labelled set is supplied.  See
    :func:`Confidence.fuse_confidence` and :class:`EvalSuite`.
    """
    if client is None:
        raise ConfigError(
            "extract() needs a client; pass docpipe.AnthropicClient(), "
            "docpipe.OpenAIClient(), or docpipe.EchoClient(...) in tests")
    adapter = SchemaAdapter.from_schema(schema)
    result = Extraction(document=doc, cost=Cost.zero())

    with Timer() as timer:
        chunks = chunk_pages(doc, max_chars=max_chars, overlap_pages=overlap_pages)
        if len(chunks) > 1:
            result.warnings.append(
                "document split into %d chunks of <=%d characters" % (len(chunks), max_chars))

        payloads: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = format_document_text(chunk)
            prompt = build_extraction_prompt(adapter, text, context, extra_instructions)
            images = None
            if include_images:
                images = [encode_jpeg(p.raster(), quality=85)
                          for p in chunk.pages if p.has_raster()][:8]

            def call() -> Tuple[str, Cost]:
                """One model call for this chunk, wrapped so it can be retried."""
                return client.complete(prompt, system=system, images=images)

            try:
                raw, cost = retry_call(call, attempts=max(1, retries + 1),
                                       give_up_on=(MissingDependency, ConfigError))
            except Exception as exc:
                result.warnings.append("model call failed: %s" % exc)
                continue
            result.cost = result.cost + cost
            result.raw_response = raw
            try:
                payload = parse_json_lenient(raw)
            except ExtractionError as exc:
                result.warnings.append(str(exc))
                continue
            if isinstance(payload, list):
                payload = payload[0] if payload and isinstance(payload[0], Mapping) else {}
            if isinstance(payload, Mapping):
                payloads.append(dict(payload))

        if not payloads:
            result.warnings.append("no usable model output; every field is null")
        data = merge_extracted_data(payloads)

        # Coerce to declared types before validating, so a business rule sees
        # Decimals and dates rather than whatever the model happened to emit.
        coerced: Dict[str, Any] = {}
        for spec in adapter.fields:
            coerced[spec.name] = adapter.coerce(spec.name, data.get(spec.name))
        for key, value in data.items():
            if key not in coerced:
                coerced[key] = value

        try:
            result.model = adapter.build(coerced)
        except ExtractionError as exc:
            result.warnings.append(str(exc))
            result.model = coerced  # type: ignore[assignment]

        result.issues = run_validators(result.model, validators or [])
        failed_fields: Set[str] = set()
        for issue in result.issues:
            if issue.severity == "error":
                failed_fields.update(issue.fields)
        any_error = any(i.severity == "error" for i in result.issues)

        for spec in adapter.fields:
            result.fields[spec.name] = _build_field_result(
                spec=spec, value=coerced.get(spec.name), raw=data.get(spec.name),
                doc=doc, client_name=getattr(client, "name", "llm"),
                locate=locate_evidence, min_evidence_score=min_evidence_score,
                failed=spec.name in failed_fields, any_error=any_error,
                weights=weights, calibrator=calibrator)

    result.latency_ms = timer.ms
    return result


def _build_field_result(spec: FieldSpec, value: Any, raw: Any, doc: Document,
                        client_name: str, locate: bool, min_evidence_score: float,
                        failed: bool, any_error: bool,
                        weights: Optional[Mapping[str, float]],
                        calibrator: Optional[Calibrator]) -> FieldResult:
    """Assemble one field's value, evidence, signals and fused confidence."""
    result = FieldResult(name=spec.name, value=value, raw=raw, method=client_name)

    if value is None:
        if spec.required:
            result.warnings.append("required field %s not found" % spec.name)
        result.signals = {"validation": 0.0 if failed else None}
        result.confidence = 0.0
        return result

    evidence: List[Evidence] = []
    if locate and not isinstance(value, (list, dict)):
        evidence = locate_value(doc, value, min_score=min_evidence_score)
        result.evidence = merge_bboxes([e.bbox for e in evidence[:3]])
        if not evidence:
            result.warnings.append(
                "value not found in the document text; it may be hallucinated, "
                "or read from a page image rather than the text layer")

    page_prior: Optional[float] = None
    if evidence:
        try:
            page = doc.page(evidence[0].bbox.page)
            page_prior = page_quality_prior(page.quality, evidence[0].bbox, page)
        except KeyError:
            page_prior = None
    elif doc.pages:
        page_prior = round(sum(p.quality.score for p in doc.pages) / len(doc.pages), 4)

    # Cross-read agreement, when a page was read twice (see EnsembleBackend).
    cross: Optional[float] = None
    if evidence:
        values: List[float] = []
        for item in evidence[:3]:
            try:
                page = doc.page(item.bbox.page)
            except KeyError:
                continue
            for span in page.spans_in(item.bbox, min_overlap=0.3):
                agreement = span.meta.get("cross_read_agreement")
                if agreement is not None:
                    values.append(float(agreement))
        if values:
            cross = round(sum(values) / len(values), 4)

    if evidence:
        # How well the extracted value matches text some engine actually read.
        text_support = round(evidence[0].score, 4)
    elif isinstance(value, (list, dict)) or not doc.pages:
        text_support = None          # nothing was looked for, so no evidence
    else:
        text_support = 0.05          # looked for and not found: likely invented

    signals: Dict[str, Optional[float]] = {
        "cross_read_agree": cross,
        "text_support": text_support,
        "backend_conf": backend_confidence_for(doc, evidence) if evidence else None,
        "page_quality": page_prior,
        "format_match": format_match_score(value, spec.type_name),
        # Asymmetric on purpose.  A validator that *failed* naming this field
        # is near-conclusive evidence that something here is wrong, so it is
        # pushed to the clamp floor.  A validator that *passed* is only weakly
        # positive: an arithmetic check is satisfied by any consistent set of
        # wrong numbers, and most fields are not covered by any check at all.
        "validation": 0.02 if failed else (0.65 if not any_error else 0.5),
        "logprob": None,
    }

    result.signals = signals
    fused = fuse_confidence(signals, weights)
    result.confidence = calibrator.predict(fused) if calibrator is not None else fused
    return result


# -- rule-based extraction (no model required) -------------------------------

@dataclass
class FieldRule(object):
    """A label-anchored extraction rule.

    Deterministic, free, auditable, and often more accurate than a model on
    fixed-layout forms.  It also makes an excellent second opinion: running a
    rule set alongside a model gives a genuine cross-read agreement signal at
    no marginal cost.

    :param name: the field this rule fills
    :param label: regex matching the printed label, e.g. ``r"total\\s*amount"``
    :param value_pattern: regex for the value; the first group wins if present
    :param direction: where to look relative to the label -- ``right``, ``below``
                      or ``same_line``
    :param type_name: declared type, used for coercion and format scoring
    """

    name: str
    label: str
    value_pattern: str = r"(.+)"
    direction: str = "right"
    type_name: str = "string"
    max_distance_pt: float = 260.0
    occurrence: int = 0

    def compiled_label(self) -> Pattern:
        """The label pattern, compiled case-insensitively."""
        return re.compile(self.label, re.IGNORECASE)

    def compiled_value(self) -> Pattern:
        """The value pattern, compiled case-insensitively."""
        return re.compile(self.value_pattern, re.IGNORECASE)


def extract_with_rules(doc: Document, rules: Sequence[FieldRule]) -> Dict[str, FieldResult]:
    """Extract fields by anchoring on printed labels.  No model, no network.

    The deterministic alternative to :func:`extract`, for fields that sit beside
    a fixed printed label.  Free, instant and exactly repeatable -- but it only
    works where the layout is stable, so the usual arrangement is rules for the
    reliable fields and a model for the rest.

    :param doc: a document that has been read, so its pages carry spans
    :param rules: the label-anchored rules to apply; see :class:`FieldRule`
    :returns: a :class:`FieldResult` per rule, keyed by ``rule.name``.  Every
        rule produces an entry: one that matched nothing reports that in its
        ``warnings`` and carries a ``None`` value, so a caller can tell "absent"
        from "never looked for".
    """
    out: Dict[str, FieldResult] = {}
    for rule in rules:
        out[rule.name] = _apply_rule(doc, rule)
    return out


def _apply_rule(doc: Document, rule: FieldRule) -> FieldResult:
    """Apply one label-anchored rule across the document.

    Scans lines for the label, then reads the value from beside or below it,
    honouring ``rule.occurrence`` when a label repeats (a per-page "Total" on
    a multi-page bill).  Always returns a :class:`FieldResult` -- a rule that
    matched nothing reports that in ``warnings`` rather than raising.
    """
    label_re = rule.compiled_label()
    value_re = rule.compiled_value()
    result = FieldResult(name=rule.name, method="rules")
    hits = 0

    for page in doc.pages:
        lines = _group_spans_into_lines(page)
        for line_index, line_spans in enumerate(lines):
            line_text = " ".join(s.text for s in line_spans)
            match = label_re.search(line_text)
            if not match:
                continue

            candidates: List[Tuple[str, List[TextSpan]]] = []
            if rule.direction in ("right", "same_line"):
                tail = line_text[match.end():].strip(" :\t-")
                if tail:
                    label_end_x = _approx_x_for_offset(line_spans, match.end())
                    right_spans = [s for s in line_spans if s.bbox.x0 >= label_end_x - 1.0]
                    candidates.append((tail, right_spans or line_spans))
            if rule.direction == "below" and line_index + 1 < len(lines):
                below = lines[line_index + 1]
                label_box = line_spans[0].bbox
                near = [s for s in below
                        if abs(s.bbox.y0 - label_box.y1) <= rule.max_distance_pt]
                if near:
                    candidates.append((" ".join(s.text for s in near), near))

            for text, spans in candidates:
                value_match = value_re.search(text)
                if not value_match:
                    continue
                hits += 1
                if hits <= rule.occurrence:
                    continue
                raw = (value_match.group(1) if value_match.groups()
                       else value_match.group(0)).strip()
                value = coerce_value(raw, rule.type_name)
                if value is None:
                    continue
                boxes = merge_bboxes([s.bbox for s in spans])
                confidences = [s.confidence for s in spans if s.confidence is not None]
                signals: Dict[str, Optional[float]] = {
                    "backend_conf": (sum(confidences) / len(confidences)
                                     if confidences else None),
                    "page_quality": page_quality_prior(page.quality,
                                                       boxes[0] if boxes else None, page),
                    "format_match": format_match_score(value, rule.type_name),
                    # A label anchor is strong evidence that the value sits
                    # where it should, but it is a single read, so it cannot
                    # claim cross-read agreement.
                    "text_support": 0.9,
                    "cross_read_agree": None,
                    "validation": None,
                }
                result.value = value
                result.raw = raw
                result.evidence = boxes
                result.signals = signals
                result.confidence = fuse_confidence(signals)
                return result

    result.warnings.append("label %r not found" % rule.label)
    return result


def _approx_x_for_offset(line_spans: Sequence[TextSpan], offset: int) -> float:
    """Approximate the x coordinate of a character offset within a joined line."""
    position = 0
    for span in line_spans:
        end = position + len(span.text)
        if offset <= end:
            fraction = _safe_div(offset - position, max(1, len(span.text)))
            return span.bbox.x0 + fraction * span.bbox.width
        position = end + 1
    return line_spans[-1].bbox.x1 if line_spans else 0.0


# =============================================================================
# 13. Layer 5 -- the evaluation harness
# =============================================================================
#
# If only one component of this library is built, it should be this one.
#
# Without a harness, "did that change help?" is answered by eyeballing a handful
# of documents.  That is not a measurement.  Every accuracy claim is anecdote,
# regressions from a model version changing underneath you are invisible, and
# confidence cannot be calibrated at all.
#
# It also makes the labelled datasets themselves shared, versioned assets --
# and ground-truth labelling is the single most expensive input to this kind of
# work.  Doing it once per document class and reusing it across projects and
# across model migrations is a large and permanent saving.


@dataclass
class EvalCase(object):
    """One labelled document: the file, the truth, and how it is grouped."""

    case_id: str
    path: str
    truth: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return {"case_id": self.case_id, "path": self.path,
                "truth": _as_jsonable(self.truth), "tags": list(self.tags)}


@dataclass
class FieldOutcome(object):
    """How one field of one document scored under one pipeline."""

    case_id: str
    pipeline: str
    field: str
    expected: Any = None
    predicted: Any = None
    exact: bool = False
    fuzzy: float = 0.0
    within_tolerance: bool = False
    confidence: float = 0.0
    has_evidence: bool = False
    page_verdict: str = "unknown"
    type_name: str = "string"

    @property
    def correct(self) -> bool:
        """Exact match, or numerically/date equal within tolerance."""
        return bool(self.exact or self.within_tolerance)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict, including the derived ``correct`` flag."""
        return {"case_id": self.case_id, "pipeline": self.pipeline, "field": self.field,
                "expected": _as_jsonable(self.expected),
                "predicted": _as_jsonable(self.predicted),
                "exact": self.exact, "fuzzy": round(self.fuzzy, 4),
                "within_tolerance": self.within_tolerance, "correct": self.correct,
                "confidence": round(self.confidence, 4),
                "has_evidence": self.has_evidence, "page_verdict": self.page_verdict,
                # Without this, a reloaded report cannot tell which outcomes
                # were numeric, and numeric_tolerance silently reads as 0.
                "type_name": self.type_name}


@dataclass
class CaseResult(object):
    """Everything one pipeline produced for one case."""

    case_id: str
    pipeline: str
    outcomes: List[FieldOutcome] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    latency_ms: float = 0.0
    error: str = ""
    page_verdict: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict of one case's outcomes, cost and timing."""
        return {"case_id": self.case_id, "pipeline": self.pipeline,
                "outcomes": [o.to_dict() for o in self.outcomes],
                "cost": self.cost.to_dict(),
                # Full precision: a saved report is compared against a later
                # run by regression_vs, and rounding here would make a report
                # differ from itself and fire a spurious latency regression.
                "latency_ms": round(self.latency_ms, 6),
                "error": self.error, "page_verdict": self.page_verdict}


def compare_values(expected: Any, predicted: Any, type_name: str = "string",
                   numeric_tolerance: float = 0.005,
                   fuzzy_threshold: float = 0.92) -> Tuple[bool, float, bool]:
    """Score a prediction against ground truth.

    Returns ``(exact, fuzzy_similarity, within_tolerance)``.

    Three numbers rather than one because they answer different questions.
    Exact match is what a downstream system needs.  Fuzzy similarity shows
    whether a miss was a single transposed digit or complete garbage -- the
    difference between a tuning problem and a broken pipeline.  Tolerance
    matching is what a human reviewer would call correct.

    :param expected: the ground-truth value.  Two ``None`` values count as a
        correct match -- the pipeline agreeing that a field is absent is right,
        not a miss.
    :param predicted: what the pipeline produced
    :param type_name: how to compare -- ``"number"``, ``"integer"``, ``"amount"``,
        ``"date"``, ``"array"``, ``"object"``, or ``"string"``.  Drives the
        comparison, so a truth value typed as a string is compared textually even
        when it looks numeric; that is what ``EvalSuite.field_types`` overrides.
    :param numeric_tolerance: relative slack for numbers, so ``0.005`` allows half
        a percent.  Applied to the larger magnitude of the pair.
    :param fuzzy_threshold: similarity at or above which a string counts as
        within tolerance, in ``0.0..1.0``.  ``0.9`` treats a single wrong
        character in a long value as close enough for a human.
    :returns: ``(exact, fuzzy_similarity, within_tolerance)``
    """
    if expected is None and predicted is None:
        return True, 1.0, True
    if expected is None or predicted is None:
        return False, 0.0, False

    kind = (type_name or "string").lower()
    if kind in ("number", "float", "decimal", "integer", "int", "amount", "money"):
        a = parse_amount(str(expected))
        b = parse_amount(str(predicted))
        if a is None or b is None:
            return False, 0.0, False
        exact = a == b
        scale = max(abs(float(a)), abs(float(b)), 1e-9)
        within = abs(float(a) - float(b)) <= numeric_tolerance * scale
        return exact, 1.0 if within else 0.0, within
    if kind in ("date", "datetime"):
        a = parse_date(str(expected))
        b = parse_date(str(predicted))
        if a is None or b is None:
            return False, 0.0, False
        return a == b, 1.0 if a == b else 0.0, a == b
    if kind in ("boolean", "bool"):
        equal = parse_bool(str(expected)) == parse_bool(str(predicted))
        return equal, 1.0 if equal else 0.0, equal
    if kind == "array":
        expected_list = expected if isinstance(expected, (list, tuple)) else [expected]
        predicted_list = predicted if isinstance(predicted, (list, tuple)) else [predicted]
        keys_a = [json.dumps(_as_jsonable(x), sort_keys=True) for x in expected_list]
        keys_b = [json.dumps(_as_jsonable(x), sort_keys=True) for x in predicted_list]
        ratio = difflib.SequenceMatcher(None, keys_a, keys_b).ratio()
        return keys_a == keys_b, round(ratio, 4), ratio >= fuzzy_threshold

    key_a = _compare_key(str(expected))
    key_b = _compare_key(str(predicted))
    exact = key_a == key_b
    ratio = similarity(key_a, key_b)
    return exact, round(ratio, 4), ratio >= fuzzy_threshold


PipelineFn = Callable[[str], Extraction]

#: Metrics :meth:`EvalSuite.run` knows how to compute.
KNOWN_METRICS = (
    "field_exact", "field_fuzzy", "numeric_tolerance", "field_recall",
    "evidence_coverage", "confidence_calibration", "brier", "cost_per_doc",
    "latency_p50", "latency_p95", "error_rate",
)


class EvalSuite(object):
    """A labelled dataset plus the machinery to run pipelines against it.

    ::

        suite = dp.EvalSuite.from_dir("datasets/hospital_bills_v3")
        report = suite.run({"baseline": pipeline_a, "candidate": pipeline_b})
        print(report.summary())
        report.regression_vs("reports/release-2026-08.json")
    """

    def __init__(self, cases: Optional[Sequence[EvalCase]] = None, name: str = "",
                 field_types: Optional[Mapping[str, str]] = None) -> None:
        """Build a suite from cases already in memory.

        :param cases: the labelled documents
        :param name: shown in reports; :meth:`from_dir` defaults it to the folder
        :param field_types: declared type per field name, overriding the type
            inferred from each ground-truth value -- needed when a truth value is
            a string that should nonetheless be compared as a number or a date
        """
        self.cases = list(cases or [])
        self.name = name
        self.field_types = dict(field_types or {})

    @classmethod
    def from_dir(cls, directory: str, truth_suffix: str = ".json",
                 name: str = "") -> EvalSuite:
        """Load a dataset laid out as ``<stem>.pdf`` beside ``<stem>.json``.

        The JSON may be the truth mapping directly, or ``{"truth": {...},
        "tags": [...]}`` when a case needs grouping metadata.

        :param directory: folder holding the documents and their truth files
        :param truth_suffix: extension identifying truth files.  Everything with
            this suffix becomes a case; the document is whichever sibling shares
            the stem, found by extension.
        :param name: label shown in reports; defaults to the folder's basename
        :returns: the loaded suite
        :raises ConfigError: the directory does not exist.  A truth file with no
            matching document is recorded on the case rather than raising, so
            one stray file does not block the suite.
        """
        if not os.path.isdir(directory):
            raise ConfigError("no such dataset directory: %s" % directory)
        cases = []
        for entry in sorted(os.listdir(directory)):
            if not entry.endswith(truth_suffix):
                continue
            stem = entry[:-len(truth_suffix)]
            truth_path = os.path.join(directory, entry)
            document = None
            for candidate in sorted(os.listdir(directory)):
                if candidate == entry or not candidate.startswith(stem + "."):
                    continue
                document = os.path.join(directory, candidate)
                break
            if document is None:
                logger.warning("ground truth %s has no matching document; skipping", entry)
                continue
            with open(truth_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            truth = payload.get("truth", payload) if isinstance(payload, dict) else {}
            tags = payload.get("tags", []) if isinstance(payload, dict) else []
            cases.append(EvalCase(case_id=stem, path=document, truth=dict(truth),
                                  tags=list(tags)))
        if not cases:
            raise ConfigError("no labelled cases found in %s" % directory)
        return cls(cases, name=name or os.path.basename(os.path.abspath(directory)))

    def add(self, case: EvalCase) -> EvalSuite:
        """Append a case.  Returns ``self``, so cases chain.

        :param case: the labelled case to add
        :returns: ``self``, so calls chain
        """
        self.cases.append(case)
        return self

    def filter(self, predicate: Callable[[EvalCase], bool]) -> EvalSuite:
        """A new suite holding only the cases satisfying ``predicate``.

        :param predicate: ``Callable[[EvalCase], bool]``, e.g.
            ``lambda c: "handwritten" in c.tags`` to measure a hard subset
            separately -- an aggregate number hides exactly the regressions
            worth catching.
        :returns: a new suite sharing the surviving cases, keeping the original's
            name and field types
        """
        return EvalSuite([c for c in self.cases if predicate(c)], self.name, self.field_types)

    def __len__(self) -> int:
        """Number of labelled cases."""
        return len(self.cases)

    def run(self, pipelines: Mapping[str, PipelineFn],
            metrics: Optional[Sequence[str]] = None, max_workers: int = 0,
            numeric_tolerance: float = 0.005,
            on_case: Optional[Callable[[str, str], None]] = None) -> EvalReport:
        """Run each pipeline over every case and score the results.

        A pipeline that raises on one document is recorded as an error for that
        case and the run continues -- a suite that aborts on the first bad
        scan measures nothing.

        :param pipelines: name to callable, each taking a document path and
            returning an :class:`Extraction`.  A configured :class:`Pipeline` is
            already such a callable.  Pass two to A/B them against identical
            cases::

                report = suite.run({"baseline": current, "candidate": tuned})

        :param metrics: which of :data:`KNOWN_METRICS` to compute; ``None``
            computes all of them.
        :param max_workers: threads across cases.  ``0`` runs serially.  Cases
            are independent, so this scales well -- but it multiplies your
            concurrent API calls by the same factor.
        :param numeric_tolerance: relative slack for numeric comparison, passed
            to :func:`compare_values`.  ``0.005`` is half a percent.
        :param on_case: called as ``on_case(pipeline_name, case_id)`` for
            progress reporting.
        :returns: an :class:`EvalReport` holding every per-field outcome, not
            just the aggregates -- which is what lets the CI gate
            of :meth:`EvalReport.regression_vs` name what broke.
        :raises ConfigError: a metric name is not in :data:`KNOWN_METRICS`
        """
        chosen = list(metrics or KNOWN_METRICS)
        unknown = [m for m in chosen if m not in KNOWN_METRICS]
        if unknown:
            raise ConfigError("unknown metric(s): %s (known: %s)"
                              % (", ".join(unknown), ", ".join(KNOWN_METRICS)))
        report = EvalReport(suite=self.name, metrics=chosen,
                            created_at=_utcnow(), case_count=len(self.cases))

        for pipeline_name, pipeline in pipelines.items():
            def run_case(case: EvalCase, _pipeline: Any = pipeline,
                         _name: str = pipeline_name) -> CaseResult:
                """Score one case, binding the pipeline for the thread pool."""
                return self._run_case(case, _pipeline, _name, numeric_tolerance)

            results = _map_maybe_parallel(run_case, self.cases, max_workers, "eval")
            report.results[pipeline_name] = list(results)
            if on_case is not None:
                for case in self.cases:
                    on_case(pipeline_name, case.case_id)
        return report

    def _run_case(self, case: EvalCase, pipeline: PipelineFn, pipeline_name: str,
                  numeric_tolerance: float) -> CaseResult:
        """Run one pipeline over one case and score every truth field.

        A pipeline that raises is captured into ``CaseResult.error`` with its
        latency intact, so the run continues and the failure still shows up in
        ``error_rate`` rather than vanishing.
        """
        result = CaseResult(case_id=case.case_id, pipeline=pipeline_name)
        with Timer() as timer:
            try:
                extraction = pipeline(case.path)
            except Exception as exc:
                logger.exception("pipeline %s failed on %s", pipeline_name, case.case_id)
                result.error = "%s: %s" % (type(exc).__name__, exc)
                result.latency_ms = timer.ms
                return result
        result.latency_ms = timer.ms
        result.cost = extraction.cost
        if extraction.document and extraction.document.pages:
            verdicts = [str(p.quality.verdict) for p in extraction.document.pages]
            # Worst page governs: a bundle is as readable as its worst page.
            for level in ("unreadable", "degraded", "clean"):
                if level in verdicts:
                    result.page_verdict = level
                    break

        for field_name, expected in case.truth.items():
            field_result = extraction.fields.get(field_name)
            predicted = field_result.value if field_result else None
            type_name = self.field_types.get(field_name) or _infer_type_name(expected)
            exact, fuzzy, within = compare_values(expected, predicted, type_name,
                                                  numeric_tolerance)
            result.outcomes.append(FieldOutcome(
                case_id=case.case_id, pipeline=pipeline_name, field=field_name,
                expected=expected, predicted=predicted, exact=exact, fuzzy=fuzzy,
                within_tolerance=within,
                confidence=field_result.confidence if field_result else 0.0,
                has_evidence=bool(field_result and field_result.evidence),
                page_verdict=result.page_verdict, type_name=type_name))
        return result


def _infer_type_name(value: Any) -> str:
    """Guess a field's type from its ground-truth value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float, decimal.Decimal)):
        return "number"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, (_dt.date, _dt.datetime)):
        return "date"
    if isinstance(value, str) and parse_date(value) and re.search(r"\d{1,4}[-/.]\d", value):
        return "date"
    return "string"


@dataclass
class EvalReport(object):
    """Scored results for one or more pipelines over one suite."""

    suite: str = ""
    metrics: List[str] = field(default_factory=list)
    created_at: str = ""
    case_count: int = 0
    results: Dict[str, List[CaseResult]] = field(default_factory=dict)

    # -- access -----------------------------------------------------------
    def pipelines(self) -> List[str]:
        """Names of the pipelines in this report, sorted."""
        return sorted(self.results)

    def outcomes(self, pipeline: str) -> List[FieldOutcome]:
        """Every field outcome for ``pipeline``, across all cases.

        :param pipeline: the pipeline name to pull results for
        :returns: every :class:`FieldOutcome` from every case, flattened.  Empty
            for an unknown name rather than raising, so a caller iterating over
            pipeline names cannot trip on one that produced nothing.
        """
        out: List[FieldOutcome] = []
        for case in self.results.get(pipeline, []):
            out.extend(case.outcomes)
        return out

    # -- metrics ----------------------------------------------------------
    def compute(self, pipeline: str) -> Dict[str, float]:
        """All requested metrics for one pipeline.

        :param pipeline: the pipeline name to score
        :returns: metric name to value, covering whichever entries
            of :data:`KNOWN_METRICS` the run requested.  ``cost_per_doc`` reads
            ``0.0`` until :func:`Pricing.set_pricing` is called, though token
            counts behind it are real.
        """
        cases = self.results.get(pipeline, [])
        outcomes = self.outcomes(pipeline)
        numeric = [o for o in outcomes
                   if o.type_name in ("number", "integer", "int", "float", "decimal")]
        expected_present = [o for o in outcomes if o.expected is not None]
        latencies = [c.latency_ms for c in cases]
        scores = [o.confidence for o in outcomes]
        labels = [o.correct for o in outcomes]

        values = {
            "field_exact": _mean([1.0 if o.exact else 0.0 for o in outcomes]),
            "field_fuzzy": _mean([o.fuzzy for o in outcomes]),
            "numeric_tolerance": _mean([1.0 if o.within_tolerance else 0.0 for o in numeric]),
            "field_recall": _mean([1.0 if o.predicted is not None else 0.0
                                   for o in expected_present]),
            "evidence_coverage": _mean([1.0 if o.has_evidence else 0.0
                                        for o in outcomes if o.predicted is not None]),
            "confidence_calibration": expected_calibration_error(scores, labels),
            "brier": brier_score(scores, labels),
            "cost_per_doc": _mean([c.cost.amount for c in cases]),
            "latency_p50": percentile(latencies, 50),
            "latency_p95": percentile(latencies, 95),
            "error_rate": _mean([1.0 if c.error else 0.0 for c in cases]),
        }
        return dict((m, round(values[m], 4)) for m in self.metrics if m in values)

    def by_field(self, pipeline: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Per-field accuracy -- which fields degraded, not just the average.

        :param pipeline: which pipeline's results to use; ``None`` takes the only
            one, which is unambiguous for a single-pipeline report


        The aggregate hides everything that matters.  A pipeline can gain two
        points overall while losing eight on the total amount, which is the
        only field anyone downstream actually cares about.
        """
        pipeline = pipeline or (self.pipelines()[0] if self.pipelines() else "")
        grouped: Dict[str, List[FieldOutcome]] = {}
        for outcome in self.outcomes(pipeline):
            grouped.setdefault(outcome.field, []).append(outcome)
        out = {}
        for name, items in sorted(grouped.items()):
            out[name] = {
                "n": len(items),
                "exact": round(_mean([1.0 if o.exact else 0.0 for o in items]), 4),
                "correct": round(_mean([1.0 if o.correct else 0.0 for o in items]), 4),
                "fuzzy": round(_mean([o.fuzzy for o in items]), 4),
                "recall": round(_mean([1.0 if o.predicted is not None else 0.0
                                       for o in items]), 4),
                "mean_confidence": round(_mean([o.confidence for o in items]), 4),
            }
        return out

    def by_page_quality(self, pipeline: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Accuracy split by page condition -- did we only improve on clean pages?

        :param pipeline: which pipeline's results to use; ``None`` takes the only
            one, which is unambiguous for a single-pipeline report


        A change that lifts the average by improving pages that were already
        fine, while leaving the degraded ones untouched, has not solved the
        problem it was meant to solve.  This is the split that catches that.
        """
        pipeline = pipeline or (self.pipelines()[0] if self.pipelines() else "")
        grouped: Dict[str, List[FieldOutcome]] = {}
        for outcome in self.outcomes(pipeline):
            grouped.setdefault(outcome.page_verdict, []).append(outcome)
        out = {}
        for verdict, items in sorted(grouped.items()):
            out[verdict] = {
                "n": len(items),
                "exact": round(_mean([1.0 if o.exact else 0.0 for o in items]), 4),
                "correct": round(_mean([1.0 if o.correct else 0.0 for o in items]), 4),
                "mean_confidence": round(_mean([o.confidence for o in items]), 4),
            }
        return out

    def confidence_curve(self, pipeline: Optional[str] = None,
                         bins: int = 10) -> List[Dict[str, float]]:
        """Is a 0.9 right 90% of the time?  This is the answer.

        :param pipeline: which pipeline's results to use; ``None`` takes the only
            one, which is unambiguous for a single-pipeline report
        :param bins: equal-width confidence bins; ``10`` is conventional
        :returns: the reliability curve as computed
            by :func:`Confidence.reliability_curve` -- one dict per bin with
            ``count``, ``mean_confidence``, ``accuracy`` and ``gap``
        """
        pipeline = pipeline or (self.pipelines()[0] if self.pipelines() else "")
        outcomes = self.outcomes(pipeline)
        return reliability_curve([o.confidence for o in outcomes],
                                 [o.correct for o in outcomes], bins)

    def fit_calibrator(self, pipeline: Optional[str] = None,
                       kind: str = "platt") -> Calibrator:
        """Fit a calibrator on this report, ready to pass to :func:`extract`.

        :param pipeline: which pipeline's results to use; ``None`` takes the only
            one, which is unambiguous for a single-pipeline report
        :param kind: ``"platt"`` (default) or ``"isotonic"``.  Platt fits two
            parameters and works from a few hundred labelled fields; isotonic
            assumes no shape but needs a few thousand not to overfit.
        :returns: a fitted :class:`Calibrator`.  Serialise it with ``to_dict``
            and pass it to :func:`extract` or :class:`Pipeline` to make the
            confidence numbers mean what they say.
        :raises ConfigError: too few samples for the chosen kind

        This closes the loop: measure, calibrate, ship the calibrator, and the
        confidence numbers a reviewer sees start meaning what they say.
        """
        pipeline = pipeline or (self.pipelines()[0] if self.pipelines() else "")
        outcomes = self.outcomes(pipeline)
        scores = [o.confidence for o in outcomes]
        labels = [o.correct for o in outcomes]
        if kind == "platt":
            return PlattCalibrator().fit(scores, labels)
        if kind == "isotonic":
            return IsotonicCalibrator().fit(scores, labels)
        raise ConfigError("unknown calibrator kind %r" % kind)

    def errors(self, pipeline: Optional[str] = None) -> List[Tuple[str, str]]:
        """``(case_id, error)`` for every case the pipeline failed on.

        :param pipeline: which pipeline's results to use; ``None`` takes the only
            one, which is unambiguous for a single-pipeline report
        :returns: ``(case_id, error_message)`` pairs.  Empty is what you want;
            a non-empty list means those cases scored zero for reasons that have
            nothing to do with extraction accuracy.
        """
        pipeline = pipeline or (self.pipelines()[0] if self.pipelines() else "")
        return [(c.case_id, c.error) for c in self.results.get(pipeline, []) if c.error]

    def worst_cases(self, pipeline: Optional[str] = None,
                    limit: int = 10) -> List[Tuple[str, float]]:
        """Cases with the lowest field accuracy -- where to look first.

        :param pipeline: which pipeline's results to use; ``None`` takes the only
            one, which is unambiguous for a single-pipeline report
        :param limit: how many to return, worst first
        :returns: ``(case_id, accuracy)`` pairs.  Read these before tuning
            anything -- an aggregate that moved usually moved because of a
            handful of documents with a shared cause.
        """
        pipeline = pipeline or (self.pipelines()[0] if self.pipelines() else "")
        scored = []
        for case in self.results.get(pipeline, []):
            if not case.outcomes:
                scored.append((case.case_id, 0.0))
                continue
            scored.append((case.case_id,
                           round(_mean([1.0 if o.correct else 0.0 for o in case.outcomes]), 4)))
        scored.sort(key=lambda kv: kv[1])
        return scored[:limit]

    # -- comparison -------------------------------------------------------
    def compare(self, baseline_pipeline: str, candidate_pipeline: str) -> Dict[str, Any]:
        """Metric deltas and per-field regressions between two pipelines.

        :param baseline_pipeline: the name to measure against
        :param candidate_pipeline: the name being evaluated
        :returns: a dict of metric deltas plus the per-field accuracy changes,
            signed so that positive means the candidate improved.  Both names
            must be present in this report -- comparing across two separate runs
            is :meth:`EvalReport.regression_vs`.
        """
        base = self.compute(baseline_pipeline)
        candidate = self.compute(candidate_pipeline)
        deltas = dict((k, round(candidate.get(k, 0.0) - base.get(k, 0.0), 4))
                      for k in set(base) | set(candidate))
        base_fields = self.by_field(baseline_pipeline)
        candidate_fields = self.by_field(candidate_pipeline)
        regressions = []
        improvements = []
        for name in sorted(set(base_fields) | set(candidate_fields)):
            before = base_fields.get(name, {}).get("correct", 0.0)
            after = candidate_fields.get(name, {}).get("correct", 0.0)
            delta = round(after - before, 4)
            if delta < -1e-9:
                regressions.append({"field": name, "before": before,
                                    "after": after, "delta": delta})
            elif delta > 1e-9:
                improvements.append({"field": name, "before": before,
                                     "after": after, "delta": delta})
        regressions.sort(key=lambda r: r["delta"])
        improvements.sort(key=lambda r: -r["delta"])
        return {"baseline": baseline_pipeline, "candidate": candidate_pipeline,
                "baseline_metrics": base, "candidate_metrics": candidate,
                "deltas": deltas, "regressions": regressions,
                "improvements": improvements,
                "regressed": bool(regressions)}

    #: Metrics that are lower-is-better.  A rise in these is a regression.
    LOWER_IS_BETTER = frozenset(["confidence_calibration", "brier", "cost_per_doc",
                                 "latency_p50", "latency_p95", "error_rate"])

    #: Metrics that are unbounded (milliseconds, currency) rather than rates in
    #: ``[0, 1]``.  For these ``tolerance`` is read as a *fraction*, because an
    #: absolute tolerance that is sensible for an accuracy rate is meaningless
    #: for a latency -- 0.05 ms of scheduler noise would fail every build.
    RELATIVE_METRICS = frozenset(["cost_per_doc", "latency_p50", "latency_p95"])

    def regression_vs(self, baseline: Union[str, EvalReport],
                      pipeline: Optional[str] = None, tolerance: float = 0.0,
                      metrics: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Compare against a saved report -- the CI gate.

        ``baseline`` is a path to a report saved by :meth:`save`, or a report
        object.  A metric that moved by less than ``tolerance`` is treated as
        noise rather than a regression.

        ``tolerance`` is read as an absolute delta for rate metrics and as a
        *fraction* for the unbounded ones (see :attr:`RELATIVE_METRICS`), so a
        single number like ``0.02`` means "two accuracy points, or two percent
        slower" -- both of which are what a person means by that number.

        Timing metrics vary between runs on shared CI hardware regardless of
        the code.  Pass ``metrics=["field_exact", "numeric_tolerance"]`` to gate
        on accuracy alone when that matters more than a stable wall clock.

        :param baseline: a path to a report saved by :meth:`save`, or else
            an :class:`EvalReport` already in memory
        :param pipeline: which pipeline in *this* report to compare; ``None``
            takes the first.  The baseline is matched by the same name, falling
            back to its own first pipeline.
        :param tolerance: movement treated as noise.  Absolute for rate metrics,
            fractional for the unbounded ones, so ``0.02`` means "two accuracy
            points, or two percent slower".  ``0.0`` gates on any movement at
            all, which is only realistic for a deterministic suite.
        :param metrics: restrict the gate to these metric names; ``None``
            compares every metric present in either report.
        :returns: a dict with the per-metric before/after/delta and a
            ``regressions`` list.  An empty ``regressions`` list is the pass
            condition -- which is what ``docpipe eval --baseline`` exits
            non-zero on.
        :raises ConfigError: the baseline report contains no pipelines
        """
        other = EvalReport.load(baseline) if isinstance(baseline, str) else baseline
        pipeline = pipeline or (self.pipelines()[0] if self.pipelines() else "")
        base_pipeline = pipeline if pipeline in other.results else (
            other.pipelines()[0] if other.pipelines() else "")
        if not base_pipeline:
            raise ConfigError("baseline report has no pipelines to compare against")

        base = other.compute(base_pipeline)
        current = self.compute(pipeline)
        selected = set(metrics) if metrics else (set(base) | set(current))
        regressions = []
        for name in sorted((set(base) | set(current)) & selected):
            before = base.get(name, 0.0)
            after = current.get(name, 0.0)
            delta = round(after - before, 4)
            if name in self.RELATIVE_METRICS:
                allowed = abs(before) * tolerance
            else:
                allowed = tolerance
            # For lower-is-better metrics the sign of a regression flips.
            worse = (delta > allowed) if name in self.LOWER_IS_BETTER else \
                    (delta < -allowed)
            if worse:
                regressions.append({"metric": name, "before": before,
                                    "after": after, "delta": delta})
        return {"baseline_report": getattr(other, "created_at", ""),
                "baseline_pipeline": base_pipeline, "pipeline": pipeline,
                "baseline_metrics": base, "current_metrics": current,
                "regressions": regressions, "regressed": bool(regressions)}

    # -- output -----------------------------------------------------------
    def summary(self) -> str:
        """A human-readable table, for a terminal or a CI log."""
        lines = ["eval suite: %s   cases: %d   %s"
                 % (self.suite or "(unnamed)", self.case_count, self.created_at)]
        names = self.pipelines()
        if not names:
            return lines[0] + "\n(no results)"
        metrics = [m for m in self.metrics]
        width = max([len(m) for m in metrics] + [22])
        header = "  %-*s" % (width, "metric") + "".join("%14s" % n[:14] for n in names)
        lines.append("")
        lines.append(header)
        lines.append("  " + "-" * (width + 14 * len(names)))
        computed = dict((n, self.compute(n)) for n in names)
        for metric in metrics:
            row = "  %-*s" % (width, metric)
            for name in names:
                value = computed[name].get(metric)
                row += "%14s" % ("-" if value is None else "%.4f" % value)
            lines.append(row)
        for name in names:
            failures = self.errors(name)
            if failures:
                lines.append("")
                lines.append("  %s: %d case(s) errored, e.g. %s: %s"
                             % (name, len(failures), failures[0][0], failures[0][1][:80]))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict: the computed metrics plus every raw outcome."""
        return {"suite": self.suite, "metrics": list(self.metrics),
                "created_at": self.created_at, "case_count": self.case_count,
                "computed": dict((n, self.compute(n)) for n in self.pipelines()),
                "results": dict((n, [c.to_dict() for c in cases])
                                for n, cases in self.results.items())}

    def save(self, path: str) -> str:
        """Write the report to ``path``, creating parent directories.

        Returns ``path``.  This is the file :meth:`regression_vs` reads as a
        baseline, so it is worth committing alongside a release.

        :param path: destination file; parent directories are created, and an
            existing file is overwritten
        :returns: ``path``, so it can be used inline
        """
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(to_json(self.to_dict()))
        return path

    @classmethod
    def load(cls, path: str) -> EvalReport:
        """Read a report back from a file written by :meth:`save`.

        :param path: a file written by :meth:`save`
        :returns: the reconstructed report, with every per-field outcome intact
            so the comparison can name individual fields
        :raises OSError: the file could not be read
        :raises ValueError: the file is not valid JSON
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        report = cls(suite=payload.get("suite", ""),
                     metrics=list(payload.get("metrics") or []),
                     created_at=payload.get("created_at", ""),
                     case_count=int(payload.get("case_count", 0)))
        for name, cases in (payload.get("results") or {}).items():
            loaded = []
            for entry in cases:
                case = CaseResult(case_id=entry.get("case_id", ""), pipeline=name,
                                  latency_ms=float(entry.get("latency_ms", 0.0)),
                                  error=entry.get("error", ""),
                                  page_verdict=entry.get("page_verdict", "unknown"))
                cost = entry.get("cost") or {}
                case.cost = Cost(currency=cost.get("currency", "USD"),
                                 amount=float(cost.get("amount", 0.0)),
                                 input_tokens=int(cost.get("input_tokens", 0)),
                                 output_tokens=int(cost.get("output_tokens", 0)),
                                 calls=int(cost.get("calls", 0)))
                for o in entry.get("outcomes") or []:
                    case.outcomes.append(FieldOutcome(
                        case_id=o.get("case_id", ""), pipeline=name,
                        field=o.get("field", ""), expected=o.get("expected"),
                        predicted=o.get("predicted"), exact=bool(o.get("exact")),
                        fuzzy=float(o.get("fuzzy", 0.0)),
                        within_tolerance=bool(o.get("within_tolerance")),
                        confidence=float(o.get("confidence", 0.0)),
                        has_evidence=bool(o.get("has_evidence")),
                        page_verdict=o.get("page_verdict", "unknown"),
                        type_name=o.get("type_name", "string")))
                loaded.append(case)
            report.results[name] = loaded
        return report

    def __repr__(self) -> str:
        """Suite name, pipelines and case count."""
        return "EvalReport(suite=%r, pipelines=%s, cases=%d)" % (
            self.suite, self.pipelines(), self.case_count)


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean, or ``0.0`` for an empty sequence."""
    return float(sum(values)) / len(values) if values else 0.0


# =============================================================================
# 14. The whole pipeline, as one configurable object
# =============================================================================


@dataclass
class Pipeline(object):
    """Ingest -> preprocess -> read -> extract, as data.

    Serialisable, so a pipeline configuration can be recorded in an eval report
    and reproduced exactly.  Every stage is optional: drop ``schema`` to get a
    text-extraction pipeline, drop ``policy`` to measure the no-preprocessing
    baseline.
    """

    #: Chooses preprocessing ops per page from its measured quality.  Also
    #: :func:`Policies.ocr_policy`, :func:`Policies.vlm_policy`, or your own
    #: ``Callable[[Page], List[Op]]``.  ``None`` skips preprocessing -- the
    #: baseline to measure against.
    policy: Optional[PolicyFn] = default_policy
    #: Picks a backend per page.  Also :class:`RuleRouter`, :class:`BudgetRouter`,
    #: or your own ``Callable[[Page], Optional[str]]``.  ``None`` sends every page
    #: to :attr:`backend`.
    router: Optional[RouterFn] = default_router
    #: Force one reader for all pages: a registered name (``"pymupdf"``,
    #: ``"tesseract"``, ``"paddle"``, ``"anthropic"``) or a :class:`TextBackend`.
    #: Takes precedence over :attr:`router`.
    backend: Optional[Union[str, TextBackend]] = None
    #: What to extract -- pydantic model, dataclass, or ``{field: type}`` dict.
    #: ``None`` means text only: no model call, no cost.
    schema: Optional[SchemaLike] = None
    #: Domain hints for the prompt, e.g. ``"Indian private hospital bill, INR"``.
    context: str = ""
    #: The :class:`LLMClient` that does the extracting.  Required when
    #: :attr:`schema` is set.
    client: Optional[LLMClient] = None
    #: Business rules run over the extracted object; each outcome feeds the
    #: confidence of the fields it touched.
    validators: List[Validator] = field(default_factory=list)
    #: Maps fused confidence onto calibrated probabilities.  Without one, scores
    #: rank correctly but are not probabilities -- see :class:`PlattCalibrator`.
    calibrator: Optional[Calibrator] = None
    #: Render resolution for PDF pages.  300 is the floor for reliable OCR; 400
    #: helps on small print; above 600 costs memory without accuracy.
    render_dpi: int = DEFAULT_DPI
    #: Stop after this many pages.  ``0`` means no limit.
    max_pages: int = 0
    #: Thread count for per-page preprocessing and reading.  ``0`` runs serially,
    #: which is what you want while debugging.
    max_workers: int = 0
    #: Split pages that hold two physical documents before reading -- see
    #: :func:`Ops.split_multi_bill_page`.
    split_pages: bool = False
    #: Apply script, digit, date and amount normalisation to spans after reading.
    normalize: bool = True
    #: Currency ceiling for the read stage; exceeding it raises
    #: :class:`BudgetExceeded`.  ``None`` means unbounded.  Meaningless until
    #: :func:`Pricing.set_pricing` is called -- unpriced models cost ``0.00``.
    budget: Optional[float] = None
    #: Label carried into eval reports so runs can be told apart.
    name: str = "pipeline"

    def ingest(self, source: Source, **kwargs: Any) -> Document:
        """Ingest ``source`` using this pipeline's DPI and page limit.

        :param source: path, path-like, directory, raw ``bytes``, or open binary
            file -- anything :func:`ingest` accepts
        :param kwargs: forwarded to :func:`ingest` (``password``, ``kind_hint``,
            ...); ``render_dpi`` and ``max_pages`` come from this pipeline
        :returns: a :class:`Document` whose rasters are still lazy -- nothing has
            been rendered yet
        """
        return ingest(source, render_dpi=self.render_dpi, max_pages=self.max_pages,
                      **kwargs)

    def run_document(self, doc: Document) -> Document:
        """Preprocess, read and normalise an already-ingested document.

        The middle of :meth:`run`, exposed separately for when you ingested the
        document yourself -- to filter its pages, attach metadata, or feed pages
        assembled from somewhere other than a file.

        :param doc: an ingested document; it is read in place and also returned
        :returns: the same document, with ``page.spans`` populated, ``page.history``
            recording every op applied, and text normalised if ``normalize`` is set
        """
        if self.split_pages:
            doc = split_document(doc)
        if self.policy is not None:
            doc = preprocess(doc, policy=self.policy, max_workers=self.max_workers)
        doc = read(doc, backend=self.backend, router=self.router,
                   max_workers=self.max_workers, budget=self.budget)
        if self.normalize:
            doc = normalize_document(doc, in_place=True)
        return doc

    def run(self, source: Source, **kwargs: Any) -> Extraction:
        """Run end to end and return an :class:`Extraction`.

        With no ``schema`` configured the result still carries the document,
        so a text-only pipeline is just this with the extraction step empty.

        :param source: path, path-like, directory, raw ``bytes``, or open binary
            file -- anything :func:`ingest` accepts
        :param kwargs: forwarded to :meth:`ingest`, e.g. ``password="secret"``
            for an encrypted PDF
        :returns: an :class:`Extraction`; when ``schema`` is ``None`` its
            ``model`` is ``None`` and only ``document`` and ``cost`` are filled in
        :raises IngestError: ``source`` could not be read as a document
        :raises BudgetExceeded: this pipeline has a ``budget`` and reading exceeded it
        """
        doc = self.run_document(self.ingest(source, **kwargs))
        if self.schema is None:
            return Extraction(model=None, document=doc, cost=Cost.zero())
        return extract(doc, self.schema, context=self.context, client=self.client,
                       validators=self.validators, calibrator=self.calibrator)

    def __call__(self, source: Source, **kwargs: Any) -> Extraction:
        """``pipeline(source)`` -- alias for :meth:`run`, so it is a ``PipelineFn``.

        Being callable is what lets a configured pipeline be handed straight to
        :meth:`EvalSuite.run`, which expects a ``Callable[[str], Extraction]``.

        :param source: as :meth:`run`
        :param kwargs: as :meth:`run`
        :returns: an :class:`Extraction`
        """
        return self.run(source, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready description of how this pipeline is configured.

        Callables are recorded by name, not serialised: the point is to make a
        report say which policy and router produced it.
        """
        return {
            "name": self.name,
            "policy": getattr(self.policy, "__name__", str(self.policy)),
            "router": getattr(self.router, "__name__", str(self.router)),
            "backend": getattr(self.backend, "name", self.backend),
            "schema": getattr(self.schema, "__name__", str(self.schema)),
            "client": getattr(self.client, "name", None),
            "validators": [getattr(v, "__name__", repr(v)) for v in self.validators],
            "render_dpi": self.render_dpi, "split_pages": self.split_pages,
            "normalize": self.normalize, "budget": self.budget,
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
            "docpipe_version": __version__,
        }


def process(source: Source, schema: Optional[SchemaLike] = None, context: str = "",
            client: Optional[LLMClient] = None, policy: Optional[PolicyFn] = default_policy,
            router: Optional[RouterFn] = default_router,
            backend: Optional[Union[str, TextBackend]] = None,
            validators: Optional[Sequence[Validator]] = None,
            **kwargs: Any) -> Extraction:
    """One-call convenience wrapper around :class:`Pipeline` -- ingest to extraction.

    Equivalent to building a :class:`Pipeline` with these arguments and calling
    it once.  Use this for a single document; build a :class:`Pipeline` when you
    want to reuse one configuration across many documents, record it in an eval
    report, or run it through :class:`EvalSuite`.

    :param source: what to read.  A path or path-like (``"scan.pdf"``,
        ``Path("bill.png")``), a directory of pages, raw ``bytes`` of a PDF or
        image, or an open binary file object.  Format is sniffed from content,
        not from the extension.
    :param schema: what to pull out -- a pydantic model, a dataclass, or a plain
        ``{field: type}`` dict such as ``{"invoice_no": str, "total": float}``.
        Leave it ``None`` for a text-only run: the read :class:`Document` still
        comes back on ``Extraction.document``, with no model call made.
    :param context: domain hints pasted into the prompt, e.g.
        ``"Indian private hospital bill, amounts in INR"``.  Cheap and effective
        -- it is what tells the model that ``1,20,000`` is one-lakh-twenty.
    :param client: the :class:`LLMClient` doing the extracting -- one
        of :class:`AnthropicClient`, :class:`OpenAIClient`, or the
        deterministic :class:`EchoClient` in tests.  Required whenever
        ``schema`` is given.
    :param policy: chooses preprocessing ops per page from its measured quality.
        Defaults to :func:`default_policy`.  Pass :func:`Policies.ocr_policy` or
        else :func:`Policies.vlm_policy` to bias for one reader, a callable of
        your own for full control, or ``None`` to skip preprocessing entirely --
        the honest baseline when measuring whether preprocessing earns its keep.
    :param router: picks a backend per page, e.g. native text layer for digital
        pages and a VLM for photographed ones.  Defaults to the
        rule-based :func:`default_router`; see also :class:`RuleRouter`
        and :class:`BudgetRouter`.  ``None`` sends every page to ``backend``.
    :param backend: force one reader for every page, as a registered name
        (``"pymupdf"``, ``"tesseract"``, ``"paddle"``, ``"anthropic"``) or an
        instance of :class:`TextBackend`.  Overrides ``router`` when both given.
    :param validators: business rules run over the extracted object; each
        outcome feeds the confidence of the fields it touched.  The
        namespace :class:`Validators` holds the ready-made ones.
    :param kwargs: forwarded to :class:`Pipeline` -- ``render_dpi``,
        ``max_pages``, ``max_workers``, ``split_pages``, ``normalize``,
        ``budget``, ``calibrator``, ``name``.
    :returns: an :class:`Extraction` carrying the parsed object, per-field
        confidence and provenance, the read document, the
        accumulated :class:`Cost`, and any warnings.  A failure that costs one
        page rather than the whole document arrives as a warning, not an
        exception.
    :raises ConfigError: a ``schema`` was given without a ``client``
    :raises IngestError: ``source`` could not be read as a document
    :raises BudgetExceeded: a ``budget`` was set and reading would exceed it

    Text only, no model call::

        doc = docpipe.process("scan.pdf").document
        print(doc.text())

    A schema, with domain context and a validator::

        result = docpipe.process(
            "bill.pdf",
            schema={"patient": str, "total": float, "bill_date": str},
            context="Indian private hospital bill, INR",
            client=docpipe.AnthropicClient(),
            validators=[docpipe.Validators.line_items_sum_to_total],
        )
        print(result.value("total"), result.confidence("total"))
        for field in result.low_confidence(threshold=0.7):
            print("review by hand:", field.name, field.confidence)

    Cheap OCR, no VLM, capped at the first 20 pages::

        result = docpipe.process("bundle.pdf", backend="tesseract", router=None,
                                 policy=docpipe.Policies.ocr_policy, max_pages=20)
    """
    pipeline = Pipeline(policy=policy, router=router, backend=backend, schema=schema,
                        context=context, client=client,
                        validators=list(validators or []), **kwargs)
    return pipeline.run(source)


# =============================================================================
# 15. Command line interface
# =============================================================================
#
#   python -m docpipe caps
#   python -m docpipe info scan.pdf
#   python -m docpipe quality scan.pdf
#   python -m docpipe preprocess scan.pdf --out ./debug
#   python -m docpipe read scan.pdf --backend tesseract
#   python -m docpipe extract bill.pdf --schema schema.json --client anthropic
#   python -m docpipe eval datasets/bills --report reports/run.json


def _cli_caps(args: Any) -> int:
    """``docpipe caps`` -- report importable integrations and backends."""
    caps = capabilities()
    print("docpipe %s on Python %s" % (__version__, sys.version.split()[0]))
    for name in sorted(caps):
        print("  %-22s %s" % (name, "yes" if caps[name] else "no"))
    print("\nregistered backends: %s" % ", ".join(registry.names()))
    print("available backends:  %s" % (", ".join(registry.available()) or "(none)"))
    print("registered ops:      %s" % ", ".join(sorted(OP_FACTORIES)))
    return 0


def _cli_info(args: Any) -> int:
    """``docpipe info`` -- ingest a document and describe what arrived."""
    doc = ingest(args.path, max_pages=args.max_pages, render_dpi=args.dpi)
    print("source:  %s" % doc.source_uri)
    print("pages:   %d" % len(doc.pages))
    print("kinds:   %s" % doc.kinds())
    print("chars:   %d" % doc.char_count)
    for key in ("format", "producer", "encrypted", "checksum"):
        if key in doc.meta:
            print("%-8s %s" % (key + ":", doc.meta[key]))
    for warning in doc.warnings:
        print("warning: %s" % warning)
    return 0


def _cli_quality(args: Any) -> int:
    """``docpipe quality`` -- measure per-page degradation."""
    doc = ingest(args.path, max_pages=args.max_pages, render_dpi=args.dpi)
    measure_document(doc, max_workers=args.workers)
    for page in doc.pages:
        print("page %3d  %s" % (page.index + 1, page.quality))
        if args.show_policy:
            print("          policy: %s" % (default_policy(page) or "(nothing to correct)"))
    return 0


def _cli_preprocess(args: Any) -> int:
    """``docpipe preprocess`` -- apply a policy, optionally writing PNGs."""
    policies = {"default": default_policy, "ocr": ocr_policy, "vlm": vlm_policy,
                "none": no_policy}
    doc = ingest(args.path, max_pages=args.max_pages, render_dpi=args.dpi)
    doc = preprocess(doc, policy=policies[args.policy], max_workers=args.workers)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    for page in doc.pages:
        print("page %3d  %s" % (page.index + 1, page.history_summary()))
        if args.out and page.has_raster():
            path = os.path.join(args.out, "page-%03d.png" % (page.index + 1))
            save_image(page.raster(), path)
            print("          -> %s" % path)
    return 0


def _cli_read(args: Any) -> int:
    """``docpipe read`` -- preprocess and OCR, printing text or JSON."""
    doc = ingest(args.path, max_pages=args.max_pages, render_dpi=args.dpi)
    if args.policy != "none":
        policies = {"default": default_policy, "ocr": ocr_policy, "vlm": vlm_policy}
        doc = preprocess(doc, policy=policies[args.policy], max_workers=args.workers)
    doc = read(doc, backend=args.backend, max_workers=args.workers)
    if args.json:
        print(to_json(doc.to_dict()))
    else:
        print(doc.text())
        print("\n--- backends: %s   cost: %s" % (doc.meta.get("read_backends"),
                                                 doc.meta.get("read_cost")))
    return 0


def _cli_extract(args: Any) -> int:
    """``docpipe extract`` -- run a schema end to end.  Exit 2 if invalid."""
    with open(args.schema, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    clients = {
        "anthropic": lambda: AnthropicClient(model=args.model or "claude-sonnet-5"),
        "openai": lambda: OpenAIClient(model=args.model or "gpt-4o"),
        "echo": lambda: EchoClient(response=args.echo or "{}"),
    }
    client = clients[args.client]()
    pipeline = Pipeline(schema=schema, context=args.context, client=client,
                        render_dpi=args.dpi, max_pages=args.max_pages,
                        max_workers=args.workers)
    result = pipeline.run(args.path)
    print(to_json(result.to_dict()))
    return 0 if result.is_valid else 2


def _cli_eval(args: Any) -> int:
    """``docpipe eval`` -- score a dataset, optionally gating on a baseline."""
    suite = EvalSuite.from_dir(args.dataset)
    with open(args.schema, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    client = (EchoClient(response=args.echo or "{}") if args.client == "echo"
              else AnthropicClient(model=args.model or "claude-sonnet-5")
              if args.client == "anthropic" else OpenAIClient(model=args.model or "gpt-4o"))
    pipeline = Pipeline(schema=schema, client=client, render_dpi=args.dpi,
                        max_workers=args.workers)
    report = suite.run({args.name: pipeline}, max_workers=args.workers)
    print(report.summary())
    if args.report:
        report.save(args.report)
        print("\nsaved report to %s" % args.report)
    if args.baseline:
        comparison = report.regression_vs(args.baseline, tolerance=args.tolerance)
        print("\nregressions vs baseline: %s" % (comparison["regressions"] or "none"))
        return 1 if comparison["regressed"] else 0
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m docpipe``.

    Subcommands are ``caps``, ``info``, ``quality``, ``preprocess``, ``read``,
    ``extract`` and ``eval``.  Run ``docpipe <command> --help`` for each one's
    flags.

    :param argv: the argument list, excluding the program name.  ``None``
        reads :data:`sys.argv`, which is what the console entry point does; pass
        a list to drive the CLI from a test.
    :returns: the process exit code -- ``0`` on success, ``1`` on error, and ``1``
        from ``eval --baseline`` when a metric regressed, which is what makes it
        usable as a CI gate
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="docpipe", description="Document intelligence substrate (v%s)" % __version__)
    parser.add_argument("--verbose", "-v", action="store_true", help="chattier logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="errors only")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: Any, with_path: bool = True) -> None:
        """Add the arguments every document subcommand shares."""
        if with_path:
            p.add_argument("path", help="PDF, image, TIFF or .eml file")
        p.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="render resolution")
        p.add_argument("--max-pages", type=int, default=0, dest="max_pages")
        p.add_argument("--workers", type=int, default=0, help="parallel page workers")

    p_caps = sub.add_parser("caps", help="show which optional integrations are usable")
    p_caps.set_defaults(func=_cli_caps)

    p_info = sub.add_parser("info", help="ingest a document and describe it")
    add_common(p_info)
    p_info.set_defaults(func=_cli_info)

    p_quality = sub.add_parser("quality", help="measure per-page degradation")
    add_common(p_quality)
    p_quality.add_argument("--show-policy", action="store_true",
                           help="also print the ops the adaptive policy would apply")
    p_quality.set_defaults(func=_cli_quality)

    p_pre = sub.add_parser("preprocess", help="apply a preprocessing policy")
    add_common(p_pre)
    p_pre.add_argument("--policy", default="default",
                       choices=["default", "ocr", "vlm", "none"])
    p_pre.add_argument("--out", default="", help="directory to write processed pages")
    p_pre.set_defaults(func=_cli_preprocess)

    p_read = sub.add_parser("read", help="OCR a document and print its text")
    add_common(p_read)
    p_read.add_argument("--backend", default=None, help="force a backend by name")
    p_read.add_argument("--policy", default="default",
                        choices=["default", "ocr", "vlm", "none"])
    p_read.add_argument("--json", action="store_true", help="emit the full IR as JSON")
    p_read.set_defaults(func=_cli_read)

    p_extract = sub.add_parser("extract", help="extract a JSON-schema of fields")
    add_common(p_extract)
    p_extract.add_argument("--schema", required=True, help="JSON file: {field: type}")
    p_extract.add_argument("--context", default="", help="domain hint for the model")
    p_extract.add_argument("--client", default="anthropic",
                           choices=["anthropic", "openai", "echo"])
    p_extract.add_argument("--model", default="")
    p_extract.add_argument("--echo", default="", help="canned JSON for --client echo")
    p_extract.set_defaults(func=_cli_extract)

    p_eval = sub.add_parser("eval", help="run a pipeline against a labelled dataset")
    add_common(p_eval, with_path=False)
    p_eval.add_argument("dataset", help="directory of documents + <stem>.json truth")
    p_eval.add_argument("--schema", required=True)
    p_eval.add_argument("--client", default="anthropic",
                        choices=["anthropic", "openai", "echo"])
    p_eval.add_argument("--model", default="")
    p_eval.add_argument("--echo", default="")
    p_eval.add_argument("--name", default="candidate", help="pipeline name in the report")
    p_eval.add_argument("--report", default="", help="write the report here")
    p_eval.add_argument("--baseline", default="", help="compare against a saved report")
    p_eval.add_argument("--tolerance", type=float, default=0.0)
    p_eval.set_defaults(func=_cli_eval)

    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s")
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return int(args.func(args) or 0)
    except DocpipeError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 1
    except BrokenPipeError:  # pragma: no cover - `docpipe read x.pdf | head`
        # Point stdout at devnull so the interpreter's own flush at exit does
        # not raise a second time on the way out.
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except Exception:
            pass
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        return 130


# =============================================================================
# 16. Public surface
# =============================================================================

# The export surface is deliberately namespaced.  Groups of free functions that
# share a subject -- raster primitives, quality measures, ops, parsers -- are
# reachable as staticmethods on a namespace class (``Image.to_gray``), and only
# the class is exported.  The flat names (``to_gray``) remain module attributes
# for every existing caller and are how the staticmethods reach one another, but
# they are not re-exported: one name per subject keeps ``import *`` and the
# generated reference honest about what this library actually offers.
__all__ = [
    # errors
    "DocpipeError", "MissingDependency", "IngestError", "BackendError",
    "ExtractionError", "ConfigError", "BudgetExceeded",
    # namespaces -- grouped free functions, see the note above
    "Caps", "Util", "Image", "Quality", "Ingest", "Ops", "Policies", "Pricing",
    "Text", "Confidence", "Validators",
    # utilities
    "Timer",
    # IR
    "PageKind", "Verdict", "RegionKind", "BBox", "TextSpan", "PageQuality",
    "LayoutRegion", "Layout", "OpRecord", "Page", "Document", "Cost",
    "merge_bboxes", "DEFAULT_DPI", "DEFAULT_IMAGE_DPI",
    # ingest
    "MIN_NATIVE_CHARS",
    # quality
    "QualityThresholds", "DEFAULT_THRESHOLDS",
    # preprocessing
    "Op", "OpFactory", "CompositeOp", "compose", "apply_ops", "register_op",
    "op_from_dict", "OP_FACTORIES", "preprocess",
    # backends
    "ReadResult", "TextBackend", "BaseBackend", "BackendRegistry", "registry",
    "PyMuPDFTextLayer", "TesseractOCR", "PaddleOCRBackend", "RapidOCRBackend",
    "EasyOCRBackend", "DocTRBackend", "SuryaBackend", "VisionBackend",
    "AnthropicVisionBackend", "OpenAIVisionBackend", "GeminiVisionBackend",
    "CachingBackend",
    "RetryingBackend", "EnsembleBackend", "default_router", "RuleRouter",
    "BudgetRouter", "read", "read_page", "PRICING", "DEFAULT_OCR_PROMPT",
    # cross-cutting
    "DiskCache", "Tracer", "TraceSpan", "CostTracker",
    # confidence
    "DEFAULT_CONFIDENCE_WEIGHTS", "Calibrator", "IdentityCalibrator",
    "PlattCalibrator", "IsotonicCalibrator",
    # extraction
    "FieldSpec", "SchemaAdapter", "coerce_value", "build_extraction_prompt",
    "format_document_text", "chunk_pages", "DEFAULT_EXTRACTION_SYSTEM",
    "LLMClient", "BaseLLMClient", "EchoClient", "AnthropicClient", "OpenAIClient",
    "parse_json_lenient", "Evidence", "locate_value", "backend_confidence_for",
    "ValidationIssue", "FieldResult", "Extraction", "extract",
    "merge_extracted_data", "FieldRule", "extract_with_rules",
    # eval
    "EvalCase", "FieldOutcome", "CaseResult", "compare_values", "EvalSuite",
    "EvalReport", "KNOWN_METRICS",
    # pipeline / cli
    "Pipeline", "process", "main", "__version__",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

