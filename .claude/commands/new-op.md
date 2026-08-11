---
description: Add a preprocessing op to docpipe.py following every convention the file enforces
argument-hint: "<op name and what degradation it corrects>"
allowed-tools: Bash, Read, Edit, Grep, Glob
---

Add a preprocessing op: **$ARGUMENTS**

Before writing anything, read `deskew` and `remove_shadow` in `docpipe.py` section 7 — they
are the reference implementations for the geometric and non-geometric cases respectively.

## 1. Justify it against the scope boundary

An op corrects a *measured channel distortion* that any document type could suffer. If it
only makes sense for one domain's paperwork, it belongs in that project, not here (see
README.md's *Scope boundary*). Say in one sentence which distortion this inverts. If you
cannot, stop and say so rather than adding it.

## 2. Implement in section 7

Place it near ops of the same family — geometric ops, tonal ops, removal ops — not at the end
of the section.

```python
@register_op("my_op", geometric=False, needs_raster=True)
def my_op(img: ImageArray, page: Page, strength: float = 1.0) -> Optional[ImageArray]:
    """What distortion this corrects.

    Then the part a reader cannot get from the code: why the threshold is where it is,
    what it costs when it fires needlessly, and when it deliberately does nothing.
    """
```

Requirements:

- **Return `None` when there is nothing to correct.** The page is then left untouched and no
  history entry is recorded. Do not return an unchanged copy.
- **Correct measured degradation, not unconditionally.** Read `page.quality` and bail below
  threshold. Resampling always costs sharpness; correcting 0.2° costs more than it recovers.
- **`geometric=True` if pixels move** — and then remap existing span coordinates through the
  same affine mapping the raster took, in points, via `_map_span_points`. Dropping provenance
  is not an option; see `deskew` for the pattern.
- **Update `page.quality`** for whatever you just fixed (e.g. `page.quality.skew_deg = 0.0`)
  and record specifics in `page.meta`.
- **OpenCV is optional.** If you reach for `cv2`, guard it and write the pure-NumPy fallback.
  Both paths get tested.
- **Python 3.8**: `Optional[X]`, never `X | None`. `%`-formatting, never f-strings.

## 3. Wire it up

- Add the public name to `__all__` in section 16, in the preprocessing group.
- Decide whether `default_policy` should reach for it, and if so under what measured
  condition. If it should not be automatic, say why in the docstring — an op nobody's policy
  invokes needs a reason to exist.
- Check whether `ocr_policy` / `vlm_policy` differ here. Binarisation is the precedent:
  helps classical OCR, hurts VLMs.

## 4. Test it in `tests/test_preprocess.py`

The suite's method is a *known* degradation applied to a synthetic page with recorded
ground-truth word boxes, then asserting the measurement recovers the known value. Follow it:

- Degrade a clean fixture page by a known amount, apply the op, assert recovery within
  tolerance.
- Assert it returns `None` / no-ops on a page that does not need it.
- If geometric: assert span boxes still land on their words after the op.
- Use the `both_image_backends` fixture so it runs through both OpenCV and NumPy.

## 5. Verify

```bash
pytest -q tests/test_preprocess.py tests/test_image_ops.py
DOCPIPE_DISABLE_OPENCV=1 pytest -q tests/test_preprocess.py
python docs/build_reference.py --out /tmp/ref.html && grep -c "id='my_op'" /tmp/ref.html
```

Then report: what it corrects, when it fires, whether any policy calls it, and the test
numbers that show it works.
