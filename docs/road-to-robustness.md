# Road to Robustness

**What this is.** A problem inventory for the documents we actually get — photographed
Indian hospital bills, handwritten legal filings, multi-generation photocopies with a
rubber stamp across the total — and a proposed route from where `docpipe` is today to
handling them out of the box.

**How to read it.** Every entry has the same four parts:

> **What you see** — the observable failure.
> **Why it happens** — from first principles: optics, the writing system, or how the
> document was produced. Not "OCR is bad at this."
> **Why the pipeline fails today** — with the line in `docpipe.py` responsible.
> **What to do** — a concrete change, sized.

Entries are tagged **[P0]** / **[P1]** / **[P2]**. P0 means the pipeline is *silently
wrong* on this input today — not slow, not degraded: wrong, with high confidence. Those
come first regardless of how hard they are.

Nothing proposed here is domain-specific. "Photographed page", "handwritten region" and
"multi-script line" are properties of documents in general; a team doing Kannada land
records needs every one of them. The hospital-bill specifics stay in the consuming
project, per README's *Scope boundary*.

---

## Part 0 — The frame

Three principles that decide most of the design questions below. Worth internalising
before reading the problem list, because every proposed solution falls out of one of them.

### 0.1 Preprocessing cannot add information

A scan is a communication channel: the document is the signal, capture is the channel,
and every degradation is information destroyed or buried. Preprocessing can only
*re-allocate* what survived — move signal out of a range the recogniser handles badly
into one it handles well. It can never restore what the channel destroyed.

This has a sharp practical edge. Contrast, illumination gradient and mild blur are
**recoverable**: the information is present, just poorly distributed, and the right op
redistributes it. Specular glare that clipped pixels to 255, and a region below the
Nyquist limit for its stroke width, are **unrecoverable**: no op, however clever, gets
them back.

> The pipeline must tell these two apart and treat them differently. Recoverable
> degradation is a job for preprocessing. Unrecoverable degradation is a job for
> *reporting* — mark the region, lower confidence, route to a human. Today we conflate
> them: everything becomes a scalar in `PageQuality` that some op tries to fix.

### 0.2 Every scalar quality metric hides a local disaster

`PageQuality` ([docpipe.py:1022](../docpipe.py#L1022)) is eight page-level numbers. Page-level
numbers are the right shape for *routing* and mostly wrong for *diagnosis*, because the
failures that cost us money are local: glare on one line, a shadow over one column, one
handwritten annexure in a 40-page bundle.

A page that is 95% pristine and 5% destroyed scores ~0.95 and routes as clean. The 5% is
the doctor's handwritten discharge note, and the total is in it.

> Metrics need to carry **where**, not just **how much**. A `BBox` alongside the scalar
> converts an unactionable score into a routable region.

### 0.3 Confidence must come from independent evidence

The central design invariant of the library, and the one most systems get wrong. See
Part 4, which is about exactly this — including the OCR-score-routing pattern in use in
the office.

---

## Part 1 — The capture problem

The library was built for *scans*. Our hardest inputs are *photographs*, and a phone
camera is a different channel with different failure modes. This is the largest single
gap.

### 1.1 Specular glare **[P0]**

**What you see.** A white blob across part of the page — laminated bill, glossy thermal
paper, a ceiling tube-light reflecting off the plastic sleeve. Text under it is gone.
OCR returns nothing there, or worse, returns something.

**Why it happens.** Two reflection modes. *Diffuse* reflection scatters light in all
directions and carries the ink's information — that is what we normally photograph.
*Specular* reflection is a mirror bounce off a smooth surface, angle-in equals
angle-out, and it carries the information of the *light source*, not the page. When the
specular lobe points at the lens, the sensor sees the tube light. If it exceeds full
well capacity the pixel clips to 255 and the ink beneath is not attenuated, not
dimmed — **absent**. Per §0.1, unrecoverable.

**Why the pipeline fails today.** `measure_illumination` ([docpipe.py:2381](../docpipe.py#L2381))
downsamples to 16×16 and takes the p90−p10 spread. That is a *global gradient* detector
— excellent for the shadow case it was written for, blind to this one. A hard highlight
over 5% of the page shifts one or two cells of a 256-cell grid; the spread barely moves.
The page scores well-lit. Then `normalize_illumination` is not even triggered, and if it
were it would divide by an estimated background and turn the clipped region into
plausible-looking grey noise — actively worse, because it launders an unrecoverable
region into one that looks readable.

**What to do.**
- `Quality.measure_glare(gray) -> Tuple[float, List[BBox]]` — connected components of
  near-saturated pixels (>250) with low local gradient variance. The gradient test is
  what separates glare from genuine white paper: blank paper is bright *and* smooth, but
  so is glare, so add the discriminator that glare regions are bright while their
  *neighbourhood* contains ink at normal exposure. Return coverage fraction and the
  boxes.
- Store boxes in `page.layout.regions` as a new `RegionKind.UNREADABLE`.
- Never try to correct it. Spans overlapping a glare box get a hard confidence penalty
  via `page_quality_prior` ([docpipe.py:6990](../docpipe.py#L6990)), which already accepts a
  `bbox` and is the right hook.
- If glare coverage exceeds ~15%, the honest verdict is `Verdict.UNREADABLE` and the
  honest action is *ask for a re-shoot*. A pipeline that can say "photograph this again
  without the flash" beats one that extracts a wrong total.

### 1.2 Perspective distortion **[P0]**

**What you see.** The page is a trapezium. Text lines converge toward one edge, the far
edge is smaller than the near one. Table columns are not parallel.

**Why it happens.** A camera is a projective device. Only when the sensor plane is
exactly parallel to the page does the image preserve parallel lines; at any other angle
the projective transform maps the page rectangle to a general quadrilateral. Handheld,
the sensor plane is *never* parallel. Bills photographed on a counter get 10–30° of tilt
as a matter of course.

**Why the pipeline fails today.** `skew_deg` is in-plane rotation only — one parameter of
a transform that needs eight. `estimate_skew` ([docpipe.py:2398](../docpipe.py#L2398)) will
report *some* angle for a keystoned page, `deskew` will rotate by it, and the result is a
page that is still trapezoidal and now also rotated. Downstream, OCR line segmentation
fails because baselines are not horizontal *and not parallel to each other*, so a single
rotation cannot make them so.

`Ops.perspective_correct` ([docpipe.py:4027](../docpipe.py#L4027)) exists and is good. It is
called by nothing. No metric detects the condition and no policy reaches for the op.

**What to do.**
- `Quality.measure_perspective(gray) -> float` — fit baselines to text rows, measure how
  much their angles *diverge*. Parallel baselines mean pure rotation; converging
  baselines mean projection. The spread of baseline angles is the signal, and it
  distinguishes the two cases cleanly.
- Add a perspective branch to a new `photo_policy` (§1.7), ordered **before** `deskew` —
  correcting projection first often leaves zero residual skew.
- Note the constraint already documented at [docpipe.py:4071](../docpipe.py#L4071): a projective
  warp invalidates axis-aligned bboxes, so `perspective_correct` clears `page.spans`.
  This op must run in preprocessing or never. It cannot be a repair applied after a bad
  read.

### 1.3 Page curvature **[P1]**

**What you see.** Baselines that curve. Common on bound registers photographed near the
spine, folded bills flattened by hand, and thermal rolls that keep their curl.

**Why it happens.** The page is not a plane, so *no* homography flattens it — projective
correction assumes a planar subject. The required transform is a per-row non-linear
resampling, which needs the surface estimated first.

**Why the pipeline fails today.** Unmeasured. `perspective_correct` will find a
quadrilateral, warp on it, and leave the curvature — sometimes worse, because the warp
stretches the curved region unevenly.

**What to do.** Measure first, correct later. `Quality.measure_curvature` fits a
quadratic to each detected baseline and reports mean curvature; even without a dewarper
this is worth having, because it routes the page to a VLM (which tolerates curvature far
better than a line-segmenting OCR) and it correctly suppresses `perspective_correct`,
which would otherwise make things worse. A proper cylindrical dewarp is a P2 follow-up
and is genuinely hard.

### 1.4 Motion blur **[P1]**

**What you see.** Text smeared in one direction. Handheld, indoors, no flash.

**Why it happens.** Hospital and court interiors are dim. The camera compensates with a
long exposure; hand tremor moves the sensor during it. The result is convolution with a
*line* kernel, not the isotropic Gaussian that defocus produces.

**Why the pipeline fails today.** `measure_blur` ([docpipe.py:2272](../docpipe.py#L2272)) is
carefully built and direction-agnostic. It correctly reports the page as blurry, so
`unsharp` fires — and isotropic sharpening on directional blur amplifies noise
perpendicular to the smear while barely touching the smear itself.

**What to do.** Extend the blur measurement to report *anisotropy*: gradient energy along
the dominant orientation versus perpendicular to it. High anisotropy means motion, which
means (a) don't bother with `unsharp`, and (b) route to a VLM, which reads through
directional blur remarkably well because it has a language prior and OCR does not. Fits
naturally into `PageQuality.extra` before earning a first-class field.

### 1.5 Shadow **[P1]**

**What you see.** A dark band across the page — the photographer's own body or phone
between the light and the document.

**Why it happens.** Occlusion. Illuminance falls where the occluder blocks the source,
so the *same ink* has different pixel values in different parts of the page. Fully
recoverable per §0.1, as long as nothing clipped to 0.

**Why the pipeline fails today.** This one is handled: `measure_illumination` detects it
and `default_policy` reaches for `normalize_illumination`. The remaining problem is
`ocr_policy` ([docpipe.py:4583](../docpipe.py#L4583)), which appends a **global** `binarize`.
Any single global threshold on a shadowed page classifies shadowed paper as ink and
lit ink as paper. The default method is `sauvola`, which is adaptive and largely immune —
so the fix is mostly to make sure we never fall back to a global method on a page whose
`illumination` is low.

**What to do.** Small: forbid non-adaptive binarisation when `illumination` is below the
threshold, and prefer `remove_shadow` ([docpipe.py:3763](../docpipe.py#L3763)) over plain
`normalize_illumination` when the gradient is localised rather than smooth.

### 1.6 Background clutter **[P2]**

**What you see.** The photo includes the desk, a hand holding the page down, the corner
of the next bill.

**Why it happens.** Nobody crops. The framing is "get the document in shot".

**Why the pipeline fails today.** Every page-level metric is computed over the whole
frame, so a dark wooden desk drags `ink_coverage` up, `contrast` around, and
`effective_dpi` down — the page is measured *including things that are not the page*.
Metrics feed policy and confidence, so one un-cropped photo produces wrong measurements,
wrong ops, and a wrong prior. `crop_to_content` ([docpipe.py:3942](../docpipe.py#L3942)) exists
and is never policy-triggered.

**What to do.** Detect the page quadrilateral **first**, before measuring anything —
`perspective_correct` already locates it, so the detection exists and needs lifting out
into a reusable `Quality.detect_page_bounds`. Then crop, then measure. This is an
ordering fix more than new code, and it makes every other metric on this list more
accurate.

### 1.7 The unifying fix: know you are looking at a photograph **[P0]**

None of the above can be applied unconditionally — running perspective detection on a
clean 400 DPI native render is wasted work at best and damage at worst, which is exactly
the objection `default_policy`'s design already encodes.

So the first thing to build is the discriminator:

```
Quality.detect_capture_mode(page) -> "native" | "scan" | "photo"
```

Signals, all cheap and already computable: native pages carry a text layer; scans have
near-uniform illumination, near-zero perspective, and background pixels tightly clustered
near white; photos have illumination gradient, non-zero perspective, non-white background
outside the page quad, and EXIF that says so when the source is a JPEG.

Then `Policies.photo_policy`, dispatched on that mode:

```
crop to page bounds → perspective correct → remove shadow →
normalize illumination → [existing default_policy branches]
```

**Do items §1.7, §1.1 and §1.2 first.** They are one coherent change — a capture-mode
classifier, two new metrics, one new policy — and every other camera item plugs into the
frame they establish.

---

## Part 2 — The writing problem

### 2.1 Handwriting is not degraded print **[P0]**

**What you see.** A doctor's discharge note, a hand-filled quantity column, a magistrate's
margin annotation. Tesseract returns confident garbage.

**Why it happens.** This is the deepest item in the document, so it is worth being
precise about the mechanism.

Printed text is drawn from a **finite alphabet of fixed glyph shapes**. A font has one
canonical form per character; every instance of "5" in a document is pixel-identical
modulo noise. Classical OCR is built on exactly this: segment into glyph-sized boxes,
match each against templates or a CNN trained on rendered fonts, done. Its whole accuracy
budget assumes shape variance comes from *the channel* (blur, noise, threshold), and
channel degradation is what the model was augmented against.

Handwriting violates every one of those assumptions:

1. **Unbounded shape variance.** There is no canonical form. The same writer's "5" varies
   between instances; across writers the variance exceeds the *between-class* variance
   of printed characters. A handwritten "5" can be geometrically closer to some writer's
   "8" than to another writer's "5". No template match survives this.
2. **No segmentation boundary.** Cursive is connected by construction. The classical
   pipeline needs to segment before it recognises, but you cannot correctly segment
   cursive without already knowing what it says — the notorious chicken-and-egg that
   killed template OCR for handwriting and is why the field moved to sequence models
   (CTC, attention) that recognise whole lines without segmenting.
3. **No baseline discipline.** Handwritten lines drift, skew locally, and vary in height
   within a line. Line segmentation by horizontal projection — which is what most OCR
   engines and our own `estimate_line_pitch` ([docpipe.py:2603](../docpipe.py#L2603)) do —
   assumes rows of ink separated by rows of paper. Drifting baselines smear that
   projection until the peaks vanish.
4. **The context requirement.** Humans do not read bad handwriting glyph-by-glyph; they
   read it word-by-word against a language prior and a domain prior. An ambiguous squiggle
   in the quantity column of a pharmacy bill is resolved by knowing that the column holds
   small integers. A recogniser with no language model *cannot do this in principle*,
   regardless of training data — the information required is not in the pixels.

Point 4 is why VLMs beat OCR on handwriting by a wide margin, and it is also the warning:
the same language prior that resolves genuine ambiguity **also fabricates plausible
readings of illegible text**. A VLM will never return "I could not read this"
unprompted — it returns the most probable string. On handwriting the failure mode
changes from *garbage* (visibly wrong, cheap to catch) to *plausible* (invisibly wrong,
expensive). See Part 4.

**Why the pipeline fails today.** `RegionKind.HANDWRITING` exists
([docpipe.py:778](../docpipe.py#L778)). `Layout.has_handwriting` exists
([docpipe.py:1117](../docpipe.py#L1117)). `default_router` branches on it to send the page to a
vision model ([docpipe.py:6066](../docpipe.py#L6066)). And **nothing in the library ever
constructs a `LayoutRegion`** — `Layout.regions` is only ever copied, never populated.
The branch is unreachable.

So handwritten pages route on `quality.score < 0.45` alone. Consider what that means for
a *well-photographed* handwritten note: sharp, well-lit, high DPI, score ~0.85. It routes
to Tesseract. **The better the photograph of the handwriting, the more certainly we send
it to the engine that cannot read it.** That is the single worst behaviour in the
pipeline today, and it is one unimplemented function away from being correct.

**What to do.**
- `Quality.detect_handwriting(gray) -> List[BBox]`, populating `page.layout.regions`.
  Discriminating features, none needing a model:
  - **Stroke-width variance.** Print has near-constant stroke width; a pen varies with
    pressure and speed. `estimate_stroke_width` ([docpipe.py:2579](../docpipe.py#L2579)) already
    computes the mean via area-to-perimeter — compute it per connected component and take
    the *variance*. High variance is the strongest single cue.
  - **Baseline irregularity.** Fit a line to the bottoms of components in a row; print has
    tiny residuals, handwriting large ones.
  - **Glyph-height variance** and **inter-glyph spacing variance**, both far higher for
    handwriting.
  - **Component connectivity** — cursive produces long multi-character components whose
    aspect ratio exceeds anything in print.
  A weighted combination on connected-component clusters gets a usable region detector
  with zero dependencies. It does not need to be excellent; it needs to be better than
  the constant `False` we have now, which is a very low bar.
- Once regions exist, `default_router`'s handwriting branch starts working as designed,
  and a mixed page (printed form, handwritten fill-in) becomes *routable per region* —
  which is the real prize, and is what a page-level router can never do.
- Carry illegibility through to confidence. `DEFAULT_OCR_PROMPT`
  ([docpipe.py:5473](../docpipe.py#L5473)) already instructs the model to write `[illegible]`
  rather than guess — its docstring names hallucinating plausible values as one of the two
  failure modes it was written against. But the only consumer is a page-level *warning
  string* ([docpipe.py:5581](../docpipe.py#L5581)). The model tells us where it could not read,
  and we log it and move on. That marker should become a `RegionKind.UNREADABLE` span and a
  hard confidence penalty on any field extracted near it.

### 2.2 Devanagari is not Latin with extra marks **[P1]**

**What you see.** Hindi/Marathi text degrading much faster than Latin on the same page,
under the same processing.

**Why it happens.** Structural differences that break Latin-tuned assumptions:

- **The shirorekha.** The horizontal headline joining characters within a word. It is
  *load-bearing*: remove it and the glyphs beneath are barely identifiable. It also means
  a Devanagari word is one connected component, so any component-per-glyph assumption is
  wrong.
- **Vertical stacking.** Matras attach above and below the base — `ि ी ु ू े ै ो ौ ं ँ`.
  The information sits in *small marks far from the glyph centre*, so a tight bounding box
  around the main body loses the vowel, which changes the word.
- **Conjuncts.** Consonant clusters form new ligature glyphs, expanding the effective
  alphabet from ~50 characters to several hundred shapes.
- **The nukta.** A single dot below that distinguishes क from क़, ज from ज़. Its total ink
  is a handful of pixels.

**Why the pipeline fails today.** Two real hazards, and one I want to explicitly retract.

*Retraction, so nobody chases it:* I said earlier that `remove_lines` eats the shirorekha.
Checking the implementation ([docpipe.py:4283](../docpipe.py#L4283)), it requires a horizontal
run of at least `min_len_ratio=0.4` — 40% of page width. A shirorekha breaks at every word
boundary, so at default settings it survives comfortably. The risk only appears if someone
tunes `min_len_ratio` down, which is worth a docstring warning and nothing more.

The genuine hazards:

- **`despeckle` ([docpipe.py:4109](../docpipe.py#L4109)) removes ink components below
  `min_area_px=6`.** At 300 DPI a nukta or an anusvara is comfortably above that. At the
  ~150 DPI effective resolution typical of a phone photo of a small-print bill, it is
  right at the boundary. `ocr_policy` applies `despeckle` unconditionally. We are, on
  some pages, deleting phonemes.
- **`denoise` ([docpipe.py:4077](../docpipe.py#L4077))** has the same character at higher
  strength — `vlm_policy`'s docstring already says so ("removes diacritics and thin
  Devanagari strokes") but `ocr_policy` does not act on the same knowledge.

**What to do.** Make the destructive ops script-aware. `Text.detect_script`
([docpipe.py:6324](../docpipe.py#L6324)) exists but only runs *after* reading — chicken-and-egg.
Two ways out, and we should do both: (a) a cheap pre-read script guess from ink
morphology, since the shirorekha gives Devanagari a distinctive horizontal projection
signature that is easy to detect without recognising anything; (b) when a document has
*any* prior read, cache the detected script on `Document.meta` and let policy consult it.
Then scale `despeckle(min_area_px)` with measured stroke width rather than leaving it a
fixed pixel count — which is the right fix regardless of script, since a fixed pixel
threshold is resolution-dependent and we know the resolution.

### 2.3 Mixed scripts on one line **[P1]**

**What you see.** `रोगी का नाम: SUNIL KUMAR`, `दवा: Amoxicillin 500mg`, `₹ 1,250.00`.
Three scripts in one line is *normal*, not exceptional.

**Why it happens.** Indian institutional documents are bilingual by law and by habit:
Devanagari letterhead and field labels, Latin proper nouns and drug names, and numerals
in either script. Nobody produced this document in one language.

**Why the pipeline fails today.** OCR engines take a language parameter that sets the
character set *and* the language model. Choosing `hin` cripples the Latin drug name;
choosing `eng` cripples the Devanagari label. Neither choice is right, and the pipeline
currently forces a single one per page.

**What to do.** Surya ([docpipe.py:5302](../docpipe.py#L5302)) already handles this, and its
docstring says exactly why it is here. The gap is that `default_router` only reaches for
it as a fallback. Add multi-script detection as a first-class routing signal — the same
ink-morphology guess from §2.2 — so mixed-script pages go to Surya or a VLM by design
rather than by luck.

### 2.4 Numerals **[P0 — but already solved, keep it that way]**

**What you see.** `४८२५०` and `48250` are the same amount.

**Why it happens.** Devanagari digits are a separate Unicode block, and the same document
frequently uses both.

**Why the pipeline is fine.** `Text.normalize_digits` ([docpipe.py:6348](../docpipe.py#L6348))
folds Devanagari, Arabic-Indic, extended Arabic-Indic and full-width digits to ASCII, and
`parse_amount` ([docpipe.py:6401](../docpipe.py#L6401)) handles the separators. This is the
part of the library most ready for our inputs.

**What to watch.** The **lakh/crore grouping** — `12,34,567.00`, not `1,234,567.00`. The
`decimal_separator="auto"` heuristic infers from separator positions, and Indian grouping
produces a 2-2-3 pattern that a Western-trained heuristic can misread. This deserves an
explicit test case with Indian grouping before we trust it, and it is a five-minute test
to write.

---

## Part 3 — The document-production problem

Degradations baked in before anyone photographed anything.

### 3.1 Dot-matrix and thermal print **[P1]**

**What you see.** Characters made of visibly separated dots; or grey, low-contrast thermal
text that fades toward the edges.

**Why it happens.** Dot-matrix impact printers — still ubiquitous in Indian hospital
billing because the ribbon produces carbon copies in one pass — draw glyphs as a sparse
pin grid. The glyph is *not connected*. Thermal printers darken heat-sensitive paper,
which fades with time, light and temperature; a bill photographed weeks after issue is
substantially lighter than when printed.

**Why the pipeline fails today.** Dot-matrix breaks two things at once: `despeckle` sees
the individual dots as speckle (they are small isolated components — exactly its target),
and `estimate_stroke_width` measures the *dot* diameter rather than the stroke, inflating
perceived resolution and suppressing the `ensure_dpi` upsample that would actually help.
For thermal, low contrast is detected and `autocontrast` fires, which is correct — the
issue is only that faded thermal often also has ink close to the paper value, where
`ink_mask`'s Otsu threshold becomes unstable.

**What to do.** Detect dot-matrix from the periodicity of the component-size distribution
(a strong single mode at small size, regularly spaced — quite distinctive), and when
found: skip `despeckle` entirely and apply a small morphological *close* to bridge the
dots into strokes before anything else looks at the page.

### 3.2 Rubber stamps and seals over text **[P1]**

**What you see.** A round purple hospital seal across the total. A court stamp over the
case number. Signature ink through the amount.

**Why it happens.** Stamps are applied *after* printing, by design — they attest to the
printed content, so they are placed on the important part. The overlap is not accidental:
it is the point. The total and the case number are exactly the fields we most need.

**Why the pipeline fails today.** `remove_stamps` ([docpipe.py:4327](../docpipe.py#L4327))
exists and works on the colour axis: stamps are usually blue/purple/red while print is
black, so a saturation threshold separates them. Good approach. But it is in no policy,
and it needs colour — meaning it must run *before* `to_grayscale`, which `ocr_policy`
applies early. The ordering makes the op unusable in the pipeline that most needs it.

**What to do.** Fix the ordering: colour-dependent ops run first, and `ocr_policy` should
insert `remove_stamps` before `to_grayscale` whenever the page is colour and saturated
regions are present. Also register the stamp as a `RegionKind.STAMP` region, because a
field whose bbox overlaps a stamp should carry lower confidence even after removal — ink
under a stamp is partially destroyed, and removing the stamp colour does not bring it
back (§0.1 again).

### 3.3 Photocopy generations **[P1]**

**What you see.** Legal filings are routinely copies of copies of copies. Contrast
collapses, thin strokes vanish, background speckle accumulates, and the whole page has a
grey cast.

**Why it happens.** Each generation applies the same channel twice — a scan and a
print — and the degradations compound multiplicatively. Xerographic reproduction has a
thresholding character: strokes below the toner threshold disappear entirely, and each
generation raises the effective threshold. Information is lost at *every* generation and
per §0.1 none of it comes back.

**Why the pipeline fails today.** The metrics catch the symptoms individually — low
contrast, high noise, low effective DPI — and the policy corrects each. What is missing
is the recognition that a page failing *all three simultaneously* is qualitatively
different from a page failing one. Compounded degradation means the *confidence* should
drop faster than any single metric implies, and `PageQuality.score` is a weighted linear
sum ([docpipe.py:1047](../docpipe.py#L1047)) — linear in each term, so it cannot express
"three moderate failures are worse than one bad one".

**What to do.** Add a multi-degradation interaction term to `score`, or more honestly:
treat this as the first thing to *calibrate* rather than guess (Part 5). The weights in
`score` carry a docstring saying they exist to be recalibrated. This is the case that
proves it.

### 3.4 Pre-printed forms with handwritten fill-in **[P0]**

**What you see.** The dominant format for admission records, consent forms, and most court
filings: printed labels and ruled boxes, values written in by hand.

**Why it happens.** It is the cheapest way to produce a structured document without a
computer, and it remains standard.

**Why the pipeline fails today.** This is §2.1 compounded by a routing problem. The page is
*mostly* clean print, so every page-level metric says clean, and the router picks an OCR
engine. The engine reads the printed labels perfectly and the handwritten values as
garbage — and since the labels are the majority of the text, page-level confidence looks
fine. **We extract the field names correctly and the field values wrongly, with high
confidence.**

**What to do.** This is the single strongest argument for region-level rather than
page-level routing, and it needs three things from earlier sections: handwriting regions
(§2.1), region-level routing, and `remove_lines` for the ruled boxes (which already works
and is already reached for when `layout.is_tabular` — though note `is_tabular` is set by
the same layout analysis that does not yet exist).

Region-level routing is the biggest architectural item in this document. Everything else
is a metric or an op; this changes the read layer's shape, because `read_page` currently
picks one backend per page. Worth doing, worth doing carefully, worth doing last of the
P0s.

---

## Part 4 — The confidence problem

**Direct relevance to the routing pattern we use in the office** — OCR first, read the
engine's confidence score, and send it to an LLM if the score is below a threshold.
The instinct is right: spend model budget only where the cheap path is failing. The
mechanism has a specific hole, and it is worth understanding exactly.

### 4.1 What an OCR confidence score actually measures

An OCR engine's per-character confidence is (roughly) the softmax probability of the
chosen class under its own classifier. That is a statement about **how well the input
matched something in the model's hypothesis space** — not about whether the answer is
right. The distinction is invisible on printed text, where the two correlate strongly,
and it becomes severe exactly where we need it most:

- **Handwriting.** A handwritten "5" that happens to resemble a printed "8" produces a
  *confident* "8". The classifier's hypothesis space contains only printed shapes, so it
  reports high confidence for the best match among the wrong candidates. **On
  out-of-distribution input, confidence measures similarity to the wrong reference set.**
  This is not noise around the true answer; it is a systematic bias toward
  overconfidence exactly where the model is least applicable.
- **Missing text.** An engine that finds nothing in a glare region has no low-confidence
  characters to report — it has *no* characters. Page-average confidence over what it did
  find stays high. Confidence is silent about recall, and the field we lost is invisible.
- **The threshold is not comparable across engines,** or across versions of the same
  engine, so a tuned threshold is a tuned-to-one-configuration constant.

Then the second half of the pattern: the LLM has *no* usable confidence at all. A VLM's
self-reported confidence, or its logprobs, measure fluency of the generated string. A
hallucinated but plausible line item scores high because it is plausible — which is the
same thing that made it hallucinate. This is why CLAUDE.md states the invariant: never
surface a model's self-reported confidence as *the* confidence.

So the office pattern has an OCR score that is overconfident on handwriting, an LLM with
no meaningful confidence at all, and a threshold between them tuned on data where both
problems are mild. It works well on the easy 80% and fails silently on our hard 20%.

### 4.2 What the library already does instead

`fuse_confidence` ([docpipe.py:6951](../docpipe.py#L6951)) fuses independent signals in
log-odds space, and the weights ([docpipe.py:7219](../docpipe.py#L7219)) put the two highest on
the two signals that are actually *independent of the reader*:

- **`cross_read_agree` (0.25)** — two independent recognisers producing the same string.
  Independent failures rarely coincide on the same wrong output, which makes agreement
  hard to fake and makes this the strongest available evidence.
- **`validation` (0.25)** — arithmetic and cross-field checks. Completely independent of
  how the page was read. A bill whose line items sum to its total is evidence no
  recogniser can manufacture.

Backend self-confidence is present but weighted below both. The architecture is already
right. What is missing is that **nothing routes hard pages down a path that produces
these signals**. `EnsembleBackend` ([docpipe.py:5974](../docpipe.py#L5974)) exists,
`cross_read_agreement` ([docpipe.py:7050](../docpipe.py#L7050)) exists and even has a fallback
for the VLM case where bboxes are approximate — but `default_router` returns *one*
backend name, so `cross_read_agree` is `None` on every page in the default configuration,
and its 0.25 weight is renormalised away.

### 4.3 What to do

**Make ensemble reading the default for hard pages.** When quality is low, or handwriting
is present, or scripts are mixed, `default_router` should return an ensemble — VLM plus a
local engine, or two different VLMs — and let agreement carry the confidence. Note the
economics work out: those are a *minority* of pages, and on that minority a second read
costs far less than a wrong total.

The rule that falls out of all this, and that I would put on the wall:

> **A single read of a hard page cannot be trusted at any confidence level, because the
> only signals available from one read are signals the reader generates about itself.**

Three concrete steps:

1. Route hard pages to `EnsembleBackend` by default.
2. Prompt VLMs to emit `[illegible]` rather than guess, and treat its presence as a hard
   confidence penalty on the affected span. A model that admits defeat is worth more than
   one that does not.
3. Weight validation harder for numeric documents. On a bill, "line items sum to total" is
   the closest thing to ground truth available without a human — and `_build_field_result`
   already encodes the right asymmetry (a failed check is strong evidence, a passed one
   is weak).

---

## Part 5 — Calibration: the part that is not code

`QualityThresholds` ([docpipe.py:2229](../docpipe.py#L2229)) says it plainly:

> These are starting points from scanned Indian hospital bills and court filings, not
> universal constants. Recalibrate them against a labelled set per document class.

Every threshold in this document — the glare fraction that means unreadable, the
handwriting score that triggers a VLM, the quality score that triggers an ensemble — is a
number somebody has to *choose*. Chosen by intuition they will be wrong in ways that are
invisible until they cost money.

The infrastructure to do this properly already exists and is unused: `EvalSuite`,
`EvalReport.by_page_quality`, `EvalReport.fit_calibrator`, `PlattCalibrator`,
`expected_calibration_error`. The library can measure whether its confidence scores mean
anything. We have never pointed it at our documents.

**~50 labelled pages spanning our actual difficulty range would tell us more than any
single item in Parts 1–4.** Not 50 clean bills — 50 pages deliberately sampled across the
failure modes: glare, handwriting, dot-matrix, third-generation photocopy, stamped total,
mixed script. With that set, every threshold above becomes a fitted number and every
proposed change becomes a measurable delta rather than an argument.

This is the highest-leverage item in the document and the least fun. It is also the only
one that makes the others verifiable.

---

## Part 6 — Proposed order of work

Sequenced by dependency, not by size. Each stage is independently shippable.

| # | Item | Why here | Sections |
|---|---|---|---|
| 1 | Labelled eval set (~50 pages across failure modes) | Everything below is unverifiable without it | Part 5 |
| 2 | `detect_capture_mode` + `detect_page_bounds`, measure after cropping | Frame that every camera metric plugs into; fixes metric accuracy immediately | §1.6, §1.7 |
| 3 | `measure_glare` + `measure_perspective` + `photo_policy` | The two P0 camera failures; op already exists for one | §1.1, §1.2, §1.7 |
| 4 | `detect_handwriting` populating `layout.regions` | Activates the dead router branch — largest accuracy win per line of code | §2.1 |
| 5 | Ensemble-by-default for hard pages + `[illegible]` prompting | Makes confidence mean something; fixes the office routing hole | Part 4 |
| 6 | Script-aware destructive ops; stroke-scaled `despeckle` | Stops the pipeline deleting Devanagari diacritics | §2.2, §3.1 |
| 7 | Op ordering: colour ops before greyscale; `remove_stamps` into `ocr_policy` | Small fix, unblocks an op we already have | §3.2 |
| 8 | Recalibrate `QualityThresholds` and `score` weights against (1) | Now possible, and now meaningful | §3.3, Part 5 |
| 9 | Region-level routing | Biggest architectural change; the pre-printed-form case | §3.4 |
| 10 | Curvature measurement, then dewarping | Genuinely hard, narrowest applicability | §1.3 |

Items 2–4 are one coherent arc and would be my recommendation for the first working
session after this branch merges. Item 1 can start in parallel with anything, since it is
labelling work rather than code, and it gates item 8.

---

## Scope check

Every proposal above is domain-independent, as the library requires. There is no
`HospitalBill` here and no rule that knows what a bill *means*. "This page is a
photograph", "this region is handwritten", "this line mixes scripts", "these pixels are
unrecoverable" are properties of documents in general.

The domain-specific half — that a hospital bill's line items must sum to its total, that a
case number matches a known format, that a date of admission precedes a date of discharge —
belongs in the consuming project as validators and a schema. That is exactly the escape
hatch the library is built to provide, and none of the work above narrows it.
