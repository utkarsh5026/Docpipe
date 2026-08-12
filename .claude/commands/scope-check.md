---
description: Audit docpipe.py against its scope boundary — find domain logic that leaked into core
argument-hint: "[section number or symbol to focus on]"
allowed-tools: Bash, Read, Grep, Glob
---

Audit the library against its own scope boundary, as stated in README.md's *Scope boundary*
section. This is the check that keeps a shared substrate from rotting into a grab-bag —
domain leakage and over-generalisation are the two failure modes that kill libraries like
this one.

If `$ARGUMENTS` names a section or symbol, focus there. Otherwise sweep the whole file.

## The test

For every public name in `__all__`, ask:

> Could a team working on a completely unrelated document type use this?

If no, it belongs in a consuming project, not in `docpipe.py`.

## What "domain leakage" looks like here

Hunt for these specifically — they are the ways it actually creeps in:

1. **Named business entities in core.** No `HospitalBill`, no `FIRDocument`, no claim or
   policy vocabulary in a type name, field name or docstring example that only makes sense
   for insurance or legal work. Schemas belong to projects, always.
2. **Business validation dressed as generic validation.** `Validators.line_items_sum_to_total`
   and `Validators.date_order` are legitimate — they are arithmetic and ordering, true of any document with
   line items or a date range. A tariff lookup, a policy-number format, or a payer-specific
   rule is not.
3. **Hardcoded constants that are really one corpus's priors.** Thresholds and fusion weights
   are *priors from scanned Indian hospital bills and court filings*, and the README says so.
   Any new constant of that kind must be overridable and documented as a prior, not presented
   as a universal.
4. **A parameter that exists for exactly one caller.** Configuration sprawl is the
   over-generalisation failure mode. If an option has one plausible user, it should have been
   an escape hatch instead.
5. **Missing escape hatch.** Every layer must be usable standalone, and custom ops, backends
   and routers must be registrable without editing `docpipe.py`. If a consumer would have to
   fork the file to do something reasonable, that is a defect — report it as one.

## Also check the invariants haven't been quietly traded away

- Does any op apply unconditionally rather than correcting *measured* degradation?
- Does any confidence path surface a model's self-reported number as *the* confidence,
  instead of fusing independent signals?
- Does any read path invent bounding boxes rather than marking spans `approximate_bbox`?
- Did a heavy import escape to module scope, or a price into `PRICING`?

```bash
grep -nE "hospital|claim|policy_number|insurer|FIR|tariff|invoice_no" docpipe.py
grep -nE "^(import|from) (numpy|cv2|PIL|fitz|pymupdf|pydantic|anthropic|openai)" docpipe.py
grep -n "PRICING = " -A 6 docpipe.py
```

The first grep will hit docstring *examples* — a `Bill` model in a usage example is fine and
intended, because it shows a consumer's schema. A `Bill` model that core code depends on is
not. Distinguish the two before reporting.

## Report

A short list, most serious first. For each: the symbol and line, which rule it breaks, and
where it should live instead. If nothing leaked, say so plainly and name the two or three
places that came closest — those are where it will leak next.

Do not fix anything in this command. Report only; moving a public symbol out of core is a
breaking change and the user decides it.
