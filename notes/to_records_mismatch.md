# `to_records` silent mismatch behavior

## Location
`src/labcore/data/datadict.py` — `DataDictBase.to_records()` (~line 164)

## Issue
When fields have mismatched outer dimensions (e.g. `x=[1,2,3]`, `z=[10,20]`),
`to_records` does not raise. Instead it falls back to `nrecs=1` and wraps all
arrays in an extra outer dimension, treating everything as a single nested record.

## Why it matters
`add_data()` calls `to_records` before `validate()`, so the mismatch is never
caught. A `ValueError` is only raised if you set `values` directly on the dict
and then call `validate()`.

## Options
- Add an explicit length check in `to_records` and raise `ValueError` on mismatch.
- Document the behavior in the docstring so callers know to use `validate()` directly.
