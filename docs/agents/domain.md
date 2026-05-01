# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo uses a **single-context** layout: one `CONTEXT.md` at the repo root and one `docs/adr/` directory.

> Note: `labcore` is one of four packages in the [toolsforexperiments ecosystem](https://toolsforexperiments.github.io/guides/software_map.html) — alongside `instrumentserver`, `plottr`, and `CQEDToolbox`. Each package lives in its own git repo with its own single-context setup. Cross-package vocabulary (e.g. how `labcore` relates to `instrumentserver`) belongs as a short "Ecosystem position" section in this repo's eventual `CONTEXT.md`, not as a separate context.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If any of these files don't exist yet, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

```
/
├── CONTEXT.md              ← domain glossary (sweep, DataDict, DDH5Writer, …)
├── docs/
│   ├── adr/                ← architectural decisions
│   │   ├── 0001-….md
│   │   └── 0002-….md
│   └── …                   ← existing Sphinx docs (unrelated; coexists)
└── src/labcore/
```

The existing Sphinx site under `docs/` is unrelated to `CONTEXT.md` and `docs/adr/` — they coexist. Sphinx will ignore `docs/adr/` unless you explicitly include it in `conf.py`.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (storage format) — but worth reopening because…_
