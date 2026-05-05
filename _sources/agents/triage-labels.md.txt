# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker (`toolsforexperiments/labcore` on GitHub).

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Of these, only `wontfix` currently exists on `toolsforexperiments/labcore`. The other four will be created on the upstream the first time the `triage` skill applies them. Create them ahead of time with:

```bash
gh label create needs-triage     --repo toolsforexperiments/labcore --description "Maintainer needs to evaluate this issue"
gh label create needs-info       --repo toolsforexperiments/labcore --description "Waiting on reporter for more information"
gh label create ready-for-agent  --repo toolsforexperiments/labcore --description "Fully specified, ready for an AFK agent"
gh label create ready-for-human  --repo toolsforexperiments/labcore --description "Requires human implementation"
```

Edit the right-hand column of the table above if you ever decide to remap to existing labels (e.g. reuse `question` as `needs-info`).
