# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues on the **canonical upstream**: [`toolsforexperiments/labcore`](https://github.com/toolsforexperiments/labcore). Use the `gh` CLI for all operations.

> **Important:** This clone has two remotes — `origin` (your fork) and `upstream` (`toolsforexperiments/labcore`). Issues live on `upstream`, not `origin`. **Always pass `--repo toolsforexperiments/labcore`** to `gh issue` commands so they don't default to `origin`.

## Conventions

- **Create an issue**: `gh issue create --repo toolsforexperiments/labcore --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --repo toolsforexperiments/labcore --comments`.
- **List issues**: `gh issue list --repo toolsforexperiments/labcore --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --repo toolsforexperiments/labcore --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --repo toolsforexperiments/labcore --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --repo toolsforexperiments/labcore --comment "..."`

## When a skill says "publish to the issue tracker"

Create a GitHub issue on `toolsforexperiments/labcore`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo toolsforexperiments/labcore --comments`.
