# Applied AI Engineering — Companion Resources

Lab code, starter files, and reference solutions for the *Applied AI Engineering*
series, published by Nexoma Labs LLC.

This repository holds runnable labs only. It does not contain the book text.

## Series

| Volume | Title | Labs |
|:------:|-------|:----:|
| I | Foundations | [`volume-1/`](volume-1/) — available now |
| II | Prompt Engineering | coming with the volume's release |
| III | Context and Agents | coming with the volume's release |
| IV | Production Systems | coming with the volume's release |

## Running a lab

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nexomalabs/prompt-engineering-resources.git
cd prompt-engineering-resources/volume-1/chapter-03
uv sync
uv run python src/lab.py     # your starting point
uv run pytest                # verify your work
```

Every lab runs from a clean clone with **no API key** — tests replay recorded
responses by default (`NEXOMA_LAB_MODE=fixture`). See each chapter's own
`README.md` for that lab's objectives, expected output, and estimated API cost
if you choose to run it live.

## Structure

```text
volume-1/
├── README.md          — lab index: chapter, type, time, API cost, status
├── shared/             — nexoma_labs package used by every lab (client, cost ledger, fixtures)
└── chapter-01/ … chapter-10/
    ├── README.md        — setup, run, expected output, failure table, API cost
    ├── src/lab.py        — starter with TODOs (chapters with runnable code)
    ├── solution/lab.py    — reference implementation
    ├── tests/              — runs in CI against the solution
    ├── expected/            — captured reference output
    └── fixtures/             — recorded model responses (chapters 8-10)
```

Chapters 1 and 2 are paper exercises: `README.md` only, no code.

## License

Code in this repository is licensed under the [MIT License](LICENSE). The book
text itself is © 2026 Nexoma Labs LLC, all rights reserved — see the volume's
copyright page.

## Errata and issues

- **Errata:** https://books.nexomalabs.com/prompt-engineering-series/errata/
- **Issues:** open one on this repository
- **Email:** books@nexomalabs.com
