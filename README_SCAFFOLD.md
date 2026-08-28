# How to use this scaffold

This folder is the starting state for Claude Code. It contains no notebooks yet —
Claude Code generates everything (library, tests, data, 15 chapter notebooks,
requirements, validation report) from these instruction files.

## Files

| File | Purpose |
|---|---|
| `CLAUDE.md` | Read automatically by Claude Code. Operating rules: Colab targeting, repo layout, artifact contract, execution discipline, runtime budgets. |
| `SPEC.md` | Your full master prompt, verbatim — the authoritative content spec. |
| `CHAPTERS.md` | The 15-chapter decomposition with SPEC-part mapping and artifact dependencies. |
| `KICKOFF_PROMPT.txt` | Paste this as your first (and only) message in Claude Code. |
| `scripts/validate_all.py` | Fresh-kernel execution of all notebooks → `VALIDATION_REPORT.md`. Claude Code runs it; you can too. |
| `Makefile` | `make deps`, `make test`, `make validate`, `make validate-one CH=06`. |

## Steps

1. Copy this whole folder somewhere, open it in Claude Code desktop.
2. Recommended: run in a project with permissions that let it install packages and
   execute notebooks without approval prompts (otherwise "without me acting in" fails
   at the first pip install). E.g. start with `claude --dangerously-skip-permissions`
   in a container/VM, or pre-approve Bash/Edit in settings.
3. Paste the contents of `KICKOFF_PROMPT.txt`. This will be a long run (research +
   15 executed notebooks) — expect it to take a while and possibly need a "continue"
   if it hits a session limit, but no decisions from you.
4. When done, check `VALIDATION_REPORT.md` says `SERIES VALIDATION: PASS` and skim a
   couple of notebooks' rendered outputs.

## Getting it onto Colab afterwards

1. Push the finished repo to GitHub (public or private).
2. Fill in the `REPO_URL` constant that Claude Code leaves in the bootstrap cell
   (it's the same cell in every notebook — one sed does all 15).
3. Open any chapter notebook in Colab (File → Open → GitHub) and Run All. Each
   notebook self-installs its deps and rebuilds any missing pipeline artifacts from
   the bundled corpus, so chapters work independently on fresh Colab VMs.

## If a chapter fails later

`make validate-one CH=NN` re-executes just that notebook fresh, or tell Claude Code:
"Chapter NN fails at cell X with <error> — fix and re-validate that chapter."
