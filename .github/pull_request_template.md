Closes Jus973/mtah-board#<!-- board issue number -->

## What

<!-- public-safe engineering summary only — this repo goes public at submission.
     Scores, estimates, strategy, and discussion live on the board issue. -->

## Checklist (AGENTS.md)

- [ ] Claimed on the board before starting (`scripts/board claim N`)
- [ ] Branch is `task/<N>-<slug>`; one task, one PR
- [ ] No scores / strategy / internal debate anywhere in this PR — "see board #N"
- [ ] CLAUDE.md data rules respected (nothing under data/ read; real-data via airlock only)
- [ ] Nothing bundle-derived committed (pre-commit passes)
- [ ] If this claims an improvement: eval posted to the board 📊 Eval log and linked here
- [ ] Protected paths untouched (`evaluation/`, `configs/final*.yaml`, `submissions/`,
      airlock targets, `.claude/`, `AGENTS.md`, `scripts/board`) — or Justin is the reviewer
