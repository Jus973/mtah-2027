# AGENTS.md — multi-agent coordination ("AgentMax")

**Read this before touching the repo.** `CLAUDE.md`'s data rules bind every agent on every task;
this file adds coordination on top and never overrides it.

Calendar: register team ≤ **10 Nov** · Form arrives ≤ **14 Nov** · deadline **17 Nov 23:59 CET**.
Submit early in the window, never at the wire.

## 0. New agent? Start here (3 minutes)

1. Read `CLAUDE.md`. Its data rules are binding and non-negotiable.
2. Identify yourself: `export BOARD_AGENT=<tool>-<name>` (e.g. `devin-1`, `claude-justin-a`,
   `human-alex`). Unset, you default to `<user>-<host>`.
3. `scripts/board start` — unread announcements, who holds what, what's claimable.
   Requires `python3` and the `gh` CLI, authenticated with access to the **private board repo
   `Jus973/mtah-board`**. A 404 there means you need an invite — ask Justin; that's expected for
   new agents.
4. Pick work: `scripts/board next`, then `scripts/board claim <N>`.
5. Branch `task/<N>-<slug>`, work, and `scripts/board beat <N> "one-line status"` at least every
   25 minutes.
6. Finish: open a PR (template guides you; body contains `Closes Jus973/mtah-board#<N>`), then
   `scripts/board done <N> "what + where" --pr <url>`.
7. Leaving: `scripts/board stop "did … / learned … / next …"` — posts your session note and
   releases your claims.

## 1. Two repos — and what may be said where

|                | `Jus973/mtah-2027` (this repo)      | `Jus973/mtah-board` (private)                                   |
|----------------|-------------------------------------|-----------------------------------------------------------------|
| contains       | code, branches, PRs                 | all issues: tasks, ideas, discussion, announcements, eval log, `DECISIONS.md` |
| visibility     | **public at submission**            | private, stays private                                          |
| allowed talk   | public-safe engineering only        | everything                                                      |

**Hard rule:** no scores, estimates, strategy, risk analysis, or internal debate anywhere on this
repo — not in PR bodies, review comments, commit messages, or issues (don't open issues here at
all). Write "see board #N" and put the substance on the board. Reason: this repo goes public
while the competition is still running.

## 2. The contribution loop (one session = one sweep)

```
sync → work (claim OR default duty) → beat while working → report → cross-pollinate → repeat
```

1. **Sync** — `scripts/board start`.
2. **Work** — claim the top claimable task. **If nothing is claimable you are not done.** Take a
   default duty, lowest-coverage first:
   - **Eval duty** — re-run the sim/fakedata gates on latest `main` (`make test`, `make paired`;
     §6 says what's allowed) and post numbers to the board's 📊 Eval log. Any open PR claiming an
     improvement with no eval posted → run its eval.
   - **Critique duty** — adversarial review of the least-reviewed recent merge, idea, or
     decision. Concrete findings become board issues, not vibes.
   - **Idea duty** — file ONE falsifiable idea on the board (template demands hypothesis /
     verification / expected gain). Check `DECISIONS.md` and the kill-list notes first: most
     owner-side ideas have been tried.
3. **Report** — `board done` with what/where; session note via `board stop`.
4. **Cross-pollinate** — before claiming your next task, leave one substantive comment on an idea
   with fewer than 2 critiques (`scripts/board ideas` lists them).

## 3. Claims, liveness, takeover

| rule             | value                                                                                     |
|------------------|-------------------------------------------------------------------------------------------|
| lease (liveness) | default 60 min; declare what you need (`--lease 3h` for a long silent run; humans `--lease 1d`) |
| heartbeat        | `board beat N "status"` ≤ every 25 min — the status line is mandatory; beats prove *progress*, not just life |
| budget (progress cap) | `size:S` = 2 h · `size:M` = 6 h · `size:L` = 24 h of held time, regardless of beats   |
| takeover is legal when | lease expired **or** budget exhausted — `board takeover N` checks and enforces this; no permission needed |
| three strikes    | 3 failed attempts at the same step → stop, write up what you tried on the issue, `board block N "…"`. A documented dead end is a contribution; silent spinning is not. |
| WIP limit        | 1 in-progress task per agent (reviews exempt)                                             |
| clocks           | GitHub server timestamps only (the script handles this); never trust a local clock        |

Claims, beats, releases, and takeovers are all issue comments, so everything is audited. Races
resolve deterministically: the script re-reads after claiming and auto-retracts if an earlier
live claim exists. Always claim, beat, and take over **through the script** — hand-rolled
comments break the liveness walker. After a takeover, `git fetch` and look for a pushed
`task/N-*` branch before restarting from zero.

## 4. Branches, PRs, review, merge

- One task = one board issue = one branch `task/<N>-<slug>` = one PR against `main`.
- PR body: `Closes Jus973/mtah-board#<N>` + the checklist. Public-safe words only.
- **No numbers, no merge.** A PR claiming any quality effect needs its eval posted to the board
  Eval log and linked from the PR ("eval: board #N / Eval-log comment"). "Should help" doesn't merge.
- **Review:** any agent who didn't author it. Reviews are claims too — comment on the board issue
  (the script's 30-min default lease is fine). Don't wait to be asked.
- **Merge:** 1 non-author approval + green checks → the reviewer merges. **Protected paths need
  Justin's approval:** `evaluation/` (vendored scorer — never modify, period), `configs/final*.yaml`,
  `submissions/`, airlock targets in `Makefile`, `.claude/`, `AGENTS.md`, `scripts/board`, and
  anything touching data handling.
- Agents never commit directly to `main`, never force-push, never change repo settings or
  visibility.

## 5. Communication

- **📢 Announcements** (pinned on the board): broadcast only — schema changed, `main` broken,
  decision made, session notes. Read on every sync; post with `board say "…"`.
- **Task talk** stays on the task's board issue. **Eval numbers** go to the 📊 Eval log
  (`board eval "…"`).
- **Ideas**: board issues with `type:idea`, falsifiable by template. An idea needs **2
  substantive critiques** before it can be promoted to a task.
- **Disagreement protocol**: each side posts position + evidence on the issue → any third agent
  reads both and breaks the tie → genuinely human judgment calls get `needs:human`. Never
  edit-war; never relitigate a `DECISIONS.md` entry without new evidence.
- **Decisions**: once settled, one line in the board repo's `DECISIONS.md` (what, when, issue #).
  Check it before proposing.

## 6. MTAH guardrails (binding)

- `CLAUDE.md` data rules: every line, always. Nothing under `data/`, `cache/`, `$MTAH_DATA_DIR`
  is ever read, printed, or decoded; candidate CSVs included.
- **Unlimited and encouraged**: `make test`, `make fake`, `make paired`, the `sim*` targets,
  open-data work — anything touching only `fakedata/` and open datasets.
- **Airlock only**: every real-data run goes through `make airlock T=<target>`. Never set
  `MTAH_DATA_DIR` yourself.
- **Human-gated, never autonomous**: `make airlock T=public-check` (the public window is scarce
  and pass/fail only), anything under `submissions/`, adopting a new FINAL (written human
  sign-off per the standing protocol), registration, the Form, repo visibility.
- **Pre-registration discipline**: any experiment that consults real-match data gets
  pre-registered as a board issue *before* it runs, and ledger-counted, per the standing protocol.
- Never commit: anything bundle-derived, `handoff/` (local-only by policy), `.mtah.local`.

## 7. Humans

You're an agent with a longer lease. Easiest path: the script — `board claim N --lease 1d` (beats
optional at that lease). Web-only fallback: add label `status:in-progress` plus a comment through
the script later; if you must hand-write, copy the exact comment format of an existing claim. The
labels are the protocol; the script is the ergonomic way in.

## 8. Command reference

| command | does |
|---|---|
| `board start` | sync: announcements, roster, your claims, claimable count |
| `board next` | claimable tasks (prio order, scope conflicts flagged) + duty menu |
| `board claim N [--lease 60m]` | claim, race-safe (`--human` to claim a needs:human task) |
| `board beat N "msg" [--lease …]` | heartbeat + status; extends the lease |
| `board done N "note" [--pr URL] [--close]` | finish → review (or close) |
| `board release N "why"` | hand a task back cleanly |
| `board block N "why"` | mark stuck, with the write-up |
| `board unblock N "why"` | back to todo |
| `board takeover N` | take a stale or over-budget task (script verifies legality) |
| `board who` | roster with beat age, lease, budget, STALE/OVERBUDGET flags |
| `board say "msg"` | post to 📢 Announcements |
| `board eval "numbers"` | post to 📊 Eval log |
| `board ideas` | ideas needing critique |
| `board stop "did / learned / next"` | session note + release your claims |
| `board setup` | idempotent bootstrap (labels, channels) — safe to re-run |
