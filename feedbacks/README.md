# Human Feedback Inbox

This folder is the dedicated inbox for human steering and review notes.

## Purpose

- `latest.md` is the current human review / observation note.
- All agents must read `latest.md` before starting any new work.
- Historical notes are archived in `archive/`.

## Required Agent Behavior

Before starting any new work:
1. Check whether `feedbacks/latest.md` exists and is non-empty.
2. If it exists, read it fully before doing anything else.
3. In the agent's own log file, record:
   - `feedback_read: yes`
   - `feedback_source: feedbacks/latest.md`
   - `feedback_summary: <1-3 lines>`
4. Follow the latest human review unless it directly conflicts with the active task/prompt.
5. If there is a conflict, log it in `chat/coordination.md` before proceeding.

## Separation Of Concerns

Human feedback notes must stay in this folder only.

Do not mix systems:
- `feedbacks/latest.md` = human review and steering
- `chat/contracts.md` = boundary/API/schema contract changes only
- `chat/coordination.md` = blockers/requests/handoffs/coordination only

Human feedback notes do NOT go into `chat/contracts.md`.
