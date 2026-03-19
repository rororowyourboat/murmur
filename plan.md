# Tasks Plugin — Implementation Plan

Tracking issue: https://github.com/rororowyourboat/murmur/issues/7

## Status

- [x] **Phase 1**: todo.txt backend + basic CLI (`add`, `list`, `done`, `drop`, `move`, `show`, `edit`)
- [ ] **Phase 2**: DSPy task extraction from transcripts
- [ ] **Phase 3**: Cross-meeting task matching + auto-extraction pipeline
- [ ] **Phase 4**: Calendar-aware workflows (agenda, standup, review)
- [ ] **Phase 5**: TaskWarrior backend + summarizer feedback loop

---

## Phase 2: DSPy Task Extraction

Extract action items from meeting summaries/transcripts into the todo.txt store.

### DSPy module: `ExtractTasks`

```python
class ExtractedTask(pydantic.BaseModel):
    title: str
    owner: str = "Unassigned"
    deadline: str = ""
    priority: str = "normal"
    project: str = ""
    source_excerpt: str = ""  # verbatim quote
    confidence: float = 1.0

class MeetingTaskAnalysis(pydantic.BaseModel):
    new_tasks: list[ExtractedTask]
    blockers_raised: list[str]
    blockers_resolved: list[str]

class ExtractTasks(dspy.Signature):
    transcript: str = dspy.InputField()
    existing_tasks: str = dspy.InputField(desc="Current open tasks as context")
    analysis: MeetingTaskAnalysis = dspy.OutputField()
```

### CLI

```bash
murmur tasks ingest <file>          # extract from .summary.md, .txt, or audio file
murmur tasks ingest --dry-run       # preview without saving
```

### Wiring

- Reads memory.md for team/project context (reuse `_get_system_prompt` pattern from summarize.py)
- Reads calendar context via `match_recording_to_event()` for attendee names
- Stores extracted tasks with `source_file` set to the recording path
- Tags extracted tasks with `murmur` tag to distinguish from manual tasks

### What to build

1. `src/murmur/plugins/tasks_extract.py` — DSPy module + ingest logic (lazy imports, same pattern as summarize.py)
2. Add `ingest` subcommand to the tasks group
3. Add `murmur[tasks]` extra that depends on `murmur[ai]` (DSPy + LiteLLM)

---

## Phase 3: Cross-Meeting Task Matching + Auto-Pipeline

Make the system aware of existing tasks so it updates them instead of creating duplicates.

### DSPy module: `MatchTaskMention`

```python
class TaskStatusUpdate(pydantic.BaseModel):
    task_id: str              # matched existing task
    new_status: str = ""
    new_deadline: str = ""
    discussion_context: str = ""
    confidence: float = 1.0

class MatchTaskMention(dspy.Signature):
    mention: str = dspy.InputField(desc="What was said about a task")
    candidates: str = dspy.InputField(desc="Existing tasks JSON")
    match: TaskStatusUpdate = dspy.OutputField()
```

### Auto-extraction pipeline

- Listen on `summary_complete` hook (already emitting from summarize.py)
- Auto-run extraction when `[tasks] auto = true` in config
- Write `.tasks.json` sibling file next to recordings
- Fuzzy-match extracted items against existing open tasks before creating new ones

### What to build

1. `MatchTaskMention` DSPy module in `tasks_extract.py`
2. Hook listener in tasks plugin `register()` — subscribe to `summary_complete`
3. Matching logic: compare extracted task titles against existing tasks using the DSPy matcher
4. Write `.tasks.json` sibling files for TUI integration

---

## Phase 4: Calendar-Aware Workflows

Meeting-driven task views using murmur's calendar plugin.

### Commands

```bash
murmur tasks agenda       # tasks relevant to next calendar event's attendees/project
murmur tasks standup      # generate standup from recent task activity (last 24h)
murmur tasks review       # GTD weekly review: stale inbox, blocked, overdue
```

### `agenda` logic

1. Call `calendar.get_next_event()` to get upcoming meeting
2. Extract attendee names from the event
3. Filter open tasks by those owners + relevant projects
4. Render as a pre-meeting briefing

### `standup` logic

1. Load tasks updated in last 24 hours (compare `created_at` dates, check done tasks)
2. Group into: completed yesterday, working on today, blocked
3. Optionally generate natural language standup via DSPy

### `review` logic

1. Inbox items needing triage (created > 2 days ago, still inbox)
2. Stale active/next tasks (no updates in > 7 days)
3. Overdue tasks (deadline passed)
4. Blocked/waiting tasks
5. Render as a checklist

---

## Phase 5: TaskWarrior Backend + Feedback Loop

### TaskWarrior backend

- Add `murmur[tasks-tw]` extra with `tasklib` dependency
- Backend dispatcher in tasks.py based on `[tasks] backend = "taskwarrior"` config
- Map GTD statuses to TaskWarrior: inbox→pending, next→pending+next, active→started, waiting→waiting, done→completed
- Map projects, tags, priorities directly

### Summarizer feedback loop

Close the loop so the LLM knows about existing tasks during summarization:

1. `murmur tasks export` — writes `~/.config/murmur/task_context.md` with open tasks grouped by project/owner
2. Summarize plugin reads this file alongside memory.md in `_get_system_prompt()`
3. The LLM can then say "Task X (assigned to Bob) was discussed and marked complete" instead of re-extracting it

### Config

```toml
[tasks]
backend = "taskwarrior"     # "todo" | "taskwarrior"
auto = true
export_context = true       # auto-export task context for summarizer

[tasks.taskwarrior]
default_project = "meetings"
default_tags = ["murmur"]
```
