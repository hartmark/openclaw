# Make HTTP continuation replay-stable for heartbeat (and any other

# display-vs-sent divergent) turns

## Summary

HTTP continuation (`compat.supportsResponsesContinuation`, see
`docs/plan/openai-responses-http-continuation-cache.md`) never engages for
heartbeat-triggered traffic — not intermittently, but on every single
heartbeat turn beyond the first, 100% reproducibly. Root cause: heartbeat
turns are **persisted** to the session transcript as a constant placeholder
string, deliberately different from what is **sent** to the model. HTTP
continuation's replay comparison requires the two to be byte-identical.
They never are, by design, so continuation falls back to full history on
every heartbeat turn, forever.

This was found and confirmed via an isolated repro (a real openclaw gateway
process, no test framework, pointed at a scripted mock OpenAI-Responses
backend) plus direct correlation against Ping's real production logs and
SQLite transcript state. Every claim below is cited to source and backed by
an empirical trace, not inferred.

## Root cause, traced end to end

**1. Heartbeat's transcript entry is a hardcoded constant, unconditionally.**

`src/auto-reply/heartbeat.ts:16`:

```ts
export const HEARTBEAT_TRANSCRIPT_PROMPT = "[OpenClaw heartbeat poll]";
```

`src/auto-reply/reply/prompt-prelude.ts:224-228`:

```ts
const transcriptBody = params.isHeartbeat
  ? HEARTBEAT_TRANSCRIPT_PROMPT
  : params.isBareSessionReset
    ? softResetTail || `[OpenClaw session ${params.startupAction}]`
    : (roomEventBody ?? (params.hasUserBody ? params.baseBody : MEDIA_ONLY_USER_TEXT));
```

For every heartbeat-triggered turn, regardless of what actually gets sent to
the model (a custom `text` payload, pending cron/exec events, scratch-file
directives, timestamps — all real, turn-specific content), the value
persisted to the session transcript is always the literal string
`"[OpenClaw heartbeat poll]"`. This is very likely intentional: it keeps the
human-visible Control UI transcript readable instead of cluttered with
internal heartbeat machinery on every firing. It is a real product decision,
not an oversight.

**2. The model-bound prompt is the real, turn-specific content, and already
diverges from the persisted transcript by design.**

`src/agents/embedded-agent-runner/run/attempt-prompt-build.ts` builds two
separate values per turn: `promptForSession` (persisted) and `promptForModel`
(sent). Line ~605:

```ts
const currentUserTimestampOverride =
  !input.isRawModelRun && typeof preparedUserTurnTimestamp === "number"
    ? {
        timestamp: preparedUserTurnTimestamp,
        text: promptForSession,
        ...(promptForModel !== promptForSession ? { alternateText: promptForModel } : {}),
      }
    : undefined;
```

`alternateText` already captures exactly the right value (what was actually
sent) at exactly the right point in the pipeline. It is computed today
purely to let `attempt-history.ts:99` and `attempt-llm-boundary.ts:442` match
the _current_ turn's freshly-submitted text against a timestamp-override
candidate. It is never persisted, and never read back for replay.

The same `promptForSession !== promptForModel` divergence — for a different
reason — also happens for orphan-repair merges
(`src/agents/embedded-agent-runner/run/attempt-prompt-helpers.ts`,
`mergeOrphanedTrailingUserPrompt`): when a prior turn's trailing user message
was never answered (see §3), the next turn merges it in with a
`[Queued user message from a previous active turn...]` wrapper for the model
call, while the persisted transcript entry stays unwrapped. Both divergence
sources land on the same gap: **there is no durable record of what was
actually sent, separate from what's displayed.**

**3. Why heartbeat turns are so often "orphaned" (a compounding, not
required, factor).**

Independently of §1/§2, heartbeat firings frequently leave their own
trailing "user" transcript leaf unanswered — confirmed via direct SQLite
inspection of `transcript_events` in the isolated repro: two separate
root-parented chains appeared after a single clean `wake` call, one from an
automatic heartbeat check at gateway startup that never reached a model call
(the mock backend's first scripted reply was consumed by the _next_ firing,
not the automatic one). `attempt-orphan-repair.ts`'s
`findTrailingMessageEntryForOrphanRepair` then finds that dangling leaf on
the _next_ turn and merges it. Ping's real logs show this "Merged and
removed orphaned user message to prevent consecutive user turns" event on
essentially every heartbeat firing (133/133 sampled,
`trigger=heartbeat`) — this is steady-state behavior, not an edge case.

This third factor is not required to reproduce the bug (§1/§2 alone
guarantee `history_changed` on every heartbeat turn), but it means the
model-bound prompt for a "normal" heartbeat turn is _also_ routinely
different from a naive replay in a second, independent way, which the fix
needs to cover too.

**4. Where this breaks continuation.**

`packages/ai/src/transports/openai-responses-continuation.ts`,
`resolveResponsesContinuationRequest`, requires the new turn's `input`
prefix to be byte-identical (via `jsonValuesEqual`, after
`normalizeAssistantReplayInput`) to what was stored as `lastRequest.input`
from the previous turn's _actually sent_ request. The prefix gets rebuilt
for a new turn by replaying the session transcript
(`packages/agent-core/src/harness/session/session.ts`,
`projectSessionEntryMessage`, returns `entry.message` — i.e. the _persisted_
`content` — verbatim). Persisted content is `HEARTBEAT_TRANSCRIPT_PROMPT`;
sent content was the real prompt. They can never match. Status is always
`history_changed`, silently, with no log line from the continuation code
itself (matching the "zero evidence of any attempt" signature originally
found on live wire capture).

Reproduced directly: two-turn isolated test, `claim-result` on turn 2 is
`status=history_changed`, and the mismatch diff shows exactly this —
`"[OpenClaw heartbeat poll]"` (replayed from transcript) vs. the real sent
text (`"[Queued user message from a previous active turn...]"` once an
orphan is also present, or the real heartbeat prompt body otherwise).

## Why this is a design fork, not a one-line fix

`resolveResponsesContinuationRequest`'s correctness _requires_ "replay of
transcript == what was sent." `prompt-prelude.ts` _deliberately_ violates
that for heartbeat, for a legitimate, separate reason (transcript
readability). The two are incompatible as currently built. Two honest ways
to resolve that:

- **A — Exclude heartbeat from HTTP continuation eligibility.** Small, safe,
  respects the existing transcript-simplification design as-is. Costs: gives
  up continuation's token/latency savings on exactly the highest-volume,
  most continuation-eligible traffic shape (heartbeat cadence), which was
  the original motivation for chasing this.
- **B — Persist a replay-authoritative variant of what was sent, separate
  from the display text, and use it (only) when rebuilding history for a new
  model request.** Preserves both the clean transcript _and_ continuation's
  benefit. Bigger: touches the shared session-entry shape and the transcript
  replay/projection layer used by every agent run, not just heartbeat.

Chosen direction: **B**, on the basis that it is the architecturally correct
fix (it repairs the actual invariant gap — "no durable record of what was
sent" — rather than routing around a symptom), and it is the one that
recovers the win. Scope is expected to be trimmed for the first landed PR;
see Phasing below.

## Proposed design (B)

### 1. Give `UserMessage` a durable, optional replay-authoritative field

`packages/llm-core/src/types.ts:320-333`:

```ts
export interface UserMessage {
  role: "user";
  content: string | (TextContent | ImageContent)[];
  timestamp: number;
  runtimeContextCarrier?: boolean;
  /** What was actually sent to the model for this turn, when it differs from
   * `content` (the persisted/display text). Populated whenever a producer
   * intentionally diverges display from sent content (heartbeat's constant
   * transcript placeholder, orphan-repair merges). Replay/history-rebuild
   * must prefer this over `content` so a later turn's request reconstructs
   * byte-identical bytes to what this turn's request actually carried —
   * required for HTTP continuation's replay comparison
   * (openai-responses-continuation.ts) to ever succeed past turn one. */
  replayContent?: string | (TextContent | ImageContent)[];
}
```

Naming note: this is deliberately _not_ called `alternateText` — that name
is already used for a narrower, currently ephemeral, current-turn-only
concept (`CurrentUserTimestampOverride.alternateText`,
`attempt-prompt-build.ts:452`). `replayContent` names the durable,
replay-facing concept precisely; the existing `alternateText` computation
becomes the value that feeds it (see step 2), not a rename of it.

### 2. Persist it wherever `promptForSession !== promptForModel` is already computed

`attempt-prompt-build.ts` already computes exactly this comparison for
`alternateText` (line ~605, cited above). Thread that same value into the
durable write path: `SessionManagerEntries.appendMessage`
(`src/agents/sessions/session-manager-entries.ts:128`), the sole
transcript-write entry point per CLAUDE.md's contract in
`src/gateway/server-methods/CLAUDE.md` ("Always write transcript messages
via `SessionManager.appendMessage(...)`"). Needs a short audit of the
`userTurnTranscriptRecorder` call site that invokes it for the current-turn
user message specifically, to confirm where `promptForModel` is available
alongside the message being appended.

### 3. Read it back — only for model-facing replay, never for display

`packages/agent-core/src/harness/session/session.ts`,
`projectSessionEntryMessage` (line 16-22 today):

```ts
case "message":
  return entry.message.role === "bashExecution" && entry.message.excludeFromContext === true
    ? undefined
    : entry.message;
```

This function is the single shared projection used both to build
`activeSession.agent.state.messages` (which flows into the actual model
request) _and_, today, into anything else that walks session history. That
dual use is exactly the trap: a naive "always prefer `replayContent`" edit
here would leak the replay-only content into transcript display, dashboard
history views, and compaction summaries meant for human review — the
opposite of what `HEARTBEAT_TRANSCRIPT_PROMPT` exists to prevent.

The fix must distinguish the two consumers. Concretely: add a
`{ preferReplayContent?: boolean }` option to `buildSessionContext` /
`projectSessionEntryMessage`, default `false` (today's behavior, unchanged,
so display/compaction/search paths need no changes), and have the one
caller that assembles the model-bound `input`/context for a live request
(the call site that ultimately feeds `convertResponsesMessages`, per
`openai-responses-replay-messages-internal.ts`) pass `true`. When true, and
`entry.message.replayContent` is present, project a message with `content:
entry.message.replayContent` instead of `entry.message.content`.

### 4. No SQLite schema-version bump required

`replayContent` is a new, optional key inside `entry_json`
(`transcript_events.event_json`), not a new SQL column. Per
`docs/reference/database-schemas.md`'s additive-surface rules, an optional
JSON key that older readers simply ignore is safe at the same schema
version. No migration, no version bump, no separate schema-change approval
gate — confirm this reading against that doc before landing, but nothing in
this design requires DDL changes.

### 5. Continuation-cache interaction

No changes needed in
`packages/ai/src/transports/openai-responses-continuation.ts` itself. Once
replay correctly reconstructs `replayContent` for historical turns,
`resolveResponsesContinuationRequest`'s existing byte-comparison should
succeed naturally — the fix is entirely upstream of the continuation cache.

## Phasing (scope to trim for the first PR)

Full design touches: `llm-core` types, the transcript-write call site,
`agent-core`'s session projection, and the Responses-replay call site that
opts into `preferReplayContent`. That is a reasonable amount of surface for
one PR, but if it needs trimming further:

- **Phase 1 (minimum to fix heartbeat, the confirmed 100%-reproducible
  case) — implemented and verified:** wire `replayContent` end-to-end for
  the heartbeat (`HEARTBEAT_TRANSCRIPT_PROMPT`) divergence only. Isolated
  repro confirmed `replayContent` persists correctly and history replay
  correctly substitutes it (`preferReplayContent: true`) at both call
  sites that build model-facing context (`sdk.ts`'s initial session
  restore, and `attempt-session-prepare.ts`'s orphan-repair rebuild).
- **Phase 2 — implemented and verified, root cause revised from the
  original plan below:** the original plan assumed the remaining
  divergence was `mergeOrphanedTrailingUserPrompt`'s merge output not
  being captured for replay, and proposed extending the same
  `replayContent` plumbing to it. Re-tracing with the isolated repro (two
  sequential heartbeat turns, same live gateway process, no restart)
  found a different, more fundamental defect instead:
  - **Root cause A (self-collision):**
    `reconcilePrePersistedCurrentUserTurn`
    (`src/agents/embedded-agent-runner/run/pre-persisted-user-turn.ts`)
    is the guard that is supposed to recognize "the orphan-repair tree
    walk landed on this turn's own already-persisted message, not a
    stale prior turn" — but it only proves that via a durable
    `idempotencyKey` match. Heartbeat (and every internal-origin turn)
    intentionally never mints one
    (`shouldMintChannelSourceTurnId` in
    `src/auto-reply/reply/source-turn-id.ts`), so the guard always fell
    through to `false` for heartbeat, and _every_ heartbeat turn's own
    message was misidentified as an orphan of a _different_ prior turn —
    even the very first heartbeat ever, with no restart or crash
    involved. `mergeOrphanedTrailingUserPrompt` then merged the turn's
    own content into its own prompt under the queued-message marker,
    which is what Root cause #2 in the original trace above actually
    observed in production 133/133 times.
  - **Fix A:** `reconcilePrePersistedCurrentUserTurn` gained a
    same-process identity check: the recorder's own admission-receipt
    `entryId` (`TranscriptEntryAnchor.entryId`, the durable row id issued
    by the SQLite append transaction) compared directly against the
    orphan-repair candidate's tree-entry id. A match is authoritative
    proof regardless of `idempotencyKey`. This does not survive a
    gateway restart between persistence and this check (the in-memory
    admission receipt does not survive the process); the existing
    `idempotencyKey` path remains the fallback for producers that mint
    one, including across restarts.
  - **Root cause B (persisted duplicate never discarded):** fixing A
    alone did not fix continuation — turn 2 still showed `history_changed`,
    now as a narrower `reply-mismatch`. Cause: the admission-time persist
    (an early, eager write for immediate transcript visibility) and the
    embedded-agent-runner's own later "runtime" append both write durable
    rows for the same logical turn, chained on the active branch.
    `reconcilePrePersistedCurrentUserTurn` already stripped the
    in-memory duplicate from `activeSession.agent.state.messages` (so the
    turn's own model call only ever saw it once) but never removed it
    from the _persisted_ branch, so every later turn's history replay
    reconstructed two consecutive user-role entries for one logical turn,
    shifting `resolveResponsesContinuationRequest`'s positional
    prefix/reply comparison.
  - **Fix B:** `attempt-session-prepare.ts` now performs the same
    discard (`sessionManager.branch()` to the candidate's parent, or
    `resetLeaf()` if the candidate was a root, then
    `replayTrailingEntriesForOrphanRepair`) whenever a candidate is found
    and removable — not only in the genuine-orphan case, but also in the
    reconciled-duplicate case. Only a genuine orphan additionally merges
    its text into the current prompt; a reconciled duplicate has nothing
    to merge, it is simply superseded by the runtime's own upcoming
    append.
  - Verified via the isolated repro: two sequential heartbeat turns, same
    live gateway process, no restart — turn 2's
    `claimOpenAIResponsesHttpContinuation` now reports
    `status=continued, claimHasPrevRespId=true, claimInputLen=1`.
- **Phase 3 (optional, only if Phase 1+2 don't fully explain remaining
  production `history_changed` cases):** audit for other
  `promptForSession !== promptForModel` producers not yet covered, and
  confirm whether root cause B above (persisted duplicate not discarded)
  also explains the previously-investigated general-room 275-turn session
  case, independent of heartbeat — plausible since the reconciliation gap
  it fixes is not heartbeat-specific, but not yet directly confirmed
  against that session.

## Evidence trail / how to reproduce

- Isolated repro: real openclaw gateway (`node dist/index.js gateway`,
  isolated `OPENCLAW_STATE_DIR`, no operator gateway touched), pointed at a
  scripted mock OpenAI-Responses HTTP+SSE backend
  (`~/code/claude/omniroute-continuation-repro/mock-responses-server.mjs`,
  extended to script more than 3 turns), with
  `compat.supportsResponsesContinuation: true` set on the mock model, driven
  via `gateway call wake --params '{"mode":"now",...}'` for two consecutive
  turns on the same session.
- Debug instrumentation added temporarily (and removed) at three points to
  get ground truth instead of guessing from static reads: the continuation
  claim/commit/status in `openai-responses-client.ts` +
  `openai-responses-continuation.ts` (routed through the existing
  `emitModelTransportDebug` structured logger — raw `process.stderr.write`
  in a worker subprocess does not reach the parent's captured output, a real
  gotcha hit early in this investigation), and the
  `promptForSession`/`promptForModel`/`attempt.transcriptPrompt` values in
  `attempt-prompt-build.ts`.
- Direct SQLite inspection of the isolated repro's own
  `agents/main/agent/openclaw-agent.sqlite` (`transcript_events`,
  `session_nodes`) confirmed the two-root-chain orphan pattern independent
  of any test-harness retry artifact.
- Cross-checked against Ping's real production gateway logs (read-only;
  her live gateway process was never restarted or modified for this
  investigation) and OmniRoute's own call-log/dashboard API for the same
  session, confirming the same failure signature and the "Merged and
  removed orphaned user message" event on 133/133 sampled heartbeat firings.
- Phase 2's revised root cause (self-collision + persisted duplicate) was
  found the same way: temporary logging (also added and removed) at
  `reconcilePrePersistedCurrentUserTurn`'s call site in
  `attempt-session-prepare.ts` and in the guard wrapper's
  `onUserMessagePersisted` callback, plus repeated direct SQLite inspection
  of `transcript_events` across several isolated-repro rebuild/relaunch
  cycles, to trace exactly which entry ids the orphan-repair tree walk and
  the recorder's own admission receipt disagreed on.

## Out of scope

- OmniRoute's own conversation-reconnection fragmentation
  (`open-sse/services/conversationTracker.ts`, `findReconnectMatch`) is a
  related but independently-owned symptom of the same class of bug (content
  no longer content-hash-matches after a divergent turn) in a different
  repo. Expected to improve as a side effect once this lands, not something
  this change touches directly.
- A dashboard feature to visualize genuine cross-session conversation merges
  (e.g. a user replying in one room to a message originally posted from a
  different session) is a distinct, legitimate product idea raised during
  this investigation, not a bug fix, and not part of this change.
- Excluding heartbeat from continuation eligibility (design option A) is not
  pursued here, but remains available as a fast, low-risk fallback if option
  B's scope proves too large to land in a reasonable timeframe.
