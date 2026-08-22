import type { AgentMessage } from "../../../../packages/agent-core/src/types.js";

export function sessionMessagesContainIdempotencyKey(
  messages: AgentMessage[],
  idempotencyKey: string,
): boolean {
  return messages.some(
    (message) =>
      typeof (message as { idempotencyKey?: unknown }).idempotencyKey === "string" &&
      (message as { idempotencyKey?: unknown }).idempotencyKey === idempotencyKey,
  );
}

export function reconcilePrePersistedCurrentUserTurn(params: {
  activeSession: { agent: { state: { messages: AgentMessage[] } } };
  currentUserTurnMessage: AgentMessage | undefined;
  durableUserTurnMessage: AgentMessage | undefined;
  userTurnAlreadyPersisted: boolean;
  /** This turn's own durable transcript entry id, from the recorder's admission receipt. */
  currentEntryId?: string;
  /** The orphan-repair candidate's durable transcript entry id. */
  candidateEntryId?: string;
}): boolean {
  // Same-process identity: the recorder's own admission receipt names the exact
  // durable row it just wrote. If orphan-repair's tree walk lands on that same
  // row, it is this turn's own just-persisted message, not a stale prior turn --
  // true even when this turn's producer never mints a durable idempotencyKey
  // (heartbeat and other internal-origin turns intentionally do not; see
  // shouldMintChannelSourceTurnId). Without this, every heartbeat turn's own
  // message was misidentified as an orphan of itself, merging its own content
  // into its own prompt on every firing. This does not survive a gateway
  // restart between persistence and this check, since the in-memory admission
  // receipt does not survive the process; the idempotencyKey path below is the
  // cross-restart fallback for producers that mint one.
  if (params.currentEntryId !== undefined && params.currentEntryId === params.candidateEntryId) {
    const messages = params.activeSession.agent.state.messages;
    const tail = messages.at(-1);
    if (tail?.role === "user") {
      params.activeSession.agent.state.messages = messages.slice(0, -1);
    }
    return true;
  }

  const idempotencyKey = (params.currentUserTurnMessage as { idempotencyKey?: unknown } | undefined)
    ?.idempotencyKey;
  if (typeof idempotencyKey !== "string" || idempotencyKey.length === 0) {
    return false;
  }
  const durableIdempotencyKey = (
    params.durableUserTurnMessage as { idempotencyKey?: unknown } | undefined
  )?.idempotencyKey;
  // Recorder state is process-local; after restart the durable keyed leaf is the
  // authoritative proof that this exact admitted turn was already persisted.
  const durableTurnMatches = durableIdempotencyKey === idempotencyKey;
  if (!params.userTurnAlreadyPersisted && !durableTurnMatches) {
    return false;
  }
  const messages = params.activeSession.agent.state.messages;
  const tail = messages.at(-1) as (AgentMessage & { idempotencyKey?: unknown }) | undefined;
  const activeTailMatches = tail?.role === "user" && tail.idempotencyKey === idempotencyKey;
  if (!activeTailMatches && !durableTurnMatches) {
    return false;
  }
  if (activeTailMatches) {
    // Persistence is recorder-owned; either synchronized representation can
    // prove identity. Remove the active copy when present so the model sees it once.
    params.activeSession.agent.state.messages = messages.slice(0, -1);
  }
  return true;
}
