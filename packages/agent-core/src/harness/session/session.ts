import { stripCompactionReplayCheckpoint } from "@openclaw/ai/transports";
import type { AgentMessage } from "../../types.js";
import {
  asAgentMessage,
  createBranchSummaryMessage,
  createCompactionSummaryMessage,
  createCustomMessage,
} from "../messages.js";
import type { CompactionEntry, ResetEntry, SessionContext, SessionTreeEntry } from "../types.js";
import { selectResetKeptEntries } from "./tool-result-pairing.js";

type ContextBoundary = CompactionEntry | ResetEntry;
const SESSION_HISTORY_PRELUDE = Symbol.for("openclaw.sessionHistoryPrelude");

/** Project persisted session entries into the message shared by replay and summarization. */
export function projectSessionEntryMessage(
  entry: SessionTreeEntry,
  options?: { preferReplayContent?: boolean },
): AgentMessage | undefined {
  switch (entry.type) {
    case "message":
      // Private shell history stays persisted but never enters replay or summarization.
      if (entry.message.role === "bashExecution" && entry.message.excludeFromContext === true) {
        return undefined;
      }
      // Model-facing replay only: swap in what was actually sent for this turn
      // when a producer intentionally persists different display text (e.g.
      // heartbeat's constant HEARTBEAT_TRANSCRIPT_PROMPT placeholder). Display,
      // search, and compaction-summary consumers never pass this option, so
      // they keep seeing the persisted `content` unchanged.
      if (
        options?.preferReplayContent &&
        entry.message.role === "user" &&
        entry.message.replayContent !== undefined
      ) {
        return { ...entry.message, content: entry.message.replayContent };
      }
      return entry.message;
    case "custom_message":
      return asAgentMessage(
        createCustomMessage(
          entry.customType,
          entry.content,
          entry.display,
          entry.details,
          entry.timestamp,
        ),
      );
    case "branch_summary":
      return asAgentMessage(
        createBranchSummaryMessage(entry.summary, entry.fromId, entry.timestamp),
      );
    case "compaction":
      return asAgentMessage(
        createCompactionSummaryMessage(entry.summary, entry.tokensBefore, entry.timestamp),
      );
    default:
      return undefined;
  }
}

function stripStalePrefixReplay(message: AgentMessage): AgentMessage {
  return message.role === "assistant" ? stripCompactionReplayCheckpoint(message) : message;
}

function appendContextMessage(
  messages: AgentMessage[],
  entry: SessionTreeEntry,
  options?: { prefixWasRewritten?: boolean; preferReplayContent?: boolean },
): void {
  if (entry.type === "compaction" || (entry.type === "branch_summary" && !entry.summary)) {
    return;
  }
  const message = projectSessionEntryMessage(entry, {
    preferReplayContent: options?.preferReplayContent,
  });
  if (message) {
    messages.push(options?.prefixWasRewritten ? stripStalePrefixReplay(message) : message);
  }
}

function appendResetKeptMessage(
  messages: AgentMessage[],
  entry: SessionTreeEntry,
  options?: { preferReplayContent?: boolean },
): void {
  if (entry.type !== "message") {
    return;
  }
  if (entry.message.role === "user" || entry.message.role === "assistant") {
    const projectedUser =
      options?.preferReplayContent &&
      entry.message.role === "user" &&
      entry.message.replayContent !== undefined
        ? { ...entry.message, content: entry.message.replayContent }
        : entry.message;
    const message = { ...stripStalePrefixReplay(projectedUser) } as AgentMessage & {
      [SESSION_HISTORY_PRELUDE]?: true;
    };
    Object.defineProperty(message, SESSION_HISTORY_PRELUDE, {
      configurable: true,
      enumerable: false,
      value: true,
    });
    messages.push(message);
  } else if (entry.message.role === "toolResult") {
    messages.push(entry.message);
  }
}

/** Build model context from an ordered session branch and its latest state markers. */
export function buildSessionContext(
  pathEntries: SessionTreeEntry[],
  options?: { preferReplayContent?: boolean },
): SessionContext {
  let thinkingLevel = "off";
  let model: { provider: string; modelId: string } | null = null;
  let boundary: ContextBoundary | null = null;

  for (const entry of pathEntries) {
    if (entry.type === "thinking_level_change") {
      thinkingLevel = entry.thinkingLevel;
    } else if (entry.type === "model_change") {
      model = { provider: entry.provider, modelId: entry.modelId };
    } else if (entry.type === "message" && entry.message.role === "assistant") {
      model = { provider: entry.message.provider, modelId: entry.message.model };
    } else if (entry.type === "compaction" || entry.type === "reset") {
      boundary = entry;
    }
  }

  const messages: AgentMessage[] = [];
  if (boundary) {
    if (boundary.type === "compaction") {
      const summary = projectSessionEntryMessage(boundary);
      if (summary) {
        messages.push(summary);
      }
    }
    const boundaryIdx = pathEntries.findIndex((entry) => entry.id === boundary.id);
    const firstKeptIdx = pathEntries.findIndex((entry) => entry.id === boundary.firstKeptEntryId);
    const keptEntries =
      firstKeptIdx >= 0 && firstKeptIdx < boundaryIdx
        ? pathEntries.slice(firstKeptIdx, boundaryIdx)
        : [];
    const replayEntries =
      boundary.type === "reset" ? selectResetKeptEntries(keptEntries) : keptEntries;
    // Both retained-tail forms follow rewritten prefixes, so prefix-bound checkpoints are stale.
    for (const entry of replayEntries) {
      if (boundary.type === "reset") {
        appendResetKeptMessage(messages, entry, {
          preferReplayContent: options?.preferReplayContent,
        });
      } else {
        appendContextMessage(messages, entry, {
          prefixWasRewritten: true,
          preferReplayContent: options?.preferReplayContent,
        });
      }
    }
    for (const entry of pathEntries.slice(boundaryIdx + 1)) {
      appendContextMessage(messages, entry, { preferReplayContent: options?.preferReplayContent });
    }
  } else {
    for (const entry of pathEntries) {
      appendContextMessage(messages, entry, { preferReplayContent: options?.preferReplayContent });
    }
  }

  return { messages, thinkingLevel, model };
}
