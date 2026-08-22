// Replay-content persistence tests cover the heartbeat-continuation replay-stability fix
// (docs/plan/heartbeat-continuation-replay-stability.md): a producer that intentionally
// persists different display text than what it actually sent must still let history
// replay reconstruct byte-identical content for a later turn's model request.
import { describe, expect, it } from "vitest";
import { buildPersistedUserTurnMessage } from "./user-turn-transcript.js";

describe("buildPersistedUserTurnMessage replayContent", () => {
  it("persists replayContent when replayText diverges from the display text", () => {
    const message = buildPersistedUserTurnMessage({
      text: "[OpenClaw heartbeat poll]",
      replayText:
        "System: [2026-08-22 13:32:55 GMT+2] heartbeat one\n\nFollow the scratch context.",
      timestamp: 123,
    });

    expect(message).toMatchObject({ role: "user", content: "[OpenClaw heartbeat poll]" });
    expect(message.replayContent).toBe(
      "System: [2026-08-22 13:32:55 GMT+2] heartbeat one\n\nFollow the scratch context.",
    );
  });

  it("omits replayContent when replayText is identical to the display text", () => {
    const message = buildPersistedUserTurnMessage({
      text: "hello",
      replayText: "hello",
      timestamp: 123,
    });

    expect(message).not.toHaveProperty("replayContent");
  });

  it("omits replayContent when the caller does not supply replayText", () => {
    const message = buildPersistedUserTurnMessage({ text: "hello", timestamp: 123 });

    expect(message).not.toHaveProperty("replayContent");
  });

  it("omits replayContent when replayText is explicitly null", () => {
    const message = buildPersistedUserTurnMessage({
      text: "hello",
      replayText: null,
      timestamp: 123,
    });

    expect(message).not.toHaveProperty("replayContent");
  });
});
