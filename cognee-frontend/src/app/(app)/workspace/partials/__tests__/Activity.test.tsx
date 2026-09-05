import { operationActor, operationStatus } from "../Activity";

describe("activity attribution and completion", () => {
  it("does not attribute an agent operation to the dataset owner", () => {
    expect(operationActor({ user_id: "agent", owner_id: "person", owner_email: "person@example.test" })).toBe("agent");
    expect(operationActor({ user_id: null, owner_id: "person", owner_email: "person@example.test" })).toBe("Not recorded");
  });
  it("does not call a launched background operation completed", () => {
    expect(operationStatus({ outcome: "succeeded", background: true, status: null })).toBe("Started in background");
    expect(operationStatus({ outcome: "failed", background: true, status: null })).toBe("Failed");
    expect(operationStatus({ outcome: null, background: false, status: "PIPELINE_RUN_COMPLETED" })).toBe("Completed");
  });
});
