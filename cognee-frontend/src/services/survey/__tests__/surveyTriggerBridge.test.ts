import { notifySurveyTrigger, setSurveyTriggerListener } from "../surveyTriggerBridge";

describe("surveyTriggerBridge", () => {
  afterEach(() => {
    setSurveyTriggerListener(null);
  });

  it("delivers the event to the active listener", () => {
    const listener = jest.fn();
    setSurveyTriggerListener(listener);

    notifySurveyTrigger("datasource_added");

    expect(listener).toHaveBeenCalledWith(expect.objectContaining({ reason: "datasource_added" }));
  });

  it("does not throw when no listener is registered", () => {
    setSurveyTriggerListener(null);

    expect(() => notifySurveyTrigger("datasource_added")).not.toThrow();
  });

  it("does not throw when the listener itself throws", () => {
    setSurveyTriggerListener(() => {
      throw new Error("listener exploded");
    });

    expect(() => notifySurveyTrigger("datasource_added")).not.toThrow();
  });

  it("stops delivering events after the listener is cleared", () => {
    const listener = jest.fn();
    setSurveyTriggerListener(listener);
    setSurveyTriggerListener(null);

    notifySurveyTrigger("datasource_added");

    expect(listener).not.toHaveBeenCalled();
  });
});
