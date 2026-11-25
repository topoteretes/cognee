import {
  clearSearchState,
  readSearchState,
  subscribeSearchState,
  updateSearchState,
} from "@/modules/searchHistory/conversationStore";
import { getSearchActiveConversation, setSearchActiveConversation } from "@/utils/browserStorage";
import type { Conversation } from "@/modules/searchHistory/conversations";

const POD = "instance-1";

function conversation(id: string): Conversation {
  return { id, title: id, messages: [], createdAt: "2026-09-03T11:00:00.000Z", updatedAt: "2026-09-03T11:00:00.000Z" };
}

describe("conversationStore", () => {
  afterEach(() => {
    clearSearchState();
    sessionStorage.clear();
  });

  it("starts empty for a pod it has not seen", () => {
    expect(readSearchState(POD)).toEqual({ conversations: [], activeId: null, inFlightId: null });
  });

  it("restores the open conversation remembered for this tab", () => {
    setSearchActiveConversation(POD, "search-ui-7");

    expect(readSearchState(POD).activeId).toBe("search-ui-7");
  });

  it("notifies subscribers and hands them the new snapshot", () => {
    const listener = jest.fn();
    subscribeSearchState(POD, listener);

    updateSearchState(POD, (state) => ({ ...state, conversations: [conversation("search-ui-1")] }));

    expect(listener).toHaveBeenCalledTimes(1);
    expect(readSearchState(POD).conversations.map((c) => c.id)).toEqual(["search-ui-1"]);
  });

  it("remembers a change of open conversation for the next reload", () => {
    updateSearchState(POD, (state) => ({ ...state, activeId: "search-ui-2" }));
    expect(getSearchActiveConversation(POD)).toBe("search-ui-2");

    updateSearchState(POD, (state) => ({ ...state, activeId: null }));
    expect(getSearchActiveConversation(POD)).toBeNull();
  });

  it("treats an updater that returns the same state as a no-op", () => {
    const listener = jest.fn();
    subscribeSearchState(POD, listener);

    updateSearchState(POD, (state) => state);

    expect(listener).not.toHaveBeenCalled();
  });

  it("stops notifying once unsubscribed", () => {
    const listener = jest.fn();
    const unsubscribe = subscribeSearchState(POD, listener);
    unsubscribe();

    updateSearchState(POD, (state) => ({ ...state, inFlightId: "search-ui-1" }));

    expect(listener).not.toHaveBeenCalled();
  });

  it("keeps pods apart", () => {
    updateSearchState(POD, (state) => ({ ...state, activeId: "search-ui-1" }));

    expect(readSearchState("instance-2").activeId).toBeNull();
  });

  it("forgets everything, including the remembered conversation, when cleared", () => {
    updateSearchState(POD, (state) => ({ ...state, activeId: "search-ui-1" }));

    clearSearchState();

    expect(getSearchActiveConversation(POD)).toBeNull();
    expect(readSearchState(POD).activeId).toBeNull();
  });
});
