const mockUseCogniInstance = jest.fn();
const mockUseTenant = jest.fn();
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => mockUseCogniInstance(),
  useTenant: () => mockUseTenant(),
}));

const mockUseFilter = jest.fn();
jest.mock("@/ui/layout/FilterContext", () => ({
  useFilter: () => mockUseFilter(),
}));

const mockListSearchSessions = jest.fn();
const mockGetSessionDetail = jest.fn();
jest.mock("@/modules/sessions/getSessions", () => ({
  SEARCH_SESSION_PREFIX: "search-ui-",
  listSearchSessions: (...args: unknown[]) => mockListSearchSessions(...args),
  getSessionDetail: (...args: unknown[]) => mockGetSessionDetail(...args),
}));

const mockGetPodUserId = jest.fn();
jest.mock("@/modules/users/getPodUserId", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetPodUserId(...args),
}));

const mockRecallKnowledge = jest.fn();
jest.mock("@/modules/datasets/recallKnowledge", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockRecallKnowledge(...args),
}));

const mockSearchCode = jest.fn();
jest.mock("@/modules/datasets/searchCode", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockSearchCode(...args),
}));

jest.mock("@/modules/analytics", () => ({
  TrackPageView: () => null,
  trackEvent: jest.fn(),
}));

jest.mock("@/ui/elements/BrainSelector", () => ({
  __esModule: true,
  default: () => <div data-testid="brain-selector" />,
}));

jest.mock("@/ui/elements/UpgradeBanner", () => ({
  __esModule: true,
  default: () => <div data-testid="upgrade-banner" />,
}));

import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { clearHistoryCache, readHistoryCache } from "@/modules/searchHistory/historyCache";
import { clearSearchState } from "@/modules/searchHistory/conversationStore";
import { setSearchActiveConversation, setSearchType } from "@/utils/browserStorage";
import { trackEvent } from "@/modules/analytics";
import SearchPage from "../SearchPage";
import type { RecallStreamHandlers } from "@/modules/datasets/recallKnowledge";

const EMPTY_STATE = "What are you looking for today?";
const QUESTION = "What is in my graph?";
const LOADING_HISTORY = { name: "Loading past conversations" } as const;

const SESSION = {
  session_id: "search-ui-1700000000000",
  user_id: "user-1",
  dataset_id: null,
  status: "completed",
  effective_status: "completed",
  started_at: "2026-09-01T10:00:00.000Z",
  last_activity_at: "2026-09-01T10:05:00.000Z",
  ended_at: null,
  tokens_in: 0,
  tokens_out: 0,
  cost_usd: 0,
  error_count: 0,
  last_model: null,
};

function detailWith(qas: Record<string, unknown>[], label: string | null = null) {
  return { ...SESSION, label, msg_count: qas.length, tool_calls: 0, traces: [], qas };
}

/**
 * The sidebar, so a conversation title is not confused with the same text
 * rendered as a chat bubble — an unscoped query matches both and throws.
 */
function sidebar(): HTMLElement {
  return screen.getByText("History").closest("div")?.parentElement as HTMLElement;
}

/**
 * Lets queued effects and promise callbacks commit before asserting.
 *
 * A macrotask, not `Promise.resolve()`: an effect reacting to loaded history
 * commits a tick after the DOM write, so a microtask flush asserts before the
 * regression this suite guards would have happened.
 */
async function settle(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

describe("SearchPage", () => {
  beforeEach(() => {
    mockUseCogniInstance.mockReturnValue({
      cogniInstance: { fetch: jest.fn(), instanceId: "instance-1" },
      isInitializing: false,
    });
    mockUseTenant.mockReturnValue({ hasAccess: true, tenantReady: true });
    mockUseFilter.mockReturnValue({ selectedDataset: null, datasets: [] });
    mockGetPodUserId.mockResolvedValue("user-1");
    mockListSearchSessions.mockResolvedValue({ sessions: [SESSION], complete: true });
    mockGetSessionDetail.mockResolvedValue(
      detailWith([{ question: QUESTION, answer: "Three entities.", time: "2026-09-01T10:01:00" }]),
    );
    mockRecallKnowledge.mockResolvedValue([{ text: "new answer" }]);
  });

  afterEach(() => {
    jest.clearAllMocks();
    clearHistoryCache();
    clearSearchState();
    sessionStorage.clear();
    localStorage.clear();
  });

  it("stays on the new-chat state after past conversations finish loading", async () => {
    render(<SearchPage />);

    await within(sidebar()).findByText(QUESTION);
    await settle();

    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
    expect(screen.queryByTestId("assistant-message")).not.toBeInTheDocument();
  });

  it("opens a past conversation only once the user selects it", async () => {
    render(<SearchPage />);

    fireEvent.click(await within(sidebar()).findByText(QUESTION));

    expect(await screen.findByTestId("assistant-message")).toHaveTextContent("Three entities.");
    expect(screen.queryByText(EMPTY_STATE)).not.toBeInTheDocument();
  });

  // Pins the merge order at the call site: reversing the arguments serves the
  // stale backend copy over the one the user just added to, and the follow-up
  // silently disappears from the chat.
  it("keeps a follow-up sent into a restored conversation", async () => {
    render(<SearchPage />);

    fireEvent.click(await within(sidebar()).findByText(QUESTION));
    fireEvent.change(screen.getByPlaceholderText("Ask a question about your data..."), {
      target: { value: "and how many edges?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "" }).closest("button") as HTMLElement);

    await waitFor(() =>
      expect(screen.getAllByTestId("assistant-message").map((n) => n.textContent)).toEqual([
        "Three entities.",
        "new answer",
      ]),
    );
  });

  it("leaves a session with no recorded question out of the history sidebar", async () => {
    mockGetSessionDetail.mockResolvedValue(
      detailWith([{ question: "", answer: null, time: "2026-09-01T10:01:00" }]),
    );

    render(<SearchPage />);

    expect(await screen.findByText("No conversations yet")).toBeInTheDocument();
  });

  it("titles a truncated conversation with the backend label", async () => {
    mockGetSessionDetail.mockResolvedValue(
      detailWith(
        [{ question: "turn twenty-one", answer: "...", time: "2026-09-01T10:21:00" }],
        "the actual first question",
      ),
    );

    render(<SearchPage />);

    expect(await within(sidebar()).findByText("the actual first question")).toBeInTheDocument();
  });

  it("shows no history when there are no search sessions", async () => {
    mockListSearchSessions.mockResolvedValue({ sessions: [], complete: true });

    render(<SearchPage />);

    expect(await screen.findByText("No conversations yet")).toBeInTheDocument();
    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
  });

  it("shows a loading placeholder instead of an empty sidebar while history loads", () => {
    mockListSearchSessions.mockReturnValue(new Promise(() => {}));

    render(<SearchPage />);

    const placeholder = screen.getByRole("status", LOADING_HISTORY);
    expect(placeholder.querySelectorAll("span[style*='animation']")).toHaveLength(4);
    expect(screen.queryByText("No conversations yet")).not.toBeInTheDocument();
  });

  it("replaces the placeholder with the conversations once they arrive", async () => {
    render(<SearchPage />);

    await within(sidebar()).findByText(QUESTION);

    expect(screen.queryByRole("status", LOADING_HISTORY)).not.toBeInTheDocument();
  });

  it("retires the placeholder when a session resolves to nothing listable", async () => {
    mockGetSessionDetail.mockResolvedValue(detailWith([]));

    render(<SearchPage />);

    expect(await screen.findByText("No conversations yet")).toBeInTheDocument();
    expect(screen.queryByRole("status", LOADING_HISTORY)).not.toBeInTheDocument();
  });

  // getSessionDetail swallows its own failures and resolves null; it does not
  // reject. This is the real unreadable-session path — a dropped row makes the
  // sidebar shorter than the user's real history, so it must not read as "none"
  // and must not be cached.
  it("retires the placeholder when a session detail comes back unreadable", async () => {
    mockGetSessionDetail.mockResolvedValue(null);

    render(<SearchPage />);

    expect(await screen.findByText("Couldn't load past conversations")).toBeInTheDocument();
    expect(screen.queryByRole("status", LOADING_HISTORY)).not.toBeInTheDocument();
    expect(screen.queryByText("No conversations yet")).not.toBeInTheDocument();
    expect(readHistoryCache("instance-1")).toBeNull();
  });

  it("says history could not be read rather than claiming there is none", async () => {
    mockListSearchSessions.mockResolvedValue({ sessions: [], complete: false });

    render(<SearchPage />);

    expect(await screen.findByText("Couldn't load past conversations")).toBeInTheDocument();
    expect(screen.queryByText("No conversations yet")).not.toBeInTheDocument();
  });

  // Fail closed: an unresolvable identity means a teammate's conversations
  // cannot be told from this user's, so none are shown.
  it("shows no history rather than a teammate's when the user id is unknown", async () => {
    mockGetPodUserId.mockResolvedValue(null);

    render(<SearchPage />);

    expect(await screen.findByText("Couldn't load past conversations")).toBeInTheDocument();
    expect(mockListSearchSessions).not.toHaveBeenCalled();
  });

  it("scopes the history request to the signed-in user", async () => {
    render(<SearchPage />);

    await waitFor(() => expect(mockListSearchSessions).toHaveBeenCalled());
    expect(mockListSearchSessions).toHaveBeenCalledWith(
      expect.anything(),
      20,
      "user-1",
      expect.any(AbortSignal),
    );
  });

  it("stops the scan when the page is left while it is still running", async () => {
    const { unmount } = render(<SearchPage />);
    await waitFor(() => expect(mockListSearchSessions).toHaveBeenCalled());
    const signal: AbortSignal = mockListSearchSessions.mock.calls[0][3];
    expect(signal.aborted).toBe(false);

    unmount();

    expect(signal.aborted).toBe(true);
  });

  // First login: the pod is provisioned but not yet confirmed reachable.
  // Asking anyway returned an empty history that the sidebar reported as
  // "no conversations", correct only after a refresh.
  it("waits for the pod to be ready before asking for history", async () => {
    mockUseTenant.mockReturnValue({ hasAccess: true, tenantReady: false });

    render(<SearchPage />);
    await settle();

    expect(mockListSearchSessions).not.toHaveBeenCalled();
    expect(screen.queryByText("No conversations yet")).not.toBeInTheDocument();
    expect(screen.getByRole("status", LOADING_HISTORY)).toBeInTheDocument();
  });

  it("loads history once the pod becomes ready, without a refresh", async () => {
    mockUseTenant.mockReturnValue({ hasAccess: true, tenantReady: false });
    const view = render(<SearchPage />);
    expect(mockListSearchSessions).not.toHaveBeenCalled();

    mockUseTenant.mockReturnValue({ hasAccess: true, tenantReady: true });
    view.rerender(<SearchPage />);

    expect(await within(sidebar()).findByText(QUESTION)).toBeInTheDocument();
  });

  it("serves a revisit from cache instead of scanning the pod again", async () => {
    const first = render(<SearchPage />);
    await within(sidebar()).findByText(QUESTION);
    first.unmount();
    jest.clearAllMocks();

    render(<SearchPage />);

    expect(await within(sidebar()).findByText(QUESTION)).toBeInTheDocument();
    expect(mockListSearchSessions).not.toHaveBeenCalled();
    expect(mockGetSessionDetail).not.toHaveBeenCalled();
  });

  it("carries a conversation from this visit into the next one", async () => {
    const first = render(<SearchPage />);
    await within(sidebar()).findByText(QUESTION);
    fireEvent.change(screen.getByPlaceholderText("Ask a question about your data..."), {
      target: { value: "a brand new question" },
    });
    fireEvent.click(screen.getByRole("button", { name: "" }).closest("button") as HTMLElement);
    await screen.findByTestId("assistant-message");
    await settle();
    first.unmount();
    jest.clearAllMocks();

    render(<SearchPage />);

    expect(await within(sidebar()).findByText("a brand new question")).toBeInTheDocument();
    expect(mockListSearchSessions).not.toHaveBeenCalled();
  });

  // The pod drops a session's Q&A after its 7-day TTL but keeps the label, so a
  // real conversation reaches the cache with a title and no messages. Filtering
  // the merged list on message count dropped it, and because a cache read skips
  // the scan, it then stayed missing from the sidebar for the whole TTL.
  it("carries a conversation whose turns the pod has aged out into the next visit", async () => {
    const AGED = { ...SESSION, session_id: "search-ui-1700000000001" };
    mockListSearchSessions.mockResolvedValue({ sessions: [SESSION, AGED], complete: true });
    mockGetSessionDetail.mockImplementation((_instance: unknown, sessionId: string) =>
      Promise.resolve(
        sessionId === AGED.session_id
          ? detailWith([], "An aged-out question")
          : detailWith([{ question: QUESTION, answer: "Three entities.", time: "2026-09-01T10:01:00" }]),
      ),
    );
    const first = render(<SearchPage />);
    await within(sidebar()).findByText("An aged-out question");
    await settle();
    first.unmount();
    jest.clearAllMocks();

    render(<SearchPage />);

    expect(await within(sidebar()).findByText("An aged-out question")).toBeInTheDocument();
    expect(mockListSearchSessions).not.toHaveBeenCalled();
  });

  // The counterpart the filter above exists for, now pinned at the call site:
  // an unused "+" row must stay out of the cache, or it returns on the next
  // mount as a real history entry and its presence suppresses the scan.
  it("does not carry an unused new conversation into the next visit", async () => {
    const first = render(<SearchPage />);
    await within(sidebar()).findByText(QUESTION);
    fireEvent.click(screen.getByTitle("New conversation"));
    await settle();
    first.unmount();

    render(<SearchPage />);
    await settle();

    expect(within(sidebar()).queryByText("New conversation")).not.toBeInTheDocument();
  });

  it("does not cache history that could not be read in full", async () => {
    mockListSearchSessions.mockResolvedValue({ sessions: [], complete: false });
    const first = render(<SearchPage />);
    await screen.findByText("Couldn't load past conversations");
    first.unmount();
    mockListSearchSessions.mockResolvedValue({ sessions: [SESSION], complete: true });

    render(<SearchPage />);

    expect(await within(sidebar()).findByText(QUESTION)).toBeInTheDocument();
  });

  it("stops the placeholder and keeps the page usable when history loading throws", async () => {
    mockListSearchSessions.mockRejectedValue(new Error("malformed page"));

    render(<SearchPage />);

    expect(await screen.findByText("Couldn't load past conversations")).toBeInTheDocument();
    expect(screen.queryByRole("status", LOADING_HISTORY)).not.toBeInTheDocument();
    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask a question about your data...")).toBeEnabled();
  });

  // ── CLO-846: the open conversation and an answer in flight outlive the page ──

  function send(text: string): void {
    fireEvent.change(screen.getByPlaceholderText("Ask a question about your data..."), {
      target: { value: text },
    });
    fireEvent.click(screen.getByRole("button", { name: "" }).closest("button") as HTMLElement);
  }

  function recallThatStays(
    onStart?: (handlers: RecallStreamHandlers) => void,
  ): { finish: (results: unknown[]) => void } {
    const control: { finish: (results: unknown[]) => void } = { finish: () => undefined };
    mockRecallKnowledge.mockImplementation(
      (_instance: unknown, _req: unknown, handlers: RecallStreamHandlers) => {
        onStart?.(handlers);
        return new Promise<unknown[]>((resolve) => {
          control.finish = (results) => resolve(results);
        });
      },
    );
    return control;
  }

  it("reopens the conversation the user had open after a remount", async () => {
    const first = render(<SearchPage />);
    fireEvent.click(await within(sidebar()).findByText(QUESTION));
    await screen.findByTestId("assistant-message");
    await settle();
    first.unmount();

    render(<SearchPage />);

    expect(await screen.findByTestId("assistant-message")).toHaveTextContent("Three entities.");
    expect(screen.queryByText(EMPTY_STATE)).not.toBeInTheDocument();
  });

  it("continues the same session when a follow-up is sent after a remount", async () => {
    const first = render(<SearchPage />);
    fireEvent.click(await within(sidebar()).findByText(QUESTION));
    await settle();
    first.unmount();

    render(<SearchPage />);
    await screen.findByTestId("assistant-message");
    send("and how many edges?");

    await waitFor(() =>
      expect(mockRecallKnowledge).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({ query: "and how many edges?", sessionId: SESSION.session_id }),
        expect.anything(),
      ),
    );
  });

  // A reload: the id is remembered for the tab, but the store and the history
  // cache are gone, and the user asks before the old turns have arrived.
  it("keeps a follow-up sent under a remembered session before its history has loaded", async () => {
    setSearchActiveConversation("instance-1", SESSION.session_id);
    let deliverHistory: (detail: unknown) => void = () => undefined;
    mockGetSessionDetail.mockReturnValue(new Promise((resolve) => { deliverHistory = resolve; }));
    render(<SearchPage />);

    send("and how many edges?");
    expect(await screen.findByTestId("assistant-message")).toHaveTextContent("new answer");
    await act(async () => {
      deliverHistory(detailWith([{ question: QUESTION, answer: "Three entities.", time: "2026-09-01T10:01:00" }]));
    });

    await waitFor(() =>
      expect(screen.getAllByTestId("assistant-message").map((n) => n.textContent)).toEqual([
        "Three entities.",
        "new answer",
      ]),
    );
    expect(mockRecallKnowledge).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ sessionId: SESSION.session_id }),
      expect.anything(),
    );
  });

  it("forgets a remembered conversation the pod no longer has", async () => {
    setSearchActiveConversation("instance-1", "search-ui-gone");
    render(<SearchPage />);

    await within(sidebar()).findByText(QUESTION);
    await settle();
    send("a fresh question");

    await waitFor(() =>
      expect(mockRecallKnowledge).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({ sessionId: expect.stringMatching(/^search-ui-\d+$/) }),
        expect.anything(),
      ),
    );
    const sent = (mockRecallKnowledge.mock.calls[0][1] as { sessionId: string }).sessionId;
    expect(sent).not.toBe("search-ui-gone");
  });

  it("keeps an answer that arrived while the page was away", async () => {
    const recall = recallThatStays();
    const first = render(<SearchPage />);
    await within(sidebar()).findByText(QUESTION);
    await settle();
    send("a slow question");
    await screen.findByTestId("assistant-message-loading");
    first.unmount();
    await act(async () => {
      recall.finish([{ text: "late answer" }]);
    });

    render(<SearchPage />);

    expect(await screen.findByTestId("assistant-message")).toHaveTextContent("late answer");
    expect(within(sidebar()).getByText("a slow question")).toBeInTheDocument();
  });

  it("shows an answer still streaming when the page is reopened", async () => {
    let live: RecallStreamHandlers = {};
    recallThatStays((handlers) => { live = handlers; });
    const first = render(<SearchPage />);
    await within(sidebar()).findByText(QUESTION);
    await settle();
    send("a slow question");
    await screen.findByTestId("assistant-message-loading");
    first.unmount();
    act(() => live.onDelta?.("arriving"));

    render(<SearchPage />);

    expect(await screen.findByTestId("assistant-message-loading")).toHaveTextContent("arriving");
  });

  it("starts a fresh chat on + without listing a placeholder conversation", async () => {
    render(<SearchPage />);
    fireEvent.click(await within(sidebar()).findByText(QUESTION));
    await screen.findByTestId("assistant-message");

    fireEvent.click(screen.getByTitle("New conversation"));

    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
    expect(within(sidebar()).queryByText("New conversation")).not.toBeInTheDocument();
  });

  // ── CLO-845: a streamed answer is painted as it arrives ──

  it("streams the answer into the bubble before the payload lands", async () => {
    const recall = recallThatStays((handlers) => {
      handlers.onStage?.("retrieving");
      handlers.onStage?.("generating");
      handlers.onDelta?.("Three ");
      handlers.onDelta?.("entities");
    });
    render(<SearchPage />);

    send("What is in my graph?");

    expect(await screen.findByTestId("assistant-message-loading")).toHaveTextContent("Three entities");
    await act(async () => {
      recall.finish([{ text: "Three entities. [1]" }]);
    });
    expect(await screen.findByTestId("assistant-message")).toHaveTextContent("Three entities. [1]");
  });

  it("says what the pod is doing while nothing has been written yet", async () => {
    recallThatStays((handlers) => handlers.onStage?.("retrieving"));
    render(<SearchPage />);

    send("What is in my graph?");

    expect(await screen.findByText("Searching your data…")).toBeInTheDocument();
  });

  it("discards the preview when the pod restarts the answer", async () => {
    recallThatStays((handlers) => {
      handlers.onDelta?.("a wrong start");
      handlers.onReset?.();
      handlers.onDelta?.("right");
    });
    render(<SearchPage />);

    send("What is in my graph?");

    const bubble = await screen.findByTestId("assistant-message-loading");
    expect(bubble).toHaveTextContent("right");
    expect(bubble).not.toHaveTextContent("wrong");
  });
  describe("search type", () => {
    const TEXT_PLACEHOLDER = "Ask a question about your data...";
    const CODE_PLACEHOLDER = "Search a function, class or file name...";
    const FACTS = {
      operation: "query_facts",
      total: 1,
      facts: [{ id: "n1", kind: "symbol", symbol_kind: "function", name: "retry_with_backoff", file: "src/net.py", line: 42 }],
    };

    beforeEach(() => {
      mockSearchCode.mockResolvedValue([FACTS]);
    });

    function radio(label: string): HTMLElement {
      return screen.getByRole("radio", { name: label });
    }

    function pick(label: string): void {
      fireEvent.click(radio(label));
    }

    function typeInto(placeholder: string, text: string): void {
      fireEvent.change(screen.getByPlaceholderText(placeholder), { target: { value: text } });
    }

    // The box is found by its placeholder, and that is what the type changes.
    function sendCode(text: string): void {
      typeInto(CODE_PLACEHOLDER, text);
      fireEvent.click(screen.getByRole("button", { name: "" }).closest("button") as HTMLElement);
    }

    function lastCodeSearch(): { query: string; codeQuery: unknown; datasetIds: unknown } {
      const calls = mockSearchCode.mock.calls;
      return calls[calls.length - 1][1] as { query: string; codeQuery: unknown; datasetIds: unknown };
    }

    function lastRecall(): { sessionId: string } {
      const calls = mockRecallKnowledge.mock.calls;
      return calls[calls.length - 1][1] as { sessionId: string };
    }

    // The request a search made before the selector existed, field for field.
    // Text is the default, so an untouched page must still send exactly this.
    it("asks the question through recall by default, exactly as before", async () => {
      render(<SearchPage />);
      await settle();

      send(QUESTION);

      await waitFor(() => expect(mockRecallKnowledge).toHaveBeenCalled());
      expect(mockRecallKnowledge.mock.calls[0][1]).toEqual({
        query: QUESTION,
        scope: "graph",
        sessionId: expect.stringMatching(/^search-ui-/),
        datasetIds: [],
      });
      expect(mockSearchCode).not.toHaveBeenCalled();
    });

    // Code is answered by the code graph, and only /v1/search takes the
    // operation it needs — so it must not go through recall at all.
    it("looks the name up in the code graph once Code is picked, and leaves recall alone", async () => {
      render(<SearchPage />);
      await settle();

      pick("Code");
      sendCode("where is the retry handled");

      await waitFor(() => expect(mockSearchCode).toHaveBeenCalled());
      expect(lastCodeSearch()).toEqual({
        query: "where is the retry handled",
        codeQuery: { operation: "query_facts", name: "retry", limit: 50 },
        datasetIds: [],
      });
      expect(mockRecallKnowledge).not.toHaveBeenCalled();
    });

    // The sentence that was reported: sent bare, the pod resolved it as a
    // symbol name and answered "could not resolve explore seed name".
    it("asks for the architecture by operation rather than sending the sentence as a name", async () => {
      render(<SearchPage />);
      await settle();

      pick("Code");
      sendCode("get the architecture");

      await waitFor(() => expect(mockSearchCode).toHaveBeenCalled());
      expect(lastCodeSearch().codeQuery).toEqual({ operation: "architecture" });
    });

    // The question as it was asked: the path used to be looked up as a name,
    // which matched only the file's own node.
    it("asks what a file defines when the question names a file", async () => {
      render(<SearchPage />);
      await settle();

      pick("Code");
      sendCode("What is in this file: src/components/billing/AddPaymentMethodModal.tsx");

      await waitFor(() => expect(mockSearchCode).toHaveBeenCalled());
      expect(lastCodeSearch().codeQuery).toEqual({
        operation: "query_facts",
        file: "src/components/billing/AddPaymentMethodModal.tsx",
        limit: 200,
      });
    });

    it("shows what the code graph found", async () => {
      render(<SearchPage />);
      await settle();

      pick("Code");
      sendCode("retry");

      // jest.setup stubs react-markdown, so this is the markdown as written;
      // what it is made of is pinned down in codeSearch.test.ts.
      const answer = await screen.findByTestId("assistant-message");
      expect(answer).toHaveTextContent("1 match for `retry`");
      expect(answer).toHaveTextContent("`retry_with_backoff` — symbol · function — `src/net.py:42`");
    });

    it("shows the pod's own message when the lookup fails", async () => {
      mockSearchCode.mockRejectedValue(new Error("Dataset has no code graph."));
      render(<SearchPage />);
      await settle();

      pick("Code");
      sendCode("retry");

      expect(await screen.findByTestId("assistant-message")).toHaveTextContent("Dataset has no code graph.");
    });

    it("reports which type each search ran on", async () => {
      render(<SearchPage />);
      await settle();

      pick("Code");
      sendCode("retry");

      await waitFor(() => expect(mockSearchCode).toHaveBeenCalled());
      expect(trackEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          eventName: "search_executed",
          additionalProperties: expect.objectContaining({ search_type: "code" }),
        }),
      );
    });

    // The same box takes a question under one type and a name under the
    // other; the placeholder is what says which before anything is sent.
    it("says what to type for the type that is picked", async () => {
      render(<SearchPage />);
      await settle();
      expect(screen.getByPlaceholderText(TEXT_PLACEHOLDER)).toBeInTheDocument();

      pick("Code");

      expect(screen.getByPlaceholderText(CODE_PLACEHOLDER)).toBeInTheDocument();
    });

    // A question chip sent under Code would look a sentence up as a name.
    it("offers starting points the picked type can answer", async () => {
      render(<SearchPage />);
      await settle();
      expect(screen.getByRole("button", { name: "What are the main entities?" })).toBeInTheDocument();

      pick("Code");

      expect(screen.queryByRole("button", { name: "What are the main entities?" })).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: "Show the architecture" }));
      await waitFor(() => expect(mockSearchCode).toHaveBeenCalled());
      expect(lastCodeSearch().codeQuery).toEqual({ operation: "architecture" });
    });

    describe("the GitHub hint", () => {
      // Elsewhere it points at a type with nothing to search.
      it("stays out of a workspace with no connected repository", async () => {
        mockUseFilter.mockReturnValue({ selectedDataset: null, datasets: [{ id: "ds-1", name: "main_dataset" }] });
        render(<SearchPage />);
        await settle();

        expect(screen.queryByTestId("code-search-hint")).not.toBeInTheDocument();
      });

      // Worded as a lookup, because "Use Code" alone invites exactly the
      // question phrasing the code graph cannot answer.
      it("says Code looks things up by name once a repository is connected", async () => {
        mockUseFilter.mockReturnValue({ selectedDataset: null, datasets: [{ id: "ds-1", name: "github_123456" }] });
        render(<SearchPage />);
        await settle();

        expect(screen.getByTestId("code-search-hint")).toHaveTextContent(
          "For GitHub repositories, use Code — it looks up functions, classes and files by name.",
        );
      });
    });

    describe("as a radio group", () => {
      it("announces one choice between two options, exactly one of them on", async () => {
        render(<SearchPage />);
        await settle();

        const group = screen.getByRole("radiogroup", { name: "Search type" });
        const options = within(group).getAllByRole("radio");
        expect(options.map((o) => o.textContent)).toEqual(["Text", "Code"]);
        expect(options.filter((o) => o.getAttribute("aria-checked") === "true")).toHaveLength(1);
      });

      // One tab stop for the group; the arrows move within it.
      it("keeps only the selected option in the tab order", async () => {
        render(<SearchPage />);
        await settle();

        expect(radio("Text")).toHaveAttribute("tabindex", "0");
        expect(radio("Code")).toHaveAttribute("tabindex", "-1");
      });

      it.each([
        ["ArrowRight", "Code"],
        ["ArrowDown", "Code"],
        ["ArrowLeft", "Code"],
        ["ArrowUp", "Code"],
      ])("moves the selection and the focus with %s, wrapping at the ends", async (key, expected) => {
        render(<SearchPage />);
        await settle();

        radio("Text").focus();
        fireEvent.keyDown(radio("Text"), { key });

        expect(radio(expected)).toHaveAttribute("aria-checked", "true");
        expect(radio(expected)).toHaveFocus();
      });

      it("ignores keys that are not arrows", async () => {
        render(<SearchPage />);
        await settle();

        fireEvent.keyDown(radio("Text"), { key: "Enter" });

        expect(radio("Text")).toHaveAttribute("aria-checked", "true");
      });
    });

    // The pod records no Code turn, so the browser does. A reload here is the
    // in-memory store and history cache going away while localStorage stays.
    describe("after a reload", () => {
      async function askCodeThenReload(unmount: () => void): Promise<void> {
        pick("Code");
        sendCode("retry");
        await screen.findByTestId("assistant-message");
        await settle();
        unmount();
        clearSearchState();
        clearHistoryCache();
      }

      it("lists the Code conversations asked before it, beside the pod's own", async () => {
        const first = render(<SearchPage />);
        await settle();
        await askCodeThenReload(first.unmount);

        render(<SearchPage />);

        expect(await within(sidebar()).findByText("retry")).toBeInTheDocument();
        expect(await within(sidebar()).findByText(QUESTION)).toBeInTheDocument();
      });

      it("reopens one as Code, with its answer", async () => {
        const first = render(<SearchPage />);
        await settle();
        await askCodeThenReload(first.unmount);
        render(<SearchPage />);

        fireEvent.click(await within(sidebar()).findByText("retry"));

        expect(radio("Code")).toHaveAttribute("aria-checked", "true");
        expect(await screen.findByTestId("assistant-message")).toHaveTextContent("retry_with_backoff");
      });

      it("keeps asking the code graph when a restored Code conversation is continued", async () => {
        const first = render(<SearchPage />);
        await settle();
        await askCodeThenReload(first.unmount);
        render(<SearchPage />);
        fireEvent.click(await within(sidebar()).findByText("retry"));

        sendCode("CodeRetriever");

        await waitFor(() => expect(mockSearchCode).toHaveBeenCalledTimes(2));
        expect(mockRecallKnowledge).not.toHaveBeenCalled();
      });
    });

    describe("per conversation", () => {
      // The report: with the type as page state, reopening a Text conversation
      // while Code was remembered sent the follow-up to the code graph, which
      // found nothing and left an unrecorded turn in a pod session.
      it("reopens a conversation from history as Text, whatever was picked last", async () => {
        setSearchType("instance-1", "code");
        render(<SearchPage />);
        await settle();
        expect(radio("Code")).toHaveAttribute("aria-checked", "true");

        fireEvent.click(await within(sidebar()).findByText(QUESTION));

        expect(radio("Text")).toHaveAttribute("aria-checked", "true");
        send("and how many edges?");
        await waitFor(() => expect(mockRecallKnowledge).toHaveBeenCalled());
        expect(lastRecall().sessionId).toBe(SESSION.session_id);
        expect(mockSearchCode).not.toHaveBeenCalled();
      });

      it("reopens a Code conversation as Code and keeps asking the code graph", async () => {
        render(<SearchPage />);
        await settle();
        pick("Code");
        sendCode("retry");
        await screen.findByTestId("assistant-message");

        fireEvent.click(await within(sidebar()).findByText(QUESTION));
        expect(radio("Text")).toHaveAttribute("aria-checked", "true");
        fireEvent.click(within(sidebar()).getByText("retry"));

        expect(radio("Code")).toHaveAttribute("aria-checked", "true");
        sendCode("CodeRetriever");
        await waitFor(() => expect(mockSearchCode).toHaveBeenCalledTimes(2));
        expect(mockRecallKnowledge).not.toHaveBeenCalled();
      });

      // Mixing would put an unrecorded Code turn inside a pod session, so the
      // other type gets a conversation of its own.
      it("starts a new conversation when the other type is picked mid-conversation, keeping what was typed", async () => {
        render(<SearchPage />);
        await settle();
        pick("Code");
        sendCode("retry");
        await screen.findByTestId("assistant-message");
        typeInto(CODE_PLACEHOLDER, "how many edges are there");

        pick("Text");

        expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
        expect(screen.getByPlaceholderText(TEXT_PLACEHOLDER)).toHaveValue("how many edges are there");
        fireEvent.click(screen.getByRole("button", { name: "" }).closest("button") as HTMLElement);
        await waitFor(() => expect(mockRecallKnowledge).toHaveBeenCalled());
        expect(within(sidebar()).getByText("retry")).toBeInTheDocument();
      });

      it("stays in the conversation when the type already open is picked again", async () => {
        render(<SearchPage />);
        await settle();
        pick("Code");
        sendCode("retry");
        await screen.findByTestId("assistant-message");

        pick("Code");

        expect(screen.queryByText(EMPTY_STATE)).not.toBeInTheDocument();
      });
    });

    describe("remembering the choice", () => {
      // The page unmounts on every navigation away from Search, so without this
      // the choice would be re-made on every visit.
      it("remembers the type for the pod across a remount", async () => {
        const { unmount } = render(<SearchPage />);
        await settle();
        pick("Code");
        unmount();

        render(<SearchPage />);
        await settle();

        expect(radio("Code")).toHaveAttribute("aria-checked", "true");
        sendCode("retry");
        await waitFor(() => expect(mockSearchCode).toHaveBeenCalled());
        expect(mockRecallKnowledge).not.toHaveBeenCalled();
      });

      // Asserted in both directions: a restore that ignored the pod would
      // carry Code into the second workspace, and no restore at all would
      // miss it in the first.
      it("keeps each workspace's choice to that workspace", async () => {
        setSearchType("instance-1", "code");
        const { rerender } = render(<SearchPage />);
        await settle();
        expect(radio("Code")).toHaveAttribute("aria-checked", "true");

        mockUseCogniInstance.mockReturnValue({
          cogniInstance: { fetch: jest.fn(), instanceId: "instance-2" },
          isInitializing: false,
        });
        rerender(<SearchPage />);
        await settle();

        expect(radio("Text")).toHaveAttribute("aria-checked", "true");
      });

      // A value from a past build (or a hand edit) must not decide which
      // retriever a search runs on — and must not stop the next choice sticking.
      it("falls back to Text on a value this build does not know, then remembers the next choice", async () => {
        setSearchType("instance-1", "HYBRID_COMPLETION");
        const first = render(<SearchPage />);
        await settle();
        expect(radio("Text")).toHaveAttribute("aria-checked", "true");

        pick("Code");
        first.unmount();
        render(<SearchPage />);
        await settle();

        expect(radio("Code")).toHaveAttribute("aria-checked", "true");
      });
    });
  });
});
