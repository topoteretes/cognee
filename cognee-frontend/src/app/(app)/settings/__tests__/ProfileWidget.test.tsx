import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MantineProvider } from "@mantine/core";

const mockUseUser = jest.fn();
jest.mock("@/modules/users/UserContext", () => ({
  useUser: () => mockUseUser(),
}));

jest.mock("@/modules/users/useCurrentUser", () => ({
  useCurrentUser: () => ({
    data: { id: "auth0|1", name: "From Auth0", email: "ada@example.com", picture: "" },
  }),
}));

const mockNotify = jest.fn();
jest.mock("@mantine/notifications", () => ({
  notifications: { show: (...args: unknown[]) => mockNotify(...args) },
}));

jest.mock("next/image", () => ({
  __esModule: true,
  // eslint-disable-next-line @next/next/no-img-element
  default: (props: { alt: string }) => <img alt={props.alt} />,
}));

import ProfileWidget from "../elements/ProfileWidget";
import { HttpError } from "@/services/http/errors";

type Profile = { firstName: string | null; lastName: string | null } | null;

function setUser(profile: Profile, update = jest.fn(), isLoading = false) {
  mockUseUser.mockReturnValue({
    userMe: profile === null ? null : { profile },
    isLoading,
    updateProfileName: update,
  });
  return update;
}

function renderWidget() {
  return render(
    <MantineProvider>
      <ProfileWidget />
    </MantineProvider>,
  );
}

const first = () => screen.getByLabelText("First name") as HTMLInputElement;
const last = () => screen.getByLabelText("Last name") as HTMLInputElement;
const save = () => screen.getByRole("button", { name: "Save" }) as HTMLButtonElement;

function type(input: HTMLInputElement, value: string) {
  fireEvent.change(input, { target: { value } });
}

// What the route answers when the backend refused a name: an HttpError with
// this body, as http.patchJson throws it.
const refusal = (fields?: string[]) =>
  new HttpError(422, "Unprocessable Entity", "name_refused", { error: "name_refused", fields });

const REFUSED = /characters we can't accept/;

describe("ProfileWidget", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe("what it shows", () => {
    it("shows the stored name in the form and in the header", () => {
      setUser({ firstName: "Ada", lastName: "Lovelace" });

      renderWidget();

      expect(first().value).toBe("Ada");
      expect(last().value).toBe("Lovelace");
      expect(screen.getByText("Ada Lovelace")).toBeTruthy();
    });

    it("falls back to the sign-in provider's name in the header for an account with no name set", () => {
      setUser({ firstName: null, lastName: null });

      renderWidget();

      expect(first().value).toBe("");
      expect(screen.getByText("From Auth0")).toBeTruthy();
    });

    it("is an editable empty state, not an error, for an account with no name", () => {
      setUser({ firstName: null, lastName: null });

      renderWidget();

      expect(first().disabled).toBe(false);
      expect(first().placeholder).toBe("First name");
      expect(screen.queryByText(/not available/)).toBeNull();
    });

    it("never shows the email as editable", () => {
      setUser({ firstName: "Ada", lastName: "Lovelace" });

      renderWidget();

      expect((screen.getByLabelText("Email") as HTMLInputElement).disabled).toBe(true);
    });
  });

  describe("while the profile is not there to edit", () => {
    it("is disabled and says nothing while /me is still loading", () => {
      // Not "unavailable": it is not known yet, and telling every person that
      // editing is broken for the length of a normal page load would be false.
      setUser(null, jest.fn(), true);

      renderWidget();

      expect(first().disabled).toBe(true);
      expect(save().disabled).toBe(true);
      expect(screen.queryByText(/not available/)).toBeNull();
    });

    it("is disabled and says so once /me has answered without a profile", () => {
      // A backend that predates the field, or a failed /me. An enabled input
      // would write to an endpoint that may not be there.
      mockUseUser.mockReturnValue({ userMe: { profile: null }, isLoading: false, updateProfileName: jest.fn() });

      renderWidget();

      expect(first().disabled).toBe(true);
      expect(last().disabled).toBe(true);
      expect(save().disabled).toBe(true);
      expect(screen.getByText(/not available right now/)).toBeTruthy();
    });

    it("says so when /me failed, not only when it answered without a profile", () => {
      setUser(null, jest.fn(), false);

      renderWidget();

      expect(screen.getByText(/not available right now/)).toBeTruthy();
    });
  });

  describe("when Save is available", () => {
    it("stays disabled until something changes", () => {
      setUser({ firstName: "Ada", lastName: "Lovelace" });

      renderWidget();
      expect(save().disabled).toBe(true);

      type(first(), "Augusta");
      expect(save().disabled).toBe(false);
    });

    it("does not count a change in whitespace alone as a change", () => {
      setUser({ firstName: "Ada", lastName: "Lovelace" });

      renderWidget();
      type(first(), "  Ada  ");

      expect(save().disabled).toBe(true);
    });

    it("does not count recomposing the same name as a change", () => {
      setUser({ firstName: "Zoë", lastName: null });

      renderWidget();
      type(first(), "Zoë");

      expect(save().disabled).toBe(true);
    });

    it("goes back to disabled when the edit is undone", () => {
      setUser({ firstName: "Ada", lastName: null });

      renderWidget();
      type(first(), "Augusta");
      type(first(), "Ada");

      expect(save().disabled).toBe(true);
    });

    it("refuses an over-long name before sending it", () => {
      const update = setUser({ firstName: null, lastName: null });

      renderWidget();
      type(first(), "a".repeat(101));

      expect(screen.getByText(/at most 100 characters/)).toBeTruthy();
      expect(save().disabled).toBe(true);
      expect(update).not.toHaveBeenCalled();
    });

    it("measures the limit on the composed name, so a long decomposed one is not refused for its parts", () => {
      setUser({ firstName: null, lastName: null });

      renderWidget();
      type(first(), "é".repeat(51));

      expect(screen.queryByText(/at most 100 characters/)).toBeNull();
      expect(save().disabled).toBe(false);
    });

    it("tells which of the two names is too long", () => {
      setUser({ firstName: null, lastName: null });

      renderWidget();
      type(last(), "a".repeat(101));

      expect(first().getAttribute("aria-invalid")).not.toBe("true");
      expect(last().getAttribute("aria-invalid")).toBe("true");
    });
  });

  describe("saving", () => {
    it("sends only the name that was changed, normalized", async () => {
      const update = setUser({ firstName: "Ada", lastName: "Lovelace" }, jest.fn().mockResolvedValue(undefined));

      renderWidget();
      type(first(), "  Augusta   Ada ");
      fireEvent.click(save());

      await waitFor(() => expect(update).toHaveBeenCalledWith({ firstName: "Augusta Ada" }));
      await waitFor(() =>
        expect(mockNotify).toHaveBeenCalledWith({
          title: "Profile updated",
          message: "Your name was saved.",
          color: "green",
          autoClose: 4000,
        }),
      );
    });

    it("does not put back a name that was changed elsewhere while this page was open", async () => {
      // Another tab or device changed the last name. This form only touched the
      // first, so saving it must not send the last name it loaded with.
      const update = setUser({ firstName: "Ada", lastName: "Lovelace" }, jest.fn().mockResolvedValue(undefined));
      const { rerender } = renderWidget();
      type(first(), "Augusta");

      setUser({ firstName: "Ada", lastName: "Byron" }, update);
      rerender(
        <MantineProvider>
          <ProfileWidget />
        </MantineProvider>,
      );
      fireEvent.click(save());

      await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
      expect(update.mock.calls[0][0]).toEqual({ firstName: "Augusta" });
      expect("lastName" in update.mock.calls[0][0]).toBe(false);
    });

    it("sends an emptied name as an explicit clear", async () => {
      const update = setUser({ firstName: "Ada", lastName: "Lovelace" }, jest.fn().mockResolvedValue(undefined));

      renderWidget();
      type(last(), "");
      fireEvent.click(save());

      await waitFor(() => expect(update).toHaveBeenCalledWith({ lastName: null }));
    });

    it("sends both when both were changed", async () => {
      const update = setUser({ firstName: null, lastName: null }, jest.fn().mockResolvedValue(undefined));

      renderWidget();
      type(first(), "Ada");
      type(last(), "Lovelace");
      fireEvent.click(save());

      await waitFor(() => expect(update).toHaveBeenCalledWith({ firstName: "Ada", lastName: "Lovelace" }));
    });

    it("saves from the keyboard", async () => {
      const update = setUser({ firstName: null, lastName: null }, jest.fn().mockResolvedValue(undefined));

      renderWidget();
      type(first(), "Ada");
      fireEvent.submit(first().closest("form") as HTMLFormElement);

      await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    });

    it("does not send twice for a second press while the first is in flight", async () => {
      let finish: () => void = () => {};
      const update = setUser(
        { firstName: null, lastName: null },
        jest.fn().mockReturnValue(new Promise<void>((done) => { finish = done; })),
      );

      renderWidget();
      type(first(), "Ada");
      fireEvent.click(save());
      fireEvent.submit(first().closest("form") as HTMLFormElement);
      finish();

      await waitFor(() => expect(mockNotify).toHaveBeenCalled());
      expect(update).toHaveBeenCalledTimes(1);
    });

    it("sends nothing when nothing changed", () => {
      const update = setUser({ firstName: "Ada", lastName: null });

      renderWidget();
      fireEvent.submit(first().closest("form") as HTMLFormElement);

      expect(update).not.toHaveBeenCalled();
    });
  });

  describe("when the backend refuses the name", () => {
    it("says so under the name that was refused, and not under the other", async () => {
      setUser({ firstName: "Ada", lastName: "Lovelace" }, jest.fn().mockRejectedValue(refusal(["lastName"])));

      renderWidget();
      type(last(), "Love\u0000lace");
      fireEvent.click(save());

      expect(await screen.findByText(REFUSED)).toBeTruthy();
      expect(screen.getAllByText(REFUSED)).toHaveLength(1);
      expect(last().getAttribute("aria-invalid")).toBe("true");
      expect(first().getAttribute("aria-invalid")).not.toBe("true");
    });

    it("moves focus to the name that was refused, so it is announced and can be fixed", async () => {
      setUser({ firstName: "Ada", lastName: "Lovelace" }, jest.fn().mockRejectedValue(refusal(["lastName"])));

      renderWidget();
      type(last(), "bad");
      fireEvent.click(save());
      await screen.findByText(REFUSED);

      await waitFor(() => expect(document.activeElement).toBe(last()));
    });

    it("says so under both when both were refused, and focuses the first", async () => {
      setUser({ firstName: null, lastName: null }, jest.fn().mockRejectedValue(refusal(["firstName", "lastName"])));

      renderWidget();
      type(first(), "a");
      type(last(), "b");
      fireEvent.click(save());

      await waitFor(() => expect(screen.getAllByText(REFUSED)).toHaveLength(2));
      await waitFor(() => expect(document.activeElement).toBe(first()));
    });

    it("says so under both when the backend did not say which", async () => {
      setUser({ firstName: null, lastName: null }, jest.fn().mockRejectedValue(refusal()));

      renderWidget();
      type(first(), "a");
      fireEvent.click(save());

      await waitFor(() => expect(screen.getAllByText(REFUSED)).toHaveLength(2));
    });

    it("keeps what was typed and does not show a notification that goes away", async () => {
      setUser({ firstName: null, lastName: null }, jest.fn().mockRejectedValue(refusal(["firstName"])));

      renderWidget();
      type(first(), "Ada");
      fireEvent.click(save());
      await screen.findByText(REFUSED);

      expect(first().value).toBe("Ada");
      expect(mockNotify).not.toHaveBeenCalled();
    });

    it("clears the refusal as soon as the name is edited", async () => {
      setUser({ firstName: null, lastName: null }, jest.fn().mockRejectedValue(refusal(["firstName"])));

      renderWidget();
      type(first(), "Ada");
      fireEvent.click(save());
      await screen.findByText(REFUSED);

      type(first(), "Adaa");

      expect(screen.queryByText(REFUSED)).toBeNull();
    });

    it("lets the person try again, and shows the refusal again if it is refused again", async () => {
      setUser({ firstName: null, lastName: null }, jest.fn().mockRejectedValue(refusal(["firstName"])));

      renderWidget();
      type(first(), "Ada");
      fireEvent.click(save());
      await screen.findByText(REFUSED);
      type(first(), "Adaa");
      fireEvent.click(save());

      expect(await screen.findByText(REFUSED)).toBeTruthy();
      expect(save().disabled).toBe(false);
    });
  });

  describe("when saving fails for any other reason", () => {
    it.each([
      ["a server error", new HttpError(502, "Bad Gateway", "management_api_error", { error: "management_api_error" })],
      ["a network error", new Error("down")],
      ["a 422 that is not a refused name", new HttpError(422, "Unprocessable", "x", { error: "something_else" })],
      ["a 403 for an API key", new HttpError(403, "Forbidden", "not_allowed", { error: "not_allowed" })],
    ])("notifies and keeps the form on %s", async (_label, error) => {
      setUser({ firstName: null, lastName: null }, jest.fn().mockRejectedValue(error));

      renderWidget();
      type(first(), "Ada");
      fireEvent.click(save());

      await waitFor(() => expect(mockNotify).toHaveBeenCalledWith(expect.objectContaining({ color: "red" })));
      expect(screen.queryByText(REFUSED)).toBeNull();
      expect(first().value).toBe("Ada");
      expect(save().disabled).toBe(false);
    });
  });
});
