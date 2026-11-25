import { fireEvent, render, screen } from "@testing-library/react";
import NameWorkspaceModal, { WORKSPACE_PRICE_LABEL } from "../NameWorkspaceModal";

function renderModal(overrides: Partial<React.ComponentProps<typeof NameWorkspaceModal>> = {}) {
  const props = {
    name: "Workspace 2",
    setName: jest.fn(),
    submitting: false,
    isFree: false,
    error: null,
    onSubmit: jest.fn(),
    onClose: jest.fn(),
    ...overrides,
  };
  const { container } = render(<NameWorkspaceModal {...props} />);
  // The backdrop is the component's root element; selected structurally rather
  // than via a test id so the assertion doesn't require production markup that
  // exists only for the test.
  return { ...props, overlay: container.firstChild as HTMLElement };
}

const cancelButton = () => screen.getByRole("button", { name: "Cancel" });
const submitButton = () => screen.getByRole("button", { name: /Create workspace|Continue to payment|Creating|Redirecting/ });

describe("NameWorkspaceModal — escaping a submit in flight", () => {
  // The user-visible half of CLO-747: a submit that never lands used to leave
  // the modal with no way out but a page reload, because Cancel and the overlay
  // were both gated on `submitting`.
  it("keeps Cancel enabled while a submit is in flight", () => {
    renderModal({ submitting: true });
    expect(cancelButton()).toBeEnabled();
  });

  it("closes on Cancel even while a submit is in flight", () => {
    const { onClose } = renderModal({ submitting: true });
    fireEvent.click(cancelButton());
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("closes on an overlay click even while a submit is in flight", () => {
    const { onClose, overlay } = renderModal({ submitting: true });
    fireEvent.click(overlay);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not close when the dialog itself is clicked", () => {
    const { onClose } = renderModal();
    fireEvent.click(screen.getByRole("textbox"));
    expect(onClose).not.toHaveBeenCalled();
  });

  it("still blocks a second submit while one is in flight", () => {
    renderModal({ submitting: true });
    expect(submitButton()).toBeDisabled();
  });
});

describe("NameWorkspaceModal — free vs paid", () => {
  it("promises no charge and offers to create directly when the workspace is free", () => {
    renderModal({ isFree: true });
    expect(screen.getByText(/costs nothing/)).toBeInTheDocument();
    expect(screen.queryByText(WORKSPACE_PRICE_LABEL)).not.toBeInTheDocument();
    expect(submitButton()).toHaveTextContent("Create workspace");
  });

  it("names the price and sends the user to payment when it is not free", () => {
    renderModal({ isFree: false });
    expect(screen.getByText(WORKSPACE_PRICE_LABEL)).toBeInTheDocument();
    expect(submitButton()).toHaveTextContent(`Continue to payment · ${WORKSPACE_PRICE_LABEL}`);
  });

  it("labels the in-flight state as creating rather than redirecting when free", () => {
    // A free workspace never goes to Stripe, so "Redirecting..." would be a lie.
    renderModal({ isFree: true, submitting: true });
    expect(submitButton()).toHaveTextContent("Creating...");
  });
});

describe("NameWorkspaceModal — name validation", () => {
  it("refuses to submit a name shorter than two characters", () => {
    const { onSubmit } = renderModal({ name: "a" });
    expect(submitButton()).toBeDisabled();
    fireEvent.submit(screen.getByRole("textbox").closest("form") as HTMLFormElement);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("treats a whitespace-only name as empty", () => {
    const { onSubmit } = renderModal({ name: "   " });
    fireEvent.submit(screen.getByRole("textbox").closest("form") as HTMLFormElement);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits a valid name", () => {
    const { onSubmit } = renderModal({ name: "Research" });
    fireEvent.submit(screen.getByRole("textbox").closest("form") as HTMLFormElement);
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("shows the error the caller passes down", () => {
    renderModal({ error: "Could not start checkout. Please try again." });
    expect(screen.getByText("Could not start checkout. Please try again.")).toBeInTheDocument();
  });
});
