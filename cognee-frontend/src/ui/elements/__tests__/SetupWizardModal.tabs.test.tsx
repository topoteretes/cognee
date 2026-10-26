jest.mock("next/navigation", () => ({ useRouter: () => ({ push: jest.fn() }) }));
jest.mock("../AgentConnectionStatus", () => ({ AgentConnectionStatus: () => <div>Waiting for connection</div> }));
jest.mock("../OsToggle", () => ({ OsToggle: () => <div>OS toggle</div> }));

import { fireEvent, render, screen } from "@testing-library/react";
import { SetupWizardModal } from "../SetupWizardModal";
import type { SetupConnectorCfg, StepDef } from "@/modules/integrations/types";

const steps: StepDef[] = [
  { title: "Open terminal", description: "Find your terminal" },
  { title: "Install plugin", description: "Run the install command" },
  { title: "Start Claude Code", description: "Check the connection" },
];

const cfg: SetupConnectorCfg = {
  key: "claude-code",
  name: "Claude Code",
  cta: "Connect via plugin",
  description: "",
  icon: <span />,
  buildSteps: () => steps,
};

it("shows all three plugin setup steps while waiting for a connection", () => {
  render(<SetupWizardModal cfg={cfg} steps={steps} currentStep={0} cogniInstance={null} onClose={jest.fn()} onStepSelect={jest.fn()} />);

  expect(screen.getByText("Find your terminal")).toBeInTheDocument();
  expect(screen.getByText("Run the install command")).toBeInTheDocument();
  expect(screen.getByText("Check the connection")).toBeInTheDocument();

  fireEvent.click(screen.getByText("Open terminal"));
  expect(screen.getByText("Find your terminal").closest("[aria-hidden]")).toHaveAttribute("aria-hidden", "true");
  expect(screen.getByText("Check the connection")).toBeInTheDocument();
});
