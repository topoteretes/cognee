import { render, screen } from "@testing-library/react";
import DocumentList, { type DocRow } from "../DocumentList";

const HASHED = "text_0f260db34f3b337254309d879ea8a12a";

const REPOSITORY: DocRow = {
  id: "repo-1",
  name: HASHED,
  label: "code_repo:github.com-topoteretes-cognee-website",
  extension: "txt",
  size: 22285,
  createdAt: "2026-09-28T08:04:00Z",
};

function renderList(docs: DocRow[]) {
  render(<DocumentList docs={docs} onDelete={jest.fn()} />);
}

describe("DocumentList", () => {
  it("names a synced repository from its label and badges it as one", () => {
    renderList([REPOSITORY]);

    expect(screen.getByText("github.com-topoteretes-cognee-website")).toBeInTheDocument();
    // The badge is drawn twice: in the icon and in the type column.
    expect(screen.getAllByText("GIT")).toHaveLength(2);
    expect(screen.queryByText(HASHED)).not.toBeInTheDocument();
  });

  it("treats a null label from the API as no label, as the uncast rows can carry one", () => {
    renderList([{ ...REPOSITORY, label: null as unknown as string }]);

    expect(screen.getByText(HASHED)).toBeInTheDocument();
    expect(screen.queryByText("GIT")).not.toBeInTheDocument();
  });

  it("leaves an ordinary file as it was", () => {
    renderList([{ id: "f-1", name: "roadmap.docx", extension: "docx", size: 4096 }]);

    expect(screen.getByText("roadmap.docx")).toBeInTheDocument();
    expect(screen.getAllByText("DOC")).toHaveLength(2);
    expect(screen.queryByText("GIT")).not.toBeInTheDocument();
  });
});
