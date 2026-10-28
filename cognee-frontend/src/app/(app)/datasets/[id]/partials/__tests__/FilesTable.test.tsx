import { render, screen } from "@testing-library/react";
import type { TextBlobDescription } from "@/modules/datasets/describeTextBlob";
import FilesTable, { type FileRow } from "../FilesTable";

const BLOB: FileRow = {
  id: "blob-1",
  name: "text_0123456789abcdef.txt",
  extension: "txt",
  size: 120,
  createdAt: "2026-09-01T10:00:00Z",
};

const REPOSITORY: FileRow = {
  id: "repo-1",
  name: "text_0f260db34f3b337254309d879ea8a12a",
  label: "code_repo:github.com-topoteretes-cognee-website",
  extension: "txt",
  size: 22285,
  createdAt: "2026-09-28T08:04:00Z",
};

const NAMED: FileRow = {
  id: "file-1",
  name: "roadmap.docx",
  extension: "docx",
  size: 4096,
  createdAt: "2026-09-01T10:00:00Z",
};

function renderTable(files: FileRow[], blobDescriptions: Record<string, TextBlobDescription | null>) {
  render(
    <FilesTable
      files={files}
      blobDescriptions={blobDescriptions}
      search=""
      loadError={false}
      onDelete={jest.fn()}
      onUploadClick={jest.fn()}
      onRetry={jest.fn()}
    />,
  );
}

describe("FilesTable", () => {
  // The whole point of reading the blob is that its name proves nothing. A row
  // that paints "Text" before the read answers is making the same unproven
  // claim "Memory" used to (review-reported).
  it("claims neither Text nor Memory while a blob's content is still being read", () => {
    renderTable([BLOB], {});

    expect(screen.queryByText("Text")).not.toBeInTheDocument();
    expect(screen.queryByText("Memory")).not.toBeInTheDocument();
  });

  it("shows a memory once its content says so", () => {
    renderTable([BLOB], { "blob-1": { kind: "memory", sessionId: "abc-123" } });

    expect(screen.getByText("Memory · abc-123")).toBeInTheDocument();
    expect(screen.getByText("Memory")).toBeInTheDocument();
  });

  it("titles a non-memory blob from its content and types it as text", () => {
    renderTable([BLOB], { "blob-1": { kind: "text", title: "Q3 roadmap" } });

    expect(screen.getByText("Q3 roadmap")).toBeInTheDocument();
    expect(screen.getByText("Text")).toBeInTheDocument();
  });

  it("falls back to Text when the blob could not be read", () => {
    renderTable([BLOB], { "blob-1": null });

    expect(screen.getAllByText("Text").length).toBeGreaterThan(0);
  });

  it("never waits on a read for a file that already has a real name", () => {
    renderTable([NAMED], {});

    expect(screen.getByText("roadmap.docx")).toBeInTheDocument();
    expect(screen.getByText("DOCX")).toBeInTheDocument();
  });

  it("names a synced repository from its label without waiting on a read", () => {
    renderTable([REPOSITORY], {});

    expect(screen.getByText("github.com-topoteretes-cognee-website")).toBeInTheDocument();
    expect(screen.getByText("Repository")).toBeInTheDocument();
    expect(screen.queryByText("Text")).not.toBeInTheDocument();
    expect(screen.queryByText(REPOSITORY.name)).not.toBeInTheDocument();
  });

  it("badges a synced repository as one, not as a text file", () => {
    renderTable([REPOSITORY], {});

    expect(screen.getByText("GIT")).toBeInTheDocument();
    expect(screen.queryByText("TXT")).not.toBeInTheDocument();
  });

  it("clips a long repository name in the cell and keeps the whole of it as a tooltip", () => {
    const long = `github.com-${"o".repeat(39)}-${"r".repeat(100)}`;
    renderTable([{ ...REPOSITORY, label: `code_repo:${long}` }], {});

    const cell = screen.getByText(`${long.slice(0, 80)}…`);
    expect(cell).toHaveAttribute("title", long);
  });

  it("treats a null label from the API as no label", () => {
    renderTable([{ ...BLOB, label: null as unknown as string }], { "blob-1": { kind: "text", title: "Q3 roadmap" } });

    expect(screen.getByText("Q3 roadmap")).toBeInTheDocument();
    expect(screen.queryByText("Repository")).not.toBeInTheDocument();
  });

  it("keeps a real filename whatever label the row carries", () => {
    renderTable([{ ...NAMED, label: "code_repo:github.com-acme-api" }], {});

    expect(screen.getByText("roadmap.docx")).toBeInTheDocument();
    expect(screen.queryByText("Repository")).not.toBeInTheDocument();
  });
});
