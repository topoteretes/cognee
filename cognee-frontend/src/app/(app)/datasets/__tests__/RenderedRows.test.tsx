import { fireEvent, render, screen } from "@testing-library/react";
import DocumentList from "../partials/DocumentList";
import FilesTable from "../[id]/partials/FilesTable";
import { MAX_RENDERED_ROWS } from "@/modules/datasets/maxRenderedRows";
import { fmtCount } from "@/utils/formatCount";

const files = Array.from({ length: MAX_RENDERED_ROWS + 1 }, (_, i) => ({ id: String(i), name: `file-${i}.txt` }));

test("DocumentList caps actual rows while keeping delete actions on the correct document", () => {
  const onDelete = jest.fn();
  const { container } = render(<DocumentList docs={files} onDelete={onDelete} />);
  expect(container.querySelectorAll('button[title="Delete file"]')).toHaveLength(MAX_RENDERED_ROWS);
  expect(screen.queryByText(`file-${fmtCount(MAX_RENDERED_ROWS)}.txt`)).not.toBeInTheDocument();
  expect(screen.getByText(`Showing ${fmtCount(MAX_RENDERED_ROWS)} of ${fmtCount(files.length)} documents`)).toBeInTheDocument();
  fireEvent.click(container.querySelector('button[title="Delete file"]')!);
  expect(onDelete).toHaveBeenCalledWith(files[0]);
});

test("FilesTable caps actual rows and does not render a hidden file", () => {
  const { container } = render(<FilesTable files={files} blobDescriptions={{}} search="" loadError={false}
    onDelete={jest.fn()} onUploadClick={jest.fn()} onRetry={jest.fn()} />);
  expect(container.querySelectorAll('button[title="Delete file"]')).toHaveLength(MAX_RENDERED_ROWS);
  expect(screen.queryByText(`file-${fmtCount(MAX_RENDERED_ROWS)}.txt`)).not.toBeInTheDocument();
  expect(screen.getByText(`Showing ${fmtCount(MAX_RENDERED_ROWS)} of ${fmtCount(files.length)} files`)).toBeInTheDocument();
});


test("DocumentList displays raw API sizes and mapped sizes", () => {
  render(<DocumentList docs={[{ id: "raw", name: "raw.txt", dataSize: 1024 },
    { id: "mapped", name: "mapped.txt", size: 2048 }]} onDelete={jest.fn()} />);
  expect(screen.getByText("1.0 KB")).toBeInTheDocument();
  expect(screen.getByText("2.0 KB")).toBeInTheDocument();
});
