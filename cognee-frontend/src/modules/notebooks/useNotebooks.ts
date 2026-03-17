import { useCallback, useState } from "react";
import { Cell, Notebook } from "@/ui/elements/Notebook/types";
import { CogneeInstance } from "@/modules/instances/types";
import createNotebook from "./createNotebook";
import deleteNotebook from "./deleteNotebook";
import getNotebooks from "./getNotebooks";
import runNotebookCell from "./runNotebookCell";
import { default as persistNotebook } from "./saveNotebook";

function useNotebooks(instance: CogneeInstance) {
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);

  const addNotebook = useCallback((notebookName: string) => {
    return createNotebook(notebookName, instance)
      .then((notebook: Notebook) => {
        setNotebooks((notebooks) => [
          ...notebooks,
          notebook,
        ]);

        return notebook;
      });
  }, [instance]);

  const removeNotebook = useCallback((notebookId: string) => {
    return deleteNotebook(notebookId, instance)
    .then(() => {
      setNotebooks((notebooks) =>
        notebooks.filter((notebook) => notebook.id !== notebookId)
      );
    });
  }, [instance]);

  const fetchNotebooks = useCallback(() => {
    return getNotebooks(instance)
    .then((notebooks) => {
      setNotebooks(notebooks);

      return notebooks;
    })
    .catch((error) => {
      console.error("Error fetching notebooks:", error.detail);
      throw error
    });
  }, [instance]);

  const updateNotebook = useCallback((updatedNotebook: Notebook) => {
    setNotebooks((existingNotebooks) =>
      existingNotebooks.map((notebook) =>
        notebook.id === updatedNotebook.id
          ? updatedNotebook
          : notebook
      )
    );
  }, []);

  const saveNotebook = useCallback((notebook: Notebook) => {
    return persistNotebook(notebook.id, {
      name: notebook.name,
      cells: notebook.cells,
    }, instance);
  }, [instance]);

  const runCell = useCallback((notebook: Notebook, cell: Cell) => {
    setNotebooks((existingNotebooks) =>
      existingNotebooks.map((existingNotebook) =>
        existingNotebook.id === notebook.id ? {
          ...existingNotebook,
          cells: existingNotebook.cells.map((existingCell) =>
            existingCell.id === cell.id ? {
              ...existingCell,
              result: undefined,
              error: undefined,
            } : existingCell
          ),
        } : existingNotebook
      )
    );

    return runNotebookCell(notebook.id, cell, instance)
      .then((response) => {
        setNotebooks((existingNotebooks) =>
          existingNotebooks.map((existingNotebook) =>
            existingNotebook.id === notebook.id ? {
              ...existingNotebook,
              cells: existingNotebook.cells.map((existingCell) =>
                existingCell.id === cell.id ? {
                 ...existingCell,
                  result: response.result,
                  error: response.error,
                } : existingCell
              ),
            } : existingNotebook
          )
        );
      });
  }, [instance]);

  return {
    notebooks,
    addNotebook,
    saveNotebook,
    updateNotebook,
    removeNotebook,
    refreshNotebooks: fetchNotebooks,
    runCell,
  };
};

export default useNotebooks;
