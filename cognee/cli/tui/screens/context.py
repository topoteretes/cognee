"""Context Management Screen"""
import asyncio
import io
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Button, Static, Input, DataTable, Checkbox
from textual.containers import Container, Vertical
from textual.binding import Binding
from textual.events import Key
import cognee
from cognee.api.v1.datasets.datasets import datasets as ds_api


class ContextScreen(Screen):
    """Context management screen"""
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("up", "arrow_up", show=False),
        Binding("down", "arrow_down", show=False),
        Binding("left", "arrow_left", show=False),
        Binding("right", "arrow_right", show=False),
    ]
    DEFAULT_DATASET = "main_dataset"
    
    def __init__(self) -> None:
        super().__init__()
        self._datasets: list[dict] = []
        self._dataset_id_to_name: dict[str, str] = {}
        self._data_items_by_id: dict[str, dict] = {}
        self._dataset_row_to_id: dict[DataTable.RowKey, str] = {}
        self._file_row_to_id: dict[DataTable.RowKey, str] = {}
        self._dataset_row_keys: list[DataTable.RowKey] = []
        self._file_row_keys: list[DataTable.RowKey] = []
        self._selected_dataset_id: str | None = None
        self._selected_data_id: str | None = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Static("[bold]📁 Context Management[/bold]\n", classes="title")
            with Vertical():
                yield Static("[b]Datasets[/b] and [b]Files[/b] in dataset", classes="center")
                # Central tables
                yield DataTable(id="datasets_table")
                yield DataTable(id="files_table")
                # Tables are the primary UI (no dropdowns)
                yield Input(placeholder="comma-separated node sets (optional)", id="nodeset_input")
                yield Static("\nEnter text or a file path to add to the selected dataset:", classes="center")
                yield Input(placeholder="Text or /absolute/path/to/file.pdf", id="data_input")
                yield Button("Add to Context", id="add_btn", variant="primary")
                yield Button("Cognify (process data)", id="cognify_btn", variant="success")
                yield Static("\nSearch (runs against selected dataset context):", classes="center")
                yield Input(placeholder="e.g., What are the main topics?", id="search_input")
                yield Button("Search", id="search_btn", variant="default")
                yield Checkbox("Save search output to searched_context.md", id="save_search_checkbox", value=False)
                yield Static("\nExport context to Markdown (runs one or more queries):", classes="center")
                yield Input(placeholder="Queries to export (comma-separated)", id="export_queries")
                yield Button("Export Context to MD", id="export_btn", variant="default")
                yield Static("", id="status")
            yield Button("← Back", id="back_btn")
        yield Footer()
    
    async def _set_status(self, message: str) -> None:
        try:
            status = self.query_one("#status", Static)
            status.update(message)
        except Exception:
            pass
    
    async def on_mount(self) -> None:
        # Load datasets and populate select
        try:
            await self._set_status("Loading datasets...")
            datasets = await ds_api.list_datasets()
            # datasets may be model objects; normalize to dicts with id, name
            normalized = []
            for d in datasets:
                # Support both object and dict
                d_id = str(getattr(d, "id", None) or d.get("id"))
                d_name = str(getattr(d, "name", None) or d.get("name"))
                normalized.append({"id": d_id, "name": d_name})
            self._datasets = normalized
            self._dataset_id_to_name = {d["id"]: d["name"] for d in normalized if d.get("id")}
            # Init datasets table
            ds_table = self.query_one("#datasets_table", DataTable)
            if not ds_table.columns:
                ds_table.add_columns("Name", "ID", "Created At")
            ds_table.clear()
            self._dataset_row_to_id.clear()
            self._dataset_row_keys = []
            for d in normalized:
                row_key = ds_table.add_row(d.get("name", ""), d.get("id", ""), d.get("created_at", "") or "")
                self._dataset_row_to_id[row_key] = d["id"]
                self._dataset_row_keys.append(row_key)
            # Focus datasets table and preselect first row if available
            ds_table.focus()
            if self._dataset_row_to_id:
                first_row_key = next(iter(self._dataset_row_to_id.keys()))
                self._selected_dataset_id = self._dataset_row_to_id[first_row_key]
                # Try to position the cursor on the first cell to ensure arrows work
                try:
                    ds_table.cursor_coordinate = (0, 0)
                except Exception:
                    pass
                await self._load_dataset_files(self._selected_dataset_id)
            await self._set_status("Datasets loaded. Select a dataset to view files.")
        except Exception as ex:
            await self._set_status(f"[red]Failed to load datasets:[/red] {ex}")
    
    async def _load_dataset_files(self, dataset_id: str) -> None:
        try:
            await self._set_status("Loading dataset files...")
            data_items = await ds_api.list_data(dataset_id)
            normalized = []
            self._data_items_by_id = {}
            for item in data_items:
                i_id = str(getattr(item, "id", None) or item.get("id"))
                i_name = str(getattr(item, "name", None) or item.get("name") or "Unnamed")
                raw_loc = (
                    getattr(item, "raw_data_location", None)
                    or item.get("raw_data_location")
                    or item.get("rawDataLocation")
                )
                orig_loc = (
                    getattr(item, "original_data_location", None)
                    or item.get("original_data_location")
                    or item.get("originalDataLocation")
                )
                normalized.append(
                    {
                        "id": i_id,
                        "name": i_name,
                        "raw_data_location": raw_loc,
                        "original_data_location": orig_loc,
                        "raw": raw_loc,
                        "orig": orig_loc,
                    }
                )
                self._data_items_by_id[i_id] = normalized[-1]
            # Populate files table
            files_table = self.query_one("#files_table", DataTable)
            if not files_table.columns:
                files_table.add_columns("Name", "ID", "Path")
            files_table.clear()
            self._file_row_to_id.clear()
            self._file_row_keys = []
            for i in normalized:
                row_key = files_table.add_row(i.get("name", ""), i.get("id", ""), (i.get("original_data_location") or i.get("raw_data_location") or i.get("orig") or i.get("raw") or "") )
                self._file_row_to_id[row_key] = i["id"]
                self._file_row_keys.append(row_key)
            self._selected_data_id = None
            await self._set_status(f"Loaded {len(normalized)} file(s) for the dataset.")
        except Exception as ex:
            await self._set_status(f"[red]Failed to load dataset files:[/red] {ex}")
    
    async def _handle_add(self) -> None:
        data_input = self.query_one("#data_input", Input)
        nodeset_input = self.query_one("#nodeset_input", Input)
        content = (data_input.value or "").strip()
        selected_dataset_id = self._selected_dataset_id
        dataset_name = self._dataset_id_to_name.get(selected_dataset_id, self.DEFAULT_DATASET)
        raw_nodeset = (nodeset_input.value or "").strip()
        node_set = [s.strip() for s in raw_nodeset.split(",") if s.strip()] if raw_nodeset else None
        if not content:
            await self._set_status("[red]Please enter text or a file path[/red]")
            return
        try:
            await self._set_status(f"Adding data to dataset [b]{dataset_name}[/b] "
                                   f"{'(with node sets: ' + ', '.join(node_set) + ')' if node_set else ''}...")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                await cognee.add(content, dataset_name=dataset_name, node_set=node_set)
            await self._set_status("[green]✓ Added successfully.[/green] You can now run Cognify.")
        except Exception as ex:
            await self._set_status(f"[red]Add failed:[/red] {ex}")
    
    async def _handle_cognify(self) -> None:
        try:
            await self._set_status("Processing data into knowledge graph (cognify)...")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                await cognee.cognify()
            await self._set_status("[green]✓ Cognify complete.[/green]")
        except Exception as ex:
            await self._set_status(f"[red]Cognify failed:[/red] {ex}")
    
    async def _handle_search(self) -> None:
        try:
            dataset_id = self._selected_dataset_id
            ds_name = self._dataset_id_to_name.get(dataset_id, None)
            q_input = self.query_one("#search_input", Input)
            save_cb = self.query_one("#save_search_checkbox", Checkbox)
            query_text = (q_input.value or "").strip()
            if not query_text:
                await self._set_status(":warning: Please enter a search query.")
                return
            await self._set_status("Searching...")
            # If a dataset is chosen, we can scope via datasets=[name]
            kwargs = {}
            if ds_name:
                kwargs["datasets"] = [ds_name]
            results = await cognee.search(query_text=query_text, **kwargs)
            rendered = "\n".join(f"- {str(item)}" for item in results) if isinstance(results, list) else str(results)
            await self._set_status(f"[b]Search results[/b]:\n{rendered}")
            # Optionally save to searched_context.md
            if save_cb.value:
                # Choose directory next to selected file if possible, else current working dir
                target_dir: Path | None = None
                if self._selected_data_id:
                    data_item = self._data_items_by_id.get(self._selected_data_id)
                    if data_item:
                        loc = data_item.get("original_data_location") or data_item.get("raw_data_location") or data_item.get("orig") or data_item.get("raw")
                        if isinstance(loc, str) and loc.startswith("file://"):
                            loc = loc[len("file://") :]
                        if isinstance(loc, str) and (loc.startswith("/") or (len(loc) > 2 and loc[1:3] in (":\\", ":/"))):
                            try:
                                p = Path(loc)
                                target_dir = p.parent if p.exists() or p.parent.exists() else None
                            except Exception:
                                target_dir = None
                if target_dir is None:
                    target_dir = Path.cwd()
                out_path = target_dir / "searched_context.md"
                try:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(f"### Query: {query_text}\n{rendered}\n\n")
                    await self._set_status(f"[green]✓ Saved search output to:[/green] {out_path}")
                except Exception as ex:
                    await self._set_status(f"[red]Failed to save search output:[/red] {ex}")
        except Exception as ex:
            await self._set_status(f"[red]Search failed:[/red] {ex}")
    
    def _choose_export_path(self, data_item: dict) -> Path | None:
        # Prefer original path; fallback to raw path
        loc = data_item.get("orig") or data_item.get("raw")
        if not loc:
            return None
        # Accept absolute POSIX paths or file:// URIs
        if isinstance(loc, str) and loc.startswith("file://"):
            loc = loc[len("file://") :]
        if not isinstance(loc, str) or not (loc.startswith("/") or loc[1:3] == ":\\" or loc[1:3] == ":/"):
            return None
        try:
            p = Path(loc)
            if p.is_dir():
                return p / "context_export.md"
            # write next to file with suffix
            stem = p.stem
            return p.with_name(f"{stem}_context.md")
        except Exception:
            return None
    
    async def _handle_export(self) -> None:
        try:
            export_queries = self.query_one("#export_queries", Input)
            data_id = self._selected_data_id
            if not data_id:
                await self._set_status(":warning: Please select a file to export context for.")
                return
            data_item = self._data_items_by_id.get(data_id)
            if not data_item:
                await self._set_status(":warning: Selected file metadata not available.")
                return
            export_path = self._choose_export_path(data_item)
            if not export_path:
                await self._set_status("[red]Can't determine a local file path to save Markdown next to the original file.[/red]")
                return
            raw_queries = (export_queries.value or "").strip()
            queries = [q.strip() for q in raw_queries.split(",") if q.strip()]
            await self._set_status("Running export...")
            md_parts: list[str] = []
            md_parts.append(f"# Context Export for {data_item.get('name','selected item')}")
            # Include simple metadata block
            md_parts.append("")
            md_parts.append("## Source")
            md_parts.append(f"- Dataset: {self._dataset_id_to_name.get(self._selected_dataset_id, self.DEFAULT_DATASET)}")
            if data_item.get("orig"):
                md_parts.append(f"- Original: {data_item.get('orig')}")
            if data_item.get("raw"):
                md_parts.append(f"- Raw: {data_item.get('raw')}")
            # Run queries if provided
            if queries:
                md_parts.append("\n## Search Results")
                for q in queries:
                    md_parts.append(f"\n### Query: {q}\n")
                    try:
                        ds_name = self._dataset_id_to_name.get(self._selected_dataset_id, None)
                        kwargs = {}
                        if ds_name:
                            kwargs["datasets"] = [ds_name]
                        results = await cognee.search(query_text=q, **kwargs)
                        if isinstance(results, list):
                            if results:
                                for r in results:
                                    md_parts.append(f"- {str(r)}")
                            else:
                                md_parts.append("- (no results)")
                        else:
                            md_parts.append(str(results))
                    except Exception as ex:
                        md_parts.append(f"- (search failed: {ex})")
            # Write file
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_parts).strip() + "\n")
            await self._set_status(f"[green]✓ Exported to:[/green] {export_path}")
        except Exception as ex:
            await self._set_status(f"[red]Export failed:[/red] {ex}")
    
    def on_button_pressed(self, event) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
            return
        if event.button.id == "add_btn":
            asyncio.create_task(self._handle_add())
            return
        if event.button.id == "cognify_btn":
            asyncio.create_task(self._handle_cognify())
            return
        if event.button.id == "search_btn":
            asyncio.create_task(self._handle_search())
            return
        if event.button.id == "export_btn":
            asyncio.create_task(self._handle_export())
            return
    
    def on_data_table_row_selected(self, message: DataTable.RowSelected) -> None:
        # Selecting a dataset row loads its files and syncs dropdown
        try:
            if message.data_table.id == "datasets_table":
                row_key = message.row_key
                dataset_id = self._dataset_row_to_id.get(row_key)
                if dataset_id:
                    self._selected_dataset_id = dataset_id
                    asyncio.create_task(self._load_dataset_files(dataset_id))
            elif message.data_table.id == "files_table":
                row_key = message.row_key
                data_id = self._file_row_to_id.get(row_key)
                if data_id:
                    self._selected_data_id = data_id
        except Exception:
            pass
    
    def _active_table(self) -> DataTable | None:
        try:
            files_table = self.query_one("#files_table", DataTable)
            datasets_table = self.query_one("#datasets_table", DataTable)
            if files_table.has_focus:
                return files_table
            return datasets_table
        except Exception:
            return None
    
    def action_arrow_up(self) -> None:
        table = self._active_table()
        if table:
            try:
                table.action_cursor_up()
            except Exception:
                pass
    
    def action_arrow_down(self) -> None:
        table = self._active_table()
        if table:
            try:
                table.action_cursor_down()
            except Exception:
                pass
    
    def action_arrow_left(self) -> None:
        table = self._active_table()
        if table:
            try:
                # Move focus to datasets table on left, otherwise move cursor
                if table.id == "files_table":
                    self.query_one("#datasets_table", DataTable).focus()
                else:
                    table.action_cursor_left()
            except Exception:
                pass
    
    def action_arrow_right(self) -> None:
        table = self._active_table()
        if table:
            try:
                # Move focus to files table on right, otherwise move cursor
                if table.id == "datasets_table":
                    self.query_one("#files_table", DataTable).focus()
                else:
                    table.action_cursor_right()
            except Exception:
                pass

    def on_key(self, event: Key) -> None:
        """Fallback manual cursor movement to guarantee arrow navigation."""
        table = self._active_table()
        if not table:
            return
        try:
            # Determine current row from cursor_coordinate if available
            current_row = 0
            try:
                coord = table.cursor_coordinate
                if coord and isinstance(coord, tuple):
                    current_row = int(coord[0])
            except Exception:
                pass
            if event.key in ("up", "down"):
                max_rows = len(self._file_row_keys) if table.id == "files_table" else len(self._dataset_row_keys)
                if max_rows <= 0:
                    return
                if event.key == "up":
                    new_row = max(0, current_row - 1)
                else:
                    new_row = min(max_rows - 1, current_row + 1)
                try:
                    table.cursor_coordinate = (new_row, 0)
                    event.stop()
                except Exception:
                    pass
            elif event.key == "left":
                if table.id == "files_table":
                    self.query_one("#datasets_table", DataTable).focus()
                    event.stop()
            elif event.key == "right":
                if table.id == "datasets_table":
                    self.query_one("#files_table", DataTable).focus()
                    event.stop()
        except Exception:
            pass
    
    def action_back(self) -> None:
        self.app.pop_screen()
