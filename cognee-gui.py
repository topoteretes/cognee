import sys
import asyncio
import cognee
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QVBoxLayout,
    QLabel,
    QMessageBox,
    QTextEdit,
)


class FileSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Button to open file dialog
        self.file_button = QPushButton("Upload File to Cognee", self)
        self.file_button.clicked.connect(self.open_file_dialog)

        # Label to display selected file path
        self.file_label = QLabel("No file selected", self)

        # Line edit for search input
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Enter text to search...")

        # Button to perform search
        self.search_button = QPushButton("Cognee Search", self)
        self.search_button.clicked.connect(self.cognee_search)

        # Text output area for search results
        self.result_output = QTextEdit(self)
        self.result_output.setReadOnly(True)
        self.result_output.setPlaceholderText("Search results will appear here...")

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.file_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.search_input)
        layout.addWidget(self.search_button)
        layout.addWidget(self.result_output)

        self.setLayout(layout)
        self.setWindowTitle("Cognee")
        self.resize(500, 300)

        # Initialize selected file path
        self.selected_file = None

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a File", "", "All Files (*.*);;Text Files (*.txt)"
        )
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(f"Selected: {file_path}")

            # Ensure an event loop is running before creating the async task
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                asyncio.run(self.process_file_async())  # Run coroutine in blocking mode
            else:
                asyncio.create_task(self.process_file_async())

    async def process_file_async(self):
        """Handles async calls within PyQt."""
        try:
            await cognee.add(self.selected_file)
            await cognee.cognify()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"File processing failed: {str(e)}")

    def cognee_search(self):
        try:
            search_text = self.search_input.text().strip()
            # Ensure an event loop is running before creating the async task
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                query = asyncio.run(
                    self.async_search(search_text)
                )  # Run coroutine in blocking mode
            else:
                query = asyncio.create_task(self.async_search(search_text))

            self.result_output.setText(query[0])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file: {str(e)}")

    async def async_search(self, search_text):
        return await cognee.search(query_text=search_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Ensure an asyncio event loop is running for PyQt
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    window = FileSearchApp()
    window.show()
    sys.exit(app.exec_())
