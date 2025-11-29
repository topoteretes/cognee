"""Common CSS styles for TUI screens to reduce repetition."""

COMMON_STYLES = """
/* Common screen background */
Screen {
    background: $surface;
}

/* Common container styles */
.tui-container {
    height: 100%;
    padding: 1;
}

.tui-content-container {
    height: auto;
    padding: 1;
    content-align: center middle;
}

.tui-form {
    width: 100%;
    height: auto;
    border: solid $primary;
    padding: 2;
    background: $surface;
}

.tui-form-compact {
    border: solid $primary;
    padding: 1;
    background: $surface;
}

/* Common title styles */
.tui-title {
    text-align: center;
    text-style: bold;
    color: $accent;
    margin-bottom: 2;
    width: 100%;
}

/* Common label styles */
.tui-label {
    color: $text-muted;
    margin-bottom: 1;
}

.tui-label-spaced {
    color: $text-muted;
    margin-top: 1;
    margin-bottom: 1;
}

/* Common input styles */
Input {
    width: 100%;
    margin-bottom: 1;
}

/* Common button styles */
Button {
    margin: 0 1;
}

/* Common status message styles */
.tui-status {
    text-align: center;
    margin-top: 2;
    height: auto;
}

/* Common footer styles */
.tui-footer {
    dock: bottom;
    padding: 1 0;
    background: $boost;
    color: $text-muted;
    content-align: center middle;
    border: solid $primary;
}

/* Common dialog/modal styles */
.tui-dialog {
    border: thick $warning;
    background: $surface;
    padding: 2;
}

.tui-dialog-title {
    text-align: center;
    text-style: bold;
    color: $warning;
    margin-bottom: 1;
}

.tui-dialog-message {
    text-align: center;
    margin-bottom: 1;
}

.tui-dialog-buttons {
    align: center middle;
    height: 3;
}

/* Common input group styles */
.tui-input-group {
    height: auto;
    margin-bottom: 2;
}
"""