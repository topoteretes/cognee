import importlib.metadata
import logging
import logging.handlers
import os
import platform
import sys
import tempfile
import traceback
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import structlog


def _get_cognee_version() -> str:
    """Get version without importing cognee (avoids triggering __init__.py)."""
    import importlib.metadata as _meta
    from contextlib import suppress

    with suppress(FileNotFoundError, StopIteration):
        _pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(_pyproject, encoding="utf-8") as f:
            _ver = (
                next(line for line in f if line.startswith("version")).split("=")[1].strip("'\"\n ")
            )
            return f"{_ver}-local"
    try:
        return _meta.version("cognee")
    except _meta.PackageNotFoundError:
        return "unknown"


cognee_version = _get_cognee_version()


# Configure external library logging
def configure_external_library_logging() -> None:
    """Configure logging for external libraries to reduce verbosity.

    Sets env vars eagerly (cheap) but only configures litellm's Python
    objects if litellm is already imported. If not, the env vars are
    enough — litellm reads them on its own import.
    """
    # Set environment variables to suppress LiteLLM logging.
    # litellm reads these on import, so setting them early is sufficient
    # even if litellm hasn't been imported yet.
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    os.environ.setdefault("LITELLM_SET_VERBOSE", "False")

    # Suppress loggers by name (works even before litellm is imported —
    # Python's logging module pre-creates the logger objects).
    loggers_to_suppress = [
        "litellm",
        "litellm.litellm_core_utils.logging_worker",
        "litellm.litellm_core_utils",
        "litellm.proxy",
        "litellm.router",
        "openai._base_client",
        "LiteLLM",
        "LiteLLM.core",
        "LiteLLM.logging_worker",
        "litellm.logging_worker",
    ]
    for logger_name in loggers_to_suppress:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        logging.getLogger(logger_name).disabled = True

    # Only touch litellm's module-level flags if it's already imported.
    # This avoids a ~900ms cold import just to set verbose=False.
    litellm = sys.modules.get("litellm")
    if litellm is not None:
        litellm.set_verbose = False  # ty:ignore[unresolved-attribute]
        if hasattr(litellm, "suppress_debug_info"):
            litellm.suppress_debug_info = True  # ty:ignore[unresolved-attribute]
        if hasattr(litellm, "turn_off_message"):
            litellm.turn_off_message = True  # ty:ignore[unresolved-attribute]
        if hasattr(litellm, "_turn_on_debug"):
            litellm._turn_on_debug = False  # ty:ignore[unresolved-attribute]


# Export common log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

log_levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

# Track if structlog logging has been configured
_is_structlog_configured = False


def resolve_logs_dir() -> Path | None:
    """Resolve a writable logs directory.

    Priority:
    1) BaseConfig.logs_root_directory (respects COGNEE_LOGS_DIR)
    2) /tmp/cognee_logs (default, best-effort create)

    Returns a Path or None if none are writable/creatable.
    """
    from cognee.base_config import get_base_config

    base_config = get_base_config()
    logs_root_directory = Path(base_config.logs_root_directory)

    try:
        logs_root_directory.mkdir(parents=True, exist_ok=True)
        if os.access(logs_root_directory, os.W_OK):
            return logs_root_directory
    except Exception:
        pass

    try:
        tmp_log_path = Path(os.path.join("/tmp", "cognee_logs"))
        tmp_log_path.mkdir(parents=True, exist_ok=True)
        if os.access(tmp_log_path, os.W_OK):
            return tmp_log_path
    except Exception:
        pass

    return None


# Maximum number of log files to keep
MAX_LOG_FILES = 10

# Log rotation defaults — override via COGNEE_LOG_MAX_BYTES / COGNEE_LOG_BACKUP_COUNT
LOG_MAX_BYTES = int(os.getenv("COGNEE_LOG_MAX_BYTES", 50 * 1024 * 1024))  # 50 MB
LOG_BACKUP_COUNT = int(os.getenv("COGNEE_LOG_BACKUP_COUNT", 5))  # 5 backups → 300 MB cap

# Version information
PYTHON_VERSION = platform.python_version()
STRUCTLOG_VERSION = structlog.__version__
COGNEE_VERSION = cognee_version

OS_INFO = f"{platform.system()} {platform.release()} ({platform.version()})"


class PlainFileHandler(logging.handlers.RotatingFileHandler):
    """A rotating file handler that writes simpler plain text log entries.

    Inherits from RotatingFileHandler so log files are automatically rotated
    when they reach maxBytes, keeping at most backupCount old files.
    """

    def emit(self, record) -> None:
        try:
            # Check if stream is available before trying to write
            if self.stream is None:
                self.stream = self._open()

            # Extract the message from the structlog record
            if isinstance(record.msg, dict) and "event" in record.msg:
                # Extract the basic message
                message = record.msg.get("event", "")

                # Extract additional context
                context = {
                    k: v
                    for k, v in record.msg.items()
                    if k not in ("event", "logger", "level", "timestamp")
                }

                # Format context if present
                context_str = ""
                if context:
                    context_str = " " + " ".join(
                        f"{k}={v}" for k, v in context.items() if k != "exc_info"
                    )

                # Get the logger name from the record or from the structlog context
                logger_name = record.msg.get("logger", record.name)

                # Format timestamp
                timestamp = datetime.now().strftime(get_timestamp_format())

                # Create the log entry
                log_entry = f"{timestamp} [{record.levelname.ljust(8)}] {message}{context_str} [{logger_name}]\n"

                # Write to file
                self.stream.write(log_entry)
                self.flush()

                # Handle exception if present
                # Check both record.exc_info and the 'exc_info' in the message dict
                record_has_exc = record.exc_info and record.exc_info != (None, None, None)
                msg_has_exc = "exc_info" in record.msg and record.msg["exc_info"]

                if record_has_exc:
                    # Use the exception info from the record
                    tb_str = "".join(traceback.format_exception(*record.exc_info))
                    self.stream.write(tb_str + "\n")
                    self.flush()
                elif msg_has_exc and isinstance(record.msg["exc_info"], tuple):
                    # Use the exception info from the message
                    tb_str = "".join(traceback.format_exception(*record.msg["exc_info"]))
                    self.stream.write(tb_str + "\n")
                    self.flush()
                elif msg_has_exc and hasattr(record.msg["exc_info"], "__traceback__"):
                    # Handle exceptions that are passed directly
                    exc = record.msg["exc_info"]
                    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    self.stream.write(tb_str + "\n")
                    self.flush()
            else:
                # Fall back to standard handling for non-structlog messages
                msg = self.format(record)
                self.stream.write(msg + self.terminator)
                self.flush()

                # Handle exception if present in regular record
                if record.exc_info and record.exc_info != (None, None, None):
                    tb_str = "".join(traceback.format_exception(*record.exc_info))
                    self.stream.write(tb_str + "\n")
                    self.flush()
        except Exception as e:
            self.handleError(record)
            # Write error about handling this record
            if self.stream:
                self.stream.write(f"Error in log handler: {e}\n")
            self.flush()


def get_logger(name=None, level=None) -> logging.Logger:
    """Get a logger.

    If `setup_logging()` has not been called, returns a standard Python logger.
    If `setup_logging()` has been called, returns a structlog logger.
    """
    if _is_structlog_configured:
        return structlog.get_logger(name if name else __name__)
    else:
        logger = logging.getLogger(name if name else __name__)
        if level is not None:
            logger.setLevel(level)
        return logger


def log_database_configuration(logger) -> None:
    """Log the current database configuration for all database types"""
    # NOTE: Has to be imporated at runtime to avoid circular import
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.relational.config import get_relational_config
    from cognee.infrastructure.databases.vector.config import get_vectordb_config

    try:
        # Get base database directory path
        from cognee.base_config import get_base_config

        base_config = get_base_config()
        databases_path = os.path.join(base_config.system_root_directory, "databases")

        # Log concise database info
        logger.info(f"Database storage: {databases_path}")

    except Exception as e:
        logger.debug(f"Could not retrieve database configuration: {str(e)}")


def cleanup_old_logs(logs_dir, max_files) -> bool:
    """
    Removes old log files, keeping only the most recent ones.

    Args:
        logs_dir: Directory containing log files
        max_files: Maximum number of log files to keep
    """
    logger = structlog.get_logger()
    try:
        # Get all .log files in the directory (excluding README and other files)
        log_files = [f for f in logs_dir.glob("*.log") if f.is_file()]

        # Sort log files by modification time (newest first)
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old files that exceed the maximum
        if len(log_files) > max_files:
            deleted_count = 0
            for old_file in log_files[max_files:]:
                try:
                    old_file.unlink()
                    deleted_count += 1
                    # Only log individual files in non-CLI mode
                    if os.getenv("COGNEE_CLI_MODE") != "true":
                        logger.info(f"Deleted old log file: {old_file}")
                except Exception as e:
                    # Always log errors
                    logger.error(f"Failed to delete old log file {old_file}: {e}")

            # In CLI mode, show compact summary
            if os.getenv("COGNEE_CLI_MODE") == "true" and deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old log files")

        return True
    except Exception as e:
        logger.error(f"Error cleaning up log files: {e}")
        return False


def setup_logging(log_level=None, name=None) -> bool:
    """Sets up the logging configuration with structlog integration.

    Args:
        log_level: The logging level to use (default: None, uses INFO)
        name: Optional logger name (default: None, uses __name__)

    Returns:
        A configured structlog logger instance
    """
    global _is_structlog_configured

    # Regular detailed logging for non-CLI usage
    log_level = log_level if log_level else log_levels[os.getenv("LOG_LEVEL", "INFO").upper()]

    # Configure external library logging early to suppress verbose output
    configure_external_library_logging()

    # Add custom filter to suppress LiteLLM worker cancellation errors
    class LiteLLMCancellationFilter(logging.Filter):
        """Filter to suppress LiteLLM worker cancellation messages"""

        def filter(self, record):
            # Check if this is a LiteLLM-related logger
            if hasattr(record, "name") and "litellm" in record.name.lower():
                return False

            # Check message content for cancellation errors
            if hasattr(record, "msg") and record.msg:
                msg_str = str(record.msg).lower()
                if any(
                    keyword in msg_str
                    for keyword in [
                        "loggingworker cancelled",
                        "logging_worker.py",
                        "cancellederror",
                        "litellm:error",
                    ]
                ):
                    return False

            # Check formatted message
            try:
                if hasattr(record, "getMessage"):
                    formatted_msg = record.getMessage().lower()
                    if any(
                        keyword in formatted_msg
                        for keyword in [
                            "loggingworker cancelled",
                            "logging_worker.py",
                            "cancellederror",
                            "litellm:error",
                        ]
                    ):
                        return False
            except Exception:
                pass

            return True

    # Apply the filter to root logger and specific loggers
    cancellation_filter = LiteLLMCancellationFilter()
    logging.getLogger().addFilter(cancellation_filter)
    logging.getLogger("litellm").addFilter(cancellation_filter)

    # Add custom filter to suppress LiteLLM worker cancellation errors
    class LiteLLMFilter(logging.Filter):
        def filter(self, record) -> bool:
            # Suppress LiteLLM worker cancellation errors
            if hasattr(record, "msg") and isinstance(record.msg, str):
                msg_lower = record.msg.lower()
                if any(
                    phrase in msg_lower
                    for phrase in [
                        "loggingworker cancelled",
                        "cancellederror",
                        "logging_worker.py",
                        "loggingerror",
                    ]
                ):
                    return False
            return True

    # Apply filter to root logger
    litellm_filter = LiteLLMFilter()
    logging.getLogger().addFilter(litellm_filter)

    def exception_handler(
        logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> dict[str, Any]:
        """Custom processor to handle uncaught exceptions."""
        event_dict = dict(event_dict)

        # Check if there's an exc_info that needs to be processed
        if event_dict.get("exc_info"):
            # If it's already a tuple, use it directly
            if isinstance(event_dict["exc_info"], tuple):
                exc_type, exc_value, tb = event_dict["exc_info"]
            else:
                exc_type, exc_value, tb = sys.exc_info()

            if exc_type and hasattr(exc_type, __name__):
                event_dict["exception_type"] = exc_type.__name__
            event_dict["exception_message"] = str(exc_value)
            event_dict["traceback"] = True

        return event_dict

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt=get_timestamp_format(), utc=True),
            structlog.processors.StackInfoRenderer(),
            exception_handler,  # Add our custom exception handler
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up system-wide exception handling
    def handle_exception(exc_type, exc_value, traceback) -> None:
        """Handle any uncaught exception."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Let KeyboardInterrupt pass through
            sys.__excepthook__(exc_type, exc_value, traceback)
            return

        logger = structlog.get_logger()
        logger.error(
            "Exception",
            exc_info=(exc_type, exc_value, traceback),
        )
        # Hand back to the original hook → prints traceback and exits
        sys.__excepthook__(exc_type, exc_value, traceback)

    # Install exception handlers
    sys.excepthook = handle_exception

    # Create console formatter for standard library logging
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(
            colors=True,
            force_colors=True,
            level_styles={
                "critical": structlog.dev.RED,
                "exception": structlog.dev.RED,
                "error": structlog.dev.RED,
                "warn": structlog.dev.YELLOW,
                "warning": structlog.dev.YELLOW,
                "info": structlog.dev.GREEN,
                "debug": structlog.dev.BLUE,
            },  # ty:ignore[invalid-argument-type]
        ),
    )

    # Setup handler with newlines for console output
    class NewlineStreamHandler(logging.StreamHandler):
        def emit(self, record) -> None:
            try:
                msg = self.format(record)
                stream = self.stream
                if hasattr(stream, "closed") and stream.closed:
                    return
                stream.write("\n" + msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)

    # Use our custom handler for console output
    stream_handler = NewlineStreamHandler(sys.stderr)
    stream_handler.setFormatter(console_formatter)
    stream_handler.setLevel(log_level)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)

    # Note: root logger needs to be set at NOTSET to allow all messages through and specific stream and file handlers
    # can define their own levels.
    root_logger.setLevel(logging.NOTSET)

    # --- File logging (opt-out via COGNEE_LOG_FILE=false) ---
    # Set COGNEE_LOG_FILE=false to disable file logging entirely.
    log_file_enabled = os.getenv("COGNEE_LOG_FILE", "true").lower() not in ("false", "0", "no")
    log_file_path = None

    if log_file_enabled:
        # Resolve logs directory with env and safe fallbacks
        logs_dir = resolve_logs_dir()

        # Check if we already have a log file path from the environment
        # NOTE: environment variable must be used here as it allows us to
        # log to a single file with a name based on a timestamp in a multiprocess setting.
        # Without it, we would have a separate log file for every process.
        log_file_path = os.environ.get("LOG_FILE_NAME")
        if not log_file_path and logs_dir is not None:
            # Create a new log file name with the cognee start time
            start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_path = str((logs_dir / f"{start_time}.log").resolve())
            os.environ["LOG_FILE_NAME"] = log_file_path

        try:
            # Rotating file handler: caps each file at LOG_MAX_BYTES,
            # keeps LOG_BACKUP_COUNT old files (default 50 MB × 5 = 250 MB total).
            file_handler = PlainFileHandler(
                log_file_path,  # ty:ignore[invalid-argument-type]
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # Logging to file is not mandatory — warn on console and continue.
            root_logger.warning(
                f"Warning: Could not create log file handler at {log_file_path}: {e}"
            )

    if log_level > logging.DEBUG:
        import warnings

        from sqlalchemy.exc import SAWarning

        warnings.filterwarnings(
            "ignore", category=SAWarning, module="dlt.destinations.impl.sqlalchemy.merge_job"
        )
        warnings.filterwarnings(
            "ignore", category=SAWarning, module="dlt.destinations.impl.sqlalchemy.load_jobs"
        )

    # Clean up old log files, keeping only the most recent ones
    if log_file_enabled and logs_dir is not None:
        cleanup_old_logs(logs_dir, MAX_LOG_FILES)

    # Mark logging as configured
    _is_structlog_configured = True

    # Get a configured logger
    logger = structlog.get_logger(name if name else __name__)

    if log_file_path is not None:
        logger.info(f"Log file created at: {log_file_path}", log_file=log_file_path)

    # Defer heavy database config logging to first actual pipeline use.
    # Importing graph/vector/relational configs triggers litellm (~900ms)
    # and other heavy dependencies. Log basic info now, details later.
    _log_deferred_info(logger)

    return logger


def _log_deferred_info(logger) -> None:
    """Log lightweight startup info. Heavy DB config is logged on first pipeline call."""
    logger.warning(
        "Cognee 1.0 changes: "
        "New API — remember/recall/forget/improve (V1 add/cognify/search still work). "
        "Session memory enabled by default (CACHING=false to disable). "
        "Multi-user access control on by default (ENABLE_BACKEND_ACCESS_CONTROL=false to disable). "
        "Agents (@cognee.agent) auto-verified on registration. "
        "See https://docs.cognee.ai/"
    )

    try:
        from cognee.base_config import get_base_config

        base_config = get_base_config()
        databases_path = os.path.join(base_config.system_root_directory, "databases")
    except Exception:
        databases_path = "unknown"

    logger.info(
        "Logging initialized",
        python_version=PYTHON_VERSION,
        structlog_version=STRUCTLOG_VERSION,
        cognee_version=COGNEE_VERSION,
        os_info=OS_INFO,
        database_path=databases_path,
    )

    logger.info(f"Database storage: {databases_path}")


def get_log_file_location() -> str | None:
    """Return the file path of the log file in use, if any."""
    root_logger = logging.getLogger()

    # Loop through handlers to find the FileHandler
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename


def get_timestamp_format() -> str:
    # NOTE: Some users have complained that Cognee crashes when trying to get microsecond value
    #       Added handler to not use microseconds if users can't access it
    logger = structlog.get_logger()
    try:
        # We call datetime.now() here to test if microseconds are supported.
        # If they are not supported a ValueError will be raised
        datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        return "%Y-%m-%dT%H:%M:%S.%f"
    except Exception as e:
        logger.debug(f"Exception caught: {e}")
        logger.debug(
            "Could not use microseconds for the logging timestamp, defaulting to use hours minutes and seconds only"
        )
        # We call datetime.now() here to test if won't break.
        datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        # We return the timestamp format without microseconds as they are not supported
        return "%Y-%m-%dT%H:%M:%S"
