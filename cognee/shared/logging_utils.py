import os
import sys
import threading
import logging
import structlog
import traceback
from datetime import datetime
from pathlib import Path

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

# Track if logging has been configured
_is_configured = False

# Create a lock for thread-safe initialization
_setup_lock = threading.Lock()

# Path to logs directory
LOGS_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"))
LOGS_DIR.mkdir(exist_ok=True)  # Create logs dir if it doesn't exist

# Maximum number of log files to keep
MAX_LOG_FILES = 10


class PlainFileHandler(logging.FileHandler):
    """A custom file handler that writes simpler plain text log entries."""

    def emit(self, record):
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
            self.stream.write(f"Error in log handler: {e}\n")
            self.flush()


class LoggerInterface:
    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def critical(self, msg, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass


def get_logger(name=None, level=None) -> LoggerInterface:
    """Get a configured structlog logger.

    Args:
        name: Logger name (default: None, uses __name__)
        level: Logging level (default: None)

    Returns:
        A configured structlog logger instance
    """
    global _is_configured

    # Always first check if logger is already configured to not use threading lock if not necessary
    if not _is_configured:
        # Use threading lock to make sure setup_logging can be called only once
        with _setup_lock:
            # Unfortunately we also need a second check in case lock was entered twice at the same time
            if not _is_configured:
                setup_logging(level)
                _is_configured = True

    return structlog.get_logger(name if name else __name__)


def cleanup_old_logs(logs_dir, max_files):
    """
    Removes old log files, keeping only the most recent ones.

    Args:
        logs_dir: Directory containing log files
        max_files: Maximum number of log files to keep
    """
    try:
        logger = structlog.get_logger()

        # Get all .log files in the directory (excluding README and other files)
        log_files = [f for f in logs_dir.glob("*.log") if f.is_file()]

        # Sort log files by modification time (newest first)
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old files that exceed the maximum
        if len(log_files) > max_files:
            for old_file in log_files[max_files:]:
                try:
                    old_file.unlink()
                    logger.info(f"Deleted old log file: {old_file}")
                except Exception as e:
                    logger.error(f"Failed to delete old log file {old_file}: {e}")

        return True
    except Exception as e:
        logger.error(f"Error cleaning up log files: {e}")
        return False


def setup_logging(log_level=None, name=None):
    """Sets up the logging configuration with structlog integration.

    Args:
        log_level: The logging level to use (default: None, uses INFO)
        name: Optional logger name (default: None, uses __name__)

    Returns:
        A configured structlog logger instance
    """

    log_level = log_level if log_level else log_levels[os.getenv("LOG_LEVEL", "INFO")]

    def exception_handler(logger, method_name, event_dict):
        """Custom processor to handle uncaught exceptions."""
        # Check if there's an exc_info that needs to be processed
        if event_dict.get("exc_info"):
            # If it's already a tuple, use it directly
            if isinstance(event_dict["exc_info"], tuple):
                exc_type, exc_value, tb = event_dict["exc_info"]
            else:
                exc_type, exc_value, tb = sys.exc_info()

            if hasattr(exc_type, __name__):
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
    def handle_exception(exc_type, exc_value, traceback):
        """Handle any uncaught exception."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Let KeyboardInterrupt pass through
            sys.__excepthook__(exc_type, exc_value, traceback)
            return

        logger = structlog.get_logger()
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, traceback),
        )

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
            },
        ),
    )

    # Setup handler with newlines for console output
    class NewlineStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                stream.write("\n" + msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)

    # Use our custom handler for console output
    stream_handler = NewlineStreamHandler(sys.stderr)
    stream_handler.setFormatter(console_formatter)
    stream_handler.setLevel(log_level)

    # Check if we already have a log file path from the environment
    # NOTE: environment variable must be used here as it allows us to
    # log to a single file with a name based on a timestamp in a multiprocess setting.
    # Without it, we would have a separate log file for every process.
    log_file_path = os.environ.get("LOG_FILE_NAME")
    if not log_file_path:
        # Create a new log file name with the cognee start time
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = os.path.join(LOGS_DIR, f"{start_time}.log")
        os.environ["LOG_FILE_NAME"] = log_file_path

    # Create a file handler that uses our custom PlainFileHandler
    file_handler = PlainFileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(DEBUG)

    # Configure root logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

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
    cleanup_old_logs(LOGS_DIR, MAX_LOG_FILES)

    # Return a configured logger
    return structlog.get_logger(name if name else __name__)


def get_log_file_location():
    # Get the root logger
    root_logger = logging.getLogger()

    # Loop through handlers to find the FileHandler
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename


def get_timestamp_format():
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
