import sys
import logging
import structlog

# Export common log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Track if logging has been configured
_is_configured = False


def get_logger(name=None, level=INFO):
    """Get a configured structlog logger.

    Args:
        name: Logger name (default: None, uses __name__)
        level: Logging level (default: INFO)

    Returns:
        A configured structlog logger instance
    """
    global _is_configured
    if not _is_configured:
        setup_logging(level)
        _is_configured = True

    return structlog.get_logger(name if name else __name__)


def setup_logging(log_level=INFO, name=None):
    """Sets up the logging configuration with structlog integration.

    Args:
        log_level: The logging level to use (default: INFO)
        name: Optional logger name (default: None, uses __name__)

    Returns:
        A configured structlog logger instance
    """

    def exception_handler(logger, method_name, event_dict):
        """Custom processor to handle uncaught exceptions."""
        # Check if there's an exc_info that needs to be processed
        if event_dict.get("exc_info"):
            # If it's already a tuple, use it directly
            if isinstance(event_dict["exc_info"], tuple):
                exc_type, exc_value, tb = event_dict["exc_info"]
            else:
                exc_type, exc_value, tb = sys.exc_info()

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
            structlog.processors.TimeStamper(fmt="iso"),
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

    # Create formatter for standard library logging
    formatter = structlog.stdlib.ProcessorFormatter(
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

    # Setup handler with newlines
    class NewlineStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                stream.write("\n" + msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)

    # Use our custom handler
    stream_handler = NewlineStreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(log_level)

    # Return a configured logger
    return structlog.get_logger(name if name else __name__)
