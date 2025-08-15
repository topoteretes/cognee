from typing import Protocol, Optional
import argparse


class SupportsCliCommand(Protocol):
    """Protocol for defining one cognee cli command"""

    command: str
    """name of the command"""
    help_string: str
    """the help string for argparse"""
    description: Optional[str]
    """the more detailed description for argparse, may include markdown for the docs"""
    docs_url: Optional[str]
    """the default docs url to be printed in case of an exception"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configures the parser for the given argument"""
        ...

    def execute(self, args: argparse.Namespace) -> None:
        """Executes the command with the given arguments"""
        ...
