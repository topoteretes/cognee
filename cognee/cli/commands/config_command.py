import argparse
import json
from typing import Optional, Any

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class ConfigCommand(SupportsCliCommand):
    command = "config"
    help_string = "Manage cognee configuration settings"
    docs_url = DEFAULT_DOCS_URL
    description = """
The `cognee config` command allows you to view and modify configuration settings.

You can:
- View all current configuration settings
- Get specific configuration values  
- Set configuration values
- Reset configuration to defaults

Configuration changes will affect how cognee processes and stores data.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="config_action", help="Configuration actions")

        # Get command
        get_parser = subparsers.add_parser("get", help="Get configuration value(s)")
        get_parser.add_argument(
            "key", nargs="?", help="Configuration key to retrieve (shows all if not specified)"
        )

        # Set command
        set_parser = subparsers.add_parser("set", help="Set configuration value")
        set_parser.add_argument("key", help="Configuration key to set")
        set_parser.add_argument("value", help="Configuration value to set")

        # List command
        subparsers.add_parser("list", help="List all configuration keys")

        # Reset command
        reset_parser = subparsers.add_parser("reset", help="Reset configuration to defaults")
        reset_parser.add_argument(
            "--force", "-f", action="store_true", help="Skip confirmation prompt"
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            # Import cognee here to avoid circular imports
            import cognee

            if not hasattr(args, "config_action") or args.config_action is None:
                fmt.error("Please specify a config action: get, set, list, or reset")
                return

            if args.config_action == "get":
                self._handle_get(args)
            elif args.config_action == "set":
                self._handle_set(args)
            elif args.config_action == "list":
                self._handle_list(args)
            elif args.config_action == "reset":
                self._handle_reset(args)
            else:
                fmt.error(f"Unknown config action: {args.config_action}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1)
            raise CliCommandException(f"Error managing configuration: {str(e)}", error_code=1)

    def _handle_get(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            if args.key:
                # Get specific key
                try:
                    value = cognee.config.get(args.key)
                    fmt.echo(f"{args.key}: {value}")
                except Exception:
                    fmt.error(f"Configuration key '{args.key}' not found")
            else:
                # Get all configuration
                try:
                    config_dict = (
                        cognee.config.get_all() if hasattr(cognee.config, "get_all") else {}
                    )
                    if config_dict:
                        fmt.echo("Current configuration:")
                        for key, value in config_dict.items():
                            fmt.echo(f"  {key}: {value}")
                    else:
                        fmt.echo("No configuration settings found")
                except Exception:
                    fmt.note("Configuration viewing not fully implemented yet")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to get configuration: {str(e)}")

    def _handle_set(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            # Try to parse value as JSON, otherwise treat as string
            try:
                value = json.loads(args.value)
            except json.JSONDecodeError:
                value = args.value

            try:
                cognee.config.set(args.key, value)
                fmt.success(f"Set {args.key} = {value}")
            except Exception:
                fmt.error(f"Failed to set configuration key '{args.key}'")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to set configuration: {str(e)}")

    def _handle_list(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            # This would need to be implemented in cognee.config
            fmt.note("Available configuration keys:")
            fmt.echo("  LLM_MODEL")
            fmt.echo("  VECTOR_DB_URL")
            fmt.echo("  GRAPH_DB_URL")
            fmt.echo("  (Use 'cognee config get' to see current values)")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to list configuration: {str(e)}")

    def _handle_reset(self, args: argparse.Namespace) -> None:
        try:
            if not args.force:
                if not fmt.confirm("Reset all configuration to defaults?"):
                    fmt.echo("Reset cancelled.")
                    return

            fmt.note("Configuration reset not fully implemented yet")
            fmt.echo("This would reset all settings to their default values")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to reset configuration: {str(e)}")
