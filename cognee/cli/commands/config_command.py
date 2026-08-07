import argparse
import json
from typing import Optional, Any

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.api.v1.exceptions.exceptions import InvalidConfigAttributeError


class ConfigCommand(SupportsCliCommand):
    command_string = "config"
    help_string = "Manage cognee configuration settings"
    docs_url = DEFAULT_DOCS_URL
    description = """
The `cognee config` command allows you to view and modify configuration settings.

You can:
- View all current configuration settings
- Get specific configuration values
- Set configuration values
- Unset (reset to default) specific configuration values
- Reset all configuration to defaults

Configuration changes will affect how cognee processes and stores data.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="config_action", help="Configuration actions")

        # Get command
        get_parser = subparsers.add_parser("get", help="Get configuration value(s)")
        get_parser.add_argument(
            "key", nargs="?", help="Configuration key to retrieve (shows all if not specified)"
        )
        get_parser.add_argument(
            "--show-secrets",
            action="store_true",
            help="Show secret values (API keys) in plaintext instead of masked",
        )

        # Set command
        set_parser = subparsers.add_parser("set", help="Set configuration value")
        set_parser.add_argument("key", help="Configuration key to set")
        set_parser.add_argument("value", help="Configuration value to set")

        # List command
        subparsers.add_parser("list", help="List all configuration keys")

        # Unset command
        unset_parser = subparsers.add_parser("unset", help="Remove/unset a configuration value")
        unset_parser.add_argument("key", help="Configuration key to unset")
        unset_parser.add_argument(
            "--force", "-f", action="store_true", help="Skip confirmation prompt"
        )

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
                fmt.error("Please specify a config action: get, set, unset, list, or reset")
                return

            if args.config_action == "get":
                self._handle_get(args)
            elif args.config_action == "set":
                self._handle_set(args)
            elif args.config_action == "unset":
                self._handle_unset(args)
            elif args.config_action == "list":
                self._handle_list(args)
            elif args.config_action == "reset":
                self._handle_reset(args)
            else:
                fmt.error(f"Unknown config action: {args.config_action}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(
                f"Error managing configuration: {str(e)}", error_code=1
            ) from e

    def _handle_get(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            reveal_secrets = getattr(args, "show_secrets", False)

            if args.key:
                # Get specific key
                try:
                    value = cognee.config.get(args.key, reveal_secrets=reveal_secrets)
                    fmt.echo(f"{args.key}: {value}")
                except InvalidConfigAttributeError:
                    fmt.error(f"Configuration key '{args.key}' not found")
                    fmt.note("Use 'cognee config get' to see all available keys")
                except Exception:
                    fmt.error(f"Configuration key '{args.key}' not found or retrieval failed")
            else:
                # Get all configuration
                config_dict = cognee.config.get_all(reveal_secrets=reveal_secrets)
                if config_dict:
                    fmt.echo("Current configuration:")
                    for key, value in config_dict.items():
                        fmt.echo(f"  {key}: {value}")
                else:
                    fmt.echo("No configuration settings found")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to get configuration: {str(e)}") from e

    def _handle_set(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            # Try to parse value as JSON, otherwise treat as string
            try:
                value = json.loads(args.value)
            except json.JSONDecodeError:
                value = args.value

            try:
                persist_info = cognee.config.set(args.key, value, persist=True)
                fmt.success(f"Set {args.key} = {value}")
                if persist_info:
                    if persist_info["created"]:
                        fmt.note(f"Created new .env file at {persist_info['path']}")
                    fmt.note(f"Persisted {persist_info['env_var']} to {persist_info['path']}")
            except Exception:
                fmt.error(f"Failed to set configuration key '{args.key}'")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to set configuration: {str(e)}") from e

    def _handle_unset(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            # Confirm unset unless forced
            if not args.force:
                if not fmt.confirm(f"Unset configuration key '{args.key}'?"):
                    fmt.echo("Unset cancelled.")
                    return

            # Since the config system doesn't have explicit unset methods, we
            # map config keys to their default values and reuse the generic
            # setter (persisted, so the reset survives past this process).
            config_key_defaults = {
                # LLM configuration
                "llm_provider": "openai",
                "llm_model": "gpt-5-mini",
                "llm_api_key": "",
                "llm_endpoint": "",
                # Database configuration
                "graph_database_provider": "ladybug",
                "vector_db_provider": "lancedb",
                "vector_db_url": "",
                "vector_db_key": "",
                # Chunking configuration
                "chunk_size": 1500,
                "chunk_overlap": 10,
            }

            if args.key in config_key_defaults:
                default_value = config_key_defaults[args.key]

                try:
                    cognee.config.set(args.key, default_value, persist=True)
                    fmt.success(f"Unset {args.key} (reset to default: {default_value})")
                except Exception as e:
                    fmt.error(f"Failed to unset '{args.key}': {str(e)}")
            else:
                fmt.error(f"Unknown configuration key '{args.key}'")
                fmt.note("Available keys: " + ", ".join(config_key_defaults.keys()))
                fmt.note("Use 'cognee config list' to see all available configuration options")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to unset configuration: {str(e)}") from e

    def _handle_list(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            fmt.note("Available configuration keys:")
            fmt.echo("  llm_provider, llm_model, llm_api_key, llm_endpoint")
            fmt.echo("  graph_database_provider, vector_db_provider")
            fmt.echo("  vector_db_url, vector_db_key")
            fmt.echo("  chunk_size, chunk_overlap")
            fmt.echo("")
            fmt.echo("Commands:")
            fmt.echo("  cognee config get [key]     - View configuration")
            fmt.echo("  cognee config set <key> <value> - Set configuration")
            fmt.echo("  cognee config unset <key>   - Reset key to default")
            fmt.echo("  cognee config reset         - Reset all to defaults")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to list configuration: {str(e)}") from e

    def _handle_reset(self, args: argparse.Namespace) -> None:
        try:
            if not args.force:
                if not fmt.confirm("Reset all configuration to defaults?"):
                    fmt.echo("Reset cancelled.")
                    return

            fmt.note("Configuration reset not fully implemented yet")
            fmt.echo("This would reset all settings to their default values")

        except Exception as e:
            raise CliCommandInnerException(f"Failed to reset configuration: {str(e)}") from e
