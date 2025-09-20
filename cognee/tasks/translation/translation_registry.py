from typing import Dict, Type
from .translation_providers_enum import TranslationProvider

_provider_registry: Dict[str, Type[TranslationProvider]] = {}

def register_translation_provider(name: str, provider_cls: Type[TranslationProvider]) -> None:
    """Register a translation provider under a canonical lowercase key."""
    _provider_registry[name.lower()] = provider_cls

def get_available_providers() -> list:
    """Return a sorted list of available provider keys."""
    return sorted(_provider_registry.keys())

def get_provider_class(name: str) -> Type[TranslationProvider]:
    """Get a provider class by name, or raise KeyError if not found."""
    return _provider_registry[name.lower()]

def snapshot_registry() -> Dict[str, Type[TranslationProvider]]:
    """Return a shallow copy snapshot of the provider registry (for tests)."""
    return dict(_provider_registry)

def restore_registry(snapshot: Dict[str, Type[TranslationProvider]]) -> None:
    """Restore the global translation provider registry from a previously captured snapshot."""
    _provider_registry.clear()
    _provider_registry.update(snapshot)

def validate_provider(name: str) -> None:
    """Ensure a provider is registered or raise ValueError."""
    if name.lower() not in _provider_registry:
        raise ValueError(f"Unknown provider: {name}")
