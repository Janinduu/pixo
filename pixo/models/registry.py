"""Model registry — thin wrapper around the plugin system for backwards compatibility."""

# Re-export from the new plugin system
from pixo.core.plugin import ModelCard, ModelVariant, loader


def list_models():
    return loader.list_cards()


def get_model(name):
    return loader.load_card(name)


def parse_model_name(model_str):
    if ":" in model_str:
        name, variant = model_str.split(":", 1)
        return name, variant
    return model_str, None
