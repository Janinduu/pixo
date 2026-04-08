"""Cloud config — stores and loads cloud backend credentials from ~/.pixo/config.yaml."""

from pathlib import Path
from dataclasses import dataclass, field

import yaml


CONFIG_PATH = Path.home() / ".pixo" / "config.yaml"


@dataclass
class KaggleConfig:
    username: str = ""
    api_key: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.username and self.api_key)


@dataclass
class ColabConfig:
    token: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.token)


@dataclass
class CloudConfig:
    kaggle: KaggleConfig = field(default_factory=KaggleConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)

    @property
    def any_configured(self) -> bool:
        return self.kaggle.is_configured or self.colab.is_configured


def load_config() -> CloudConfig:
    """Load cloud config from disk."""
    if not CONFIG_PATH.exists():
        return CloudConfig()

    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    cloud = data.get("cloud", {})

    kaggle_data = cloud.get("kaggle", {})
    colab_data = cloud.get("colab", {})

    return CloudConfig(
        kaggle=KaggleConfig(
            username=kaggle_data.get("username", ""),
            api_key=kaggle_data.get("api_key", ""),
        ),
        colab=ColabConfig(
            token=colab_data.get("token", ""),
        ),
    )


def save_config(config: CloudConfig):
    """Save cloud config to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config to preserve other settings
    existing = {}
    if CONFIG_PATH.exists():
        existing = yaml.safe_load(CONFIG_PATH.read_text()) or {}

    existing["cloud"] = {
        "kaggle": {
            "username": config.kaggle.username,
            "api_key": config.kaggle.api_key,
        },
        "colab": {
            "token": config.colab.token,
        },
    }

    CONFIG_PATH.write_text(yaml.dump(existing, default_flow_style=False))
