"""Plugin system — loads models from modelcard.yaml + run.py."""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console

console = Console()

CARDS_DIR = Path(__file__).parent.parent / "models" / "cards"


# --- ModelCard dataclass ---

@dataclass
class ModelSource:
    type: str  # "huggingface", "github"
    repo: str
    branch: str = "main"


@dataclass
class ModelHardware:
    min_ram_gb: int = 2
    recommended_ram_gb: int = 8
    min_vram_gb: int = 0
    recommended_vram_gb: int = 0
    cpu_fallback: bool = True


@dataclass
class ModelVariant:
    name: str
    filename: str
    size_mb: int
    description: str = ""


@dataclass
class ModelCheckpoint:
    supported: bool = False
    every: int = 100  # save every N frames


@dataclass
class ModelCard:
    name: str
    description: str
    version: str
    task: str
    author: str
    source: ModelSource
    hardware: ModelHardware
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    variants: dict[str, ModelVariant] = field(default_factory=dict)
    dependencies: dict = field(default_factory=dict)
    checkpoint: ModelCheckpoint = field(default_factory=ModelCheckpoint)

    @property
    def default_variant(self) -> ModelVariant:
        if "default" in self.variants:
            return self.variants["default"]
        return next(iter(self.variants.values()))

    def get_variant(self, variant_name: str | None) -> ModelVariant:
        if variant_name is None or variant_name == "default":
            return self.default_variant
        if variant_name not in self.variants:
            available = ", ".join(self.variants.keys())
            raise KeyError(
                f"Variant '{variant_name}' not found for {self.name}. "
                f"Available: {available}"
            )
        return self.variants[variant_name]

    @property
    def input_types(self) -> list[str]:
        return [i["type"] for i in self.inputs]

    @property
    def output_types(self) -> list[str]:
        return [o["type"] for o in self.outputs]

    @property
    def huggingface_repo(self) -> str:
        return self.source.repo


# --- Loading ---

def _parse_card(data: dict) -> ModelCard:
    """Parse raw YAML dict into a ModelCard."""
    source_data = data.get("source", {})
    source = ModelSource(
        type=source_data.get("type", "huggingface"),
        repo=source_data.get("repo", ""),
        branch=source_data.get("branch", "main"),
    )

    hw_data = data.get("hardware", {})
    hardware = ModelHardware(
        min_ram_gb=hw_data.get("min_ram_gb", 2),
        recommended_ram_gb=hw_data.get("recommended_ram_gb", 8),
        min_vram_gb=hw_data.get("min_vram_gb", 0),
        recommended_vram_gb=hw_data.get("recommended_vram_gb", 0),
        cpu_fallback=hw_data.get("cpu_fallback", True),
    )

    variants = {}
    for vname, vdata in data.get("variants", {}).items():
        variants[vname] = ModelVariant(
            name=vname,
            filename=vdata.get("filename", ""),
            size_mb=vdata.get("size_mb", 0),
            description=vdata.get("description", ""),
        )

    ckpt_data = data.get("checkpoint", {})
    checkpoint = ModelCheckpoint(
        supported=ckpt_data.get("supported", False),
        every=ckpt_data.get("every", 100),
    )

    return ModelCard(
        name=data["name"],
        description=data.get("description", ""),
        version=data.get("version", "0.1.0"),
        task=data.get("task", ""),
        author=data.get("author", ""),
        source=source,
        hardware=hardware,
        inputs=data.get("inputs", []),
        outputs=data.get("outputs", []),
        variants=variants,
        dependencies=data.get("dependencies", {}),
        checkpoint=checkpoint,
    )


# --- PluginLoader ---

class PluginLoader:
    """Loads models from modelcard.yaml + run.py in cards/ directory."""

    def __init__(self, cards_dir: Path = CARDS_DIR):
        self.cards_dir = cards_dir

    def scan_models(self) -> list[str]:
        """Return names of all available models."""
        if not self.cards_dir.exists():
            return []
        return sorted(
            d.name for d in self.cards_dir.iterdir()
            if d.is_dir() and (d / "modelcard.yaml").exists()
        )

    def load_card(self, name: str) -> ModelCard:
        """Load a model's card. Raises KeyError if not found."""
        card_path = self.cards_dir / name / "modelcard.yaml"
        if not card_path.exists():
            available = self.scan_models()
            raise KeyError(
                f"Model '{name}' not found. Available: {', '.join(available)}"
            )
        data = yaml.safe_load(card_path.read_text())
        return _parse_card(data)

    def list_cards(self) -> list[ModelCard]:
        """Load all model cards."""
        cards = []
        for name in self.scan_models():
            try:
                cards.append(self.load_card(name))
            except Exception:
                continue
        return cards

    def load_runner(self, name: str):
        """Import a model's run.py module. Returns the module with setup() and run()."""
        run_path = self.cards_dir / name / "run.py"
        if not run_path.exists():
            raise FileNotFoundError(f"No run.py found for model '{name}'")

        spec = importlib.util.spec_from_file_location(f"pixo.models.cards.{name}.run", run_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, "setup") or not hasattr(mod, "run"):
            raise ValueError(f"run.py for '{name}' must define setup() and run() functions")

        return mod

    def has_runner(self, name: str) -> bool:
        """Check if a model has a run.py file."""
        return (self.cards_dir / name / "run.py").exists()


# Convenience: default loader instance
loader = PluginLoader()
