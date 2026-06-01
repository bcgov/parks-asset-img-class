"""Schema loader for the 12-attribute x 5-asset-type matrix.

Every modelling pipeline in `src/models/*` reads this schema instead of
hard-coding column names or vocabularies.  Keeping the schema in YAML
(at `configs/schema.yaml`) makes it easy for the partner to review and
makes the leaderboard renderer attribute-agnostic.

The single entry point is `load_schema()`, which returns a `Schema`
dataclass with a few convenience methods.

Example
-------
>>> schema = load_schema()
>>> schema.attributes_for_asset_type("stairs")
['attr_fall_height', 'attr_has_pedestrian_railing',
 'attr_material_frame,_tank,_body', 'attr_number_of_steps',
 'attr_structure_material', 'attr_structure_position']
>>> schema.kind_of("attr_decking_material")
<AttributeKind.CATEGORICAL: 'categorical'>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / "schema.yaml"


class AttributeKind(str, Enum):
    """Maps to the metric family used to score an attribute."""

    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    COUNT = "count"
    ORDINAL_BIN = "ordinal_bin"


@dataclass(frozen=True)
class AssetType:
    """One row in the asset-type matrix."""

    id: str
    profile_name: str


@dataclass(frozen=True)
class Attribute:
    """One attribute column of the prediction CSV."""

    column: str
    display_name: str
    kind: AttributeKind
    asset_types: tuple[str, ...]
    values: tuple[str, ...] = ()
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    bin_column: str | None = None
    measured_column: str | None = None
    units: str | None = None

    def normalise_label(self, raw: Any) -> str | None:
        """Return the canonical label for a raw CSV value, or None."""
        if raw is None:
            return None
        s = str(raw).strip()
        if not s or s.lower() in {"nan", "none", "null", "tbd", "unknown"}:
            return None
        if self.kind not in {AttributeKind.CATEGORICAL, AttributeKind.BOOLEAN}:
            return s
        for canonical, alias_list in self.aliases.items():
            if s == canonical or s in alias_list:
                return canonical
        return s


@dataclass(frozen=True)
class Schema:
    """The full schema: asset types + attributes + ordinal bins."""

    asset_types: dict[str, AssetType]
    attributes: dict[str, Attribute]
    ordinal_bins: dict[str, tuple[str, ...]]
    tasks: dict[str, dict[str, Any]]

    def attribute_columns(self) -> list[str]:
        return list(self.attributes.keys())

    def attributes_for_asset_type(self, asset_type_id: str) -> list[str]:
        return sorted(
            col
            for col, attr in self.attributes.items()
            if asset_type_id in attr.asset_types
        )

    def kind_of(self, column: str) -> AttributeKind:
        return self.attributes[column].kind

    def asset_type_for_profile_name(self, profile_name: str) -> str | None:
        for type_id, at in self.asset_types.items():
            if at.profile_name == profile_name:
                return type_id
        return None


def _parse_attribute(column: str, raw: dict[str, Any]) -> Attribute:
    aliases_raw = raw.get("aliases", {}) or {}
    aliases = {k: tuple(v) for k, v in aliases_raw.items()}
    return Attribute(
        column=column,
        display_name=raw["display_name"],
        kind=AttributeKind(raw["kind"]),
        asset_types=tuple(raw.get("asset_types", [])),
        values=tuple(raw.get("values", [])),
        aliases=aliases,
        bin_column=raw.get("bin_column"),
        measured_column=raw.get("measured_column"),
        units=raw.get("units"),
    )


def load_schema(path: str | Path | None = None) -> Schema:
    """Load and validate the project schema YAML."""
    schema_path = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    with schema_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    asset_types = {
        type_id: AssetType(id=type_id, profile_name=meta["profile_name"])
        for type_id, meta in raw["asset_types"].items()
    }

    attributes = {
        column: _parse_attribute(column, meta)
        for column, meta in raw["attributes"].items()
    }

    ordinal_bins_raw = raw.get("ordinal_bins", {}) or {}
    ordinal_bins = {
        bin_col: tuple(meta["values"]) for bin_col, meta in ordinal_bins_raw.items()
    }

    return Schema(
        asset_types=asset_types,
        attributes=attributes,
        ordinal_bins=ordinal_bins,
        tasks=raw.get("tasks", {}) or {},
    )
