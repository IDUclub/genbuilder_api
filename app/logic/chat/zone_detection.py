"""Find the functional zone type in an uploaded blocks file and canonicalize it.

Uploaded files come from anywhere (Urban DB, Prostor exports, QGIS, hand-made),
so the zone type may live in ``zone``, ``functional_zone_type_name``,
``functional_zone_type.name`` or some other attribute, and be spelled as an
Urban DB type (``residential_lowrise``), a Russian label ("Жилая зона") or with
typos. Detection is hybrid:

1. rules — known attributes first, values resolved via an exact dictionary,
   Russian keyword stems and a strict fuzzy match against Urban DB type names;
2. LLM fallback — only for what the rules could not settle (which attribute, or
   which type an unrecognized value means). Guided decoding restricts the answer
   to candidate attributes and to ``ZONE_TYPES`` + ``"none"``, and the answer is
   validated again here, so a misspelled or made-up type never leaks through:
   a value either becomes a known Urban DB type or is reported as unrecognized.

Never raises: an LLM failure degrades to "rules only" and is reported in
``ZoneDetection.llm_error``.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.infrastructure.vllm_chat_client import VLLMChatClient, VLLMChatError

# Urban DB functional_zone_type names a block may be tagged with.
ZONE_TYPES: tuple[str, ...] = (
    "residential",
    "residential_individual",
    "residential_lowrise",
    "residential_midrise",
    "residential_multistorey",
    "business",
    "mixed_use",
    "industrial",
    "transport",
    "special",
    "recreation",
    "agriculture",
    "unknown",
)

ZONE_TYPE_LABELS: dict[str, str] = {
    "residential": "жилая",
    "residential_individual": "ИЖС",
    "residential_lowrise": "малоэтажная жилая",
    "residential_midrise": "среднеэтажная жилая",
    "residential_multistorey": "многоэтажная жилая",
    "business": "общественно-деловая",
    "mixed_use": "многофункциональная",
    "industrial": "промышленная",
    "transport": "транспортная",
    "special": "специального назначения",
    "recreation": "рекреационная",
    "agriculture": "сельскохозяйственная",
    "unknown": "не определена",
}

# Attributes that carry the zone type in known exports, in priority order.
PREFERRED_ATTRIBUTES: tuple[str, ...] = (
    "zone",
    "functional_zone_type_name",
    "functional_zone_type.name",
)

_NONE = "none"
_MIN_COVERAGE = 0.5
_FUZZY_CUTOFF = 0.85
_MAX_LLM_ATTRIBUTES = 15
_MAX_LLM_SAMPLES = 20
_MAX_LLM_VALUES = 50
_MAX_VALUE_LEN = 80
_EMPTY = {"", "none", "nan", "null"}

# Russian stems -> type, most specific first. A value matching stems of
# different (non-residential-subtype) types is ambiguous and stays unresolved.
_STEM_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ижс", "индивидуальн"), "residential_individual"),
    (("малоэтажн",), "residential_lowrise"),
    (("среднеэтажн",), "residential_midrise"),
    (("многоэтажн",), "residential_multistorey"),
    (("многофункциональн", "смешанн"), "mixed_use"),
    (("общественно делов", "делов"), "business"),
    (("промышлен", "производствен"), "industrial"),
    (("транспорт",), "transport"),
    (("специальн", "особого назначения", "режимн"), "special"),
    (("рекреац", "природн", "озелен"), "recreation"),
    (("сельскохоз", "сельхоз"), "agriculture"),
    (("жил",), "residential"),
)
_NOISE_WORDS = {"зона", "зоны", "зон", "территория", "территории", "zone", "type"}


def _normalize(value: str) -> str:
    text = value.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    words = [w for w in text.split() if w not in _NOISE_WORDS]
    return " ".join(words)


_EXACT: dict[str, str] = {_normalize(z): z for z in ZONE_TYPES}
_EXACT.update({_normalize(label): z for z, label in ZONE_TYPE_LABELS.items()})
_EXACT.update(
    {
        "mixed use": "mixed_use",
        "mixeduse": "mixed_use",
        "многоэтажная": "residential_multistorey",
        "среднеэтажная": "residential_midrise",
        "малоэтажная": "residential_lowrise",
    }
)
_FUZZY_KEYS: tuple[str, ...] = tuple(_normalize(z) for z in ZONE_TYPES)


def resolve_zone_value(value: object) -> str | None:
    """Map one raw attribute value to an Urban DB zone type, or None if unsure."""
    if value is None or isinstance(value, (bool, int, float)):
        return None
    norm = _normalize(str(value))
    if not norm:
        return None
    if norm in _EXACT:
        return _EXACT[norm]
    if "нежил" in norm:
        return None
    hits = {zone for stems, zone in _STEM_RULES if any(s in norm for s in stems)}
    if hits:
        subtypes = {h for h in hits if h.startswith("residential_")}
        if hits <= subtypes | {"residential"} and len(subtypes) <= 1:
            return subtypes.pop() if subtypes else "residential"
        return None  # e.g. "жилая и деловая" — ambiguous, not guessed
    scored = sorted(
        ((difflib.SequenceMatcher(None, norm, key).ratio(), key) for key in _FUZZY_KEYS),
        reverse=True,
    )
    best_score, best_key = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= _FUZZY_CUTOFF and best_score - runner_up >= 0.05:
        return _EXACT[best_key]
    return None


def _flatten(props: object) -> dict[str, Any]:
    """Scalar properties, one nested level deep (``functional_zone_type.name``)."""
    if not isinstance(props, dict):
        return {}
    flat: dict[str, Any] = {}
    for key, value in props.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if not isinstance(sub_value, (dict, list)):
                    flat[f"{key}.{sub_key}"] = sub_value
        elif not isinstance(value, list):
            flat[str(key)] = value
    return flat


def _text(value: Any) -> str | None:
    """Non-empty textual value; numbers are never zone types (codes are ambiguous)."""
    if value is None or isinstance(value, (bool, int, float)):
        return None
    text = str(value).strip()
    if text.lower() in _EMPTY:
        return None
    try:
        float(text)
        return None
    except ValueError:
        return text


@dataclass
class ZoneDetection:
    """Outcome of zone detection over an uploaded FeatureCollection."""

    attribute: str | None
    features: list[dict[str, Any]]  # features whose value resolved; ``zone`` set
    mapping: dict[str, str] = field(default_factory=dict)  # raw value -> zone type
    llm_mapping: dict[str, str] = field(default_factory=dict)  # subset set by the LLM
    unrecognized: dict[str, int] = field(default_factory=dict)  # raw value -> count
    missing: int = 0  # features with no value in the attribute
    attributes: list[str] = field(default_factory=list)  # all textual attributes seen
    attribute_by_llm: bool = False
    llm_error: str | None = None

    @property
    def type_counts(self) -> Counter:
        return Counter(f["properties"]["zone"] for f in self.features)


_ATTRIBUTE_PROMPT = (
    "Ты анализируешь GeoJSON-файл с функциональными зонами территории. Тебе дан "
    "список атрибутов (properties) с примерами значений. Выбери ОДИН атрибут, "
    "значения которого обозначают тип функциональной зоны (жилая, "
    "общественно-деловая, многофункциональная, рекреационная, промышленная и т. п.). "
    "Не выбирай идентификаторы, названия конкретных объектов, адреса, числовые "
    "коды и площади. Если подходящего атрибута нет — верни null."
)

_VALUES_PROMPT = (
    "Сопоставь каждое значение атрибута функциональной зоны с типом зоны из "
    "списка: {types}. Значения могут быть на русском или английском и с "
    "опечатками. Выбирай тип, только если значение однозначно его обозначает. "
    "Если значение бессмысленное, неоднозначное или не является типом "
    "функциональной зоны — ставь none. Не угадывай."
)


async def _llm_pick_attribute(
    llm_client: VLLMChatClient,
    samples: dict[str, list[str]],
    model: str | None,
) -> str | None:
    names = list(samples)
    schema = {
        "type": "object",
        "properties": {"attribute": {"type": ["string", "null"], "enum": [*names, None]}},
        "required": ["attribute"],
    }
    listing = "\n".join(
        f"- {name}: " + "; ".join(f"«{v}»" for v in values) for name, values in samples.items()
    )
    raw = await llm_client.complete_json(
        [
            {"role": "system", "content": _ATTRIBUTE_PROMPT},
            {"role": "user", "content": f"Атрибуты и примеры значений:\n{listing}"},
        ],
        schema=schema,
        model=model,
    )
    attribute = raw.get("attribute") if isinstance(raw, dict) else None
    return attribute if attribute in samples else None


async def _llm_map_values(
    llm_client: VLLMChatClient,
    attribute: str,
    values: list[str],
    model: str | None,
) -> dict[str, str]:
    allowed = [*ZONE_TYPES, _NONE]
    schema = {
        "type": "object",
        "properties": {
            "mapping": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "enum": values},
                        "zone": {"type": "string", "enum": allowed},
                    },
                    "required": ["value", "zone"],
                },
            }
        },
        "required": ["mapping"],
    }
    types = ", ".join(f"{z} ({ZONE_TYPE_LABELS[z]})" for z in ZONE_TYPES)
    listing = "\n".join(f"- «{v}»" for v in values)
    raw = await llm_client.complete_json(
        [
            {"role": "system", "content": _VALUES_PROMPT.format(types=types)},
            {"role": "user", "content": f"Атрибут «{attribute}», значения:\n{listing}"},
        ],
        schema=schema,
        model=model,
    )
    items = raw.get("mapping") if isinstance(raw, dict) else None
    mapping: dict[str, str] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        value, zone = item.get("value"), item.get("zone")
        # Guided decoding already restricts both; re-check — nothing outside the
        # enum (or a value we never asked about) is accepted.
        if value in values and zone in ZONE_TYPES:
            mapping.setdefault(value, zone)
    return mapping


async def detect_zones(
    geojson: dict[str, Any],
    llm_client: VLLMChatClient | None,
    *,
    model: str | None = None,
) -> ZoneDetection:
    """Find the zone attribute and resolve every feature to an Urban DB type."""
    features = [f for f in geojson.get("features") or [] if isinstance(f, dict)]
    flat = [_flatten(f.get("properties")) for f in features]

    values_by_attr: dict[str, list[str | None]] = {}
    for props in flat:
        for key in props:
            values_by_attr.setdefault(key, [])
    for key in values_by_attr:
        values_by_attr[key] = [_text(props.get(key)) for props in flat]
    textual = {k: v for k, v in values_by_attr.items() if any(x is not None for x in v)}

    def coverage(attr: str) -> float:
        present = [v for v in textual[attr] if v is not None]
        return sum(resolve_zone_value(v) is not None for v in present) / len(present)

    detection = ZoneDetection(attribute=None, features=[], attributes=sorted(textual))
    if not features:
        return detection

    attribute = next(
        (a for a in PREFERRED_ATTRIBUTES if a in textual and coverage(a) >= _MIN_COVERAGE),
        None,
    )
    if attribute is None and textual:
        best = max(textual, key=coverage)
        if coverage(best) >= _MIN_COVERAGE:
            attribute = best

    if attribute is None and textual and llm_client is not None:
        candidates = sorted(textual, key=lambda a: len(set(textual[a]) - {None}))
        samples = {
            a: [
                v[:_MAX_VALUE_LEN]
                for v, _ in Counter(x for x in textual[a] if x is not None).most_common(
                    _MAX_LLM_SAMPLES
                )
            ]
            for a in candidates[:_MAX_LLM_ATTRIBUTES]
        }
        try:
            attribute = await _llm_pick_attribute(llm_client, samples, model)
            detection.attribute_by_llm = attribute is not None
        except VLLMChatError as exc:
            logger.warning("zone attribute detection via LLM failed: {}", exc)
            detection.llm_error = str(exc)

    if attribute is None:
        return detection
    detection.attribute = attribute

    raw_values = textual[attribute]
    distinct = list(dict.fromkeys(v for v in raw_values if v is not None))
    mapping = {v: z for v in distinct if (z := resolve_zone_value(v)) is not None}

    unresolved = [v for v in distinct if v not in mapping]
    if unresolved and llm_client is not None:
        asked = [v for v in unresolved if len(v) <= _MAX_VALUE_LEN][:_MAX_LLM_VALUES]
        if asked:
            try:
                llm_mapping = await _llm_map_values(llm_client, attribute, asked, model)
                mapping.update(llm_mapping)
                detection.llm_mapping = llm_mapping
            except VLLMChatError as exc:
                logger.warning("zone value mapping via LLM failed: {}", exc)
                detection.llm_error = str(exc)

    unrecognized: Counter = Counter()
    for feature, value in zip(features, raw_values):
        if value is None:
            detection.missing += 1
        elif value in mapping:
            props = dict(feature.get("properties") or {})
            props["zone"] = mapping[value]
            detection.features.append({**feature, "properties": props})
        else:
            unrecognized[value] += 1
    detection.mapping = mapping
    detection.unrecognized = dict(unrecognized.most_common())
    return detection
