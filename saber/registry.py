"""Discovery of the matcher functions and the taxonomy used to select them.

A *structure* here is one entry of ``LABEL_TRANSLATIONS`` in
:mod:`saber.label_translation` that has a corresponding matcher function in
``saber/matchers``. Its identifier, e.g. ``a5d2_1_A2``, encodes everything the
selection API needs:

===============  ===========================================  ==================
Part             Meaning                                      Example
===============  ===========================================  ==================
first two chars  broad group (first-level ``LEVEL_``           ``a5`` (Pronomes)
                 ``DESCRIPTIONS`` key)
up to first "_"  category (second-level key)                  ``a5d2``
                                                              (Demonstrativos)
last two chars   CEFR level at which it is taught             ``A2``
before the last  matcher function name                        ``a5d2_1``
underscore
===============  ===========================================  ==================
"""

import importlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence, Union

from unidecode import unidecode

from saber import _compat
from saber.label_translation import LABEL_TRANSLATIONS, LEVEL_DESCRIPTIONS

logger = logging.getLogger(__name__)

#: Matcher functions are named after their structure minus the CEFR suffix.
MATCHER_NAME_RE = re.compile(r"^[ab]\d+d\d+_\d+$")

MATCHERS_DIR = Path(__file__).resolve().parent / "matchers"

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

Selector = Union[str, Iterable[str], None]


@dataclass(frozen=True)
class Structure:
    """One selectable grammatical structure."""

    structure: str
    """Full identifier, e.g. ``a5d2_1_A2``."""

    matcher: str
    """Name of the matcher function, e.g. ``a5d2_1``."""

    func: Callable
    """The matcher function itself."""

    level: str
    """CEFR level, one of ``A1``-``C2``."""

    group: str
    """Broad group key, e.g. ``a5``."""

    group_name: str
    """Portuguese name of the broad group, e.g. ``Pronomes``."""

    category: str
    """Category key, e.g. ``a5d2``."""

    category_name: str
    """Portuguese name of the category, e.g. ``Demonstrativos``."""

    description: str
    """Portuguese description of the structure."""

    requires_spacy: bool
    """True if the matcher must run on the spaCy doc rather than the Stanza one."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def load_matchers():
    """Import every module in ``saber/matchers`` and return ``{name: function}``.

    Loading the matchers also loads the NLP pipelines, because most matcher
    modules import them at module level. This is cached, so it happens once per
    process.
    """
    # The matchers import their dependencies by bare name; make those resolve
    # before importing any of them.
    _compat.install()

    functions = {}
    origins = {}
    for path in sorted(MATCHERS_DIR.glob("*.py")):
        if path.name.startswith("__"):
            continue
        module_name = f"saber.matchers.{path.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            logger.exception("Could not import matcher module %s", module_name)
            continue
        for attr_name in dir(module):
            if not MATCHER_NAME_RE.match(attr_name):
                continue
            attr = getattr(module, attr_name)
            if not callable(attr):
                continue
            if attr_name in functions and origins[attr_name] != path.stem:
                logger.warning(
                    "Matcher %s is defined in both %s and %s; using %s",
                    attr_name, origins[attr_name], path.stem, path.stem,
                )
            functions[attr_name] = attr
            origins[attr_name] = path.stem

    if not functions:
        raise RuntimeError(
            f"No matcher functions found in {MATCHERS_DIR}. Expected modules "
            "defining functions named like 'a5d2_1'."
        )
    return functions


@lru_cache(maxsize=1)
def all_structures():
    """Every structure that has both a label and a matcher function.

    Structures are returned in the order they are declared in
    ``LABEL_TRANSLATIONS``, which is curated taxonomically (nouns, adjectives,
    verbs, ...). That order also determines the column order of the feature
    table. Labels commented out in :mod:`saber.label_translation` are absent, so
    that file is the switch for enabling and disabling structures.
    """
    functions = load_matchers()
    resolved = []
    missing = []
    for label, description in LABEL_TRANSLATIONS.items():
        matcher_name = label.rsplit("_", 1)[0]
        func = functions.get(matcher_name)
        if func is None:
            missing.append(label)
            continue
        category = label.split("_", 1)[0]
        group = category[:2]
        resolved.append(
            Structure(
                structure=label,
                matcher=matcher_name,
                func=func,
                level=label.rsplit("_", 1)[-1],
                group=group,
                group_name=LEVEL_DESCRIPTIONS.get(group, group),
                category=category,
                category_name=LEVEL_DESCRIPTIONS.get(category, category),
                description=description,
                requires_spacy=bool(getattr(func, "REQUIRES_SPACY", False)),
            )
        )

    if missing:
        logger.warning(
            "%d labelled structure(s) have no matcher function and are "
            "unavailable: %s", len(missing), ", ".join(missing),
        )
    return tuple(resolved)


def missing_labels():
    """Labels declared in ``LABEL_TRANSLATIONS`` with no matcher function."""
    available = {s.structure for s in all_structures()}
    return tuple(label for label in LABEL_TRANSLATIONS if label not in available)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def _normalize(value):
    """Casefold and strip diacritics so ``"Advérbios"`` matches ``"adverbios"``."""
    return unidecode(str(value)).strip().lower()


def _as_list(value: Selector):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _key_lookup(keys):
    """Map normalized keys *and* normalized display names to sets of keys."""
    lookup = {}
    for key in keys:
        lookup.setdefault(_normalize(key), set()).add(key)
        name = LEVEL_DESCRIPTIONS.get(key)
        if name:
            # Second-level names are not unique -- "Demonstrativos" is both
            # a5d2 and a6d2 -- so a name may legitimately select several keys.
            lookup.setdefault(_normalize(name), set()).add(key)
    return lookup


def _resolve_keys(values, keys, kind):
    lookup = _key_lookup(keys)
    resolved = set()
    for value in values:
        found = lookup.get(_normalize(value))
        if not found:
            raise ValueError(
                f"Unknown {kind} {value!r}. Valid {kind} keys: "
                f"{', '.join(sorted(keys))}. Names are accepted too; see "
                "saber.list_structures()."
            )
        resolved |= found
    return resolved


def _resolve_levels(values, available):
    resolved = set()
    for value in values:
        level = str(value).strip().upper()
        if level not in available:
            raise ValueError(
                f"Unknown CEFR level {value!r}. Valid levels: "
                f"{', '.join(sorted(available))}."
            )
        resolved.add(level)
    return resolved


def _resolve_structures(values, structures, kind):
    by_id = {}
    for structure in structures:
        by_id.setdefault(_normalize(structure.structure), set()).add(structure.structure)
        by_id.setdefault(_normalize(structure.matcher), set()).add(structure.structure)
    resolved = set()
    for value in values:
        found = by_id.get(_normalize(value))
        if not found:
            raise ValueError(
                f"Unknown {kind} {value!r}. Expected a structure id such as "
                "'a5d2_1_A2' or a matcher name such as 'a5d2_1'; see "
                "saber.list_structures()."
            )
        resolved |= found
    return resolved


def resolve(
    groups: Selector = None,
    categories: Selector = None,
    levels: Selector = None,
    structures: Selector = None,
    exclude: Selector = None,
) -> Sequence[Structure]:
    """Select structures by group, category, CEFR level and/or identifier.

    Values within one argument are OR-ed; the arguments are AND-ed. Every
    argument accepts a single string or an iterable of strings, and matching is
    case- and diacritic-insensitive.

    Args:
        groups: Broad groups, as first-level ``LEVEL_DESCRIPTIONS`` keys
            (``"a5"``) or names (``"Pronomes"``).
        categories: Categories, as second-level keys (``"a5d2"``) or names
            (``"Demonstrativos"``). A name that several categories share
            selects all of them.
        levels: CEFR levels, ``"A1"`` to ``"C2"``.
        structures: Individual structures, by id (``"a5d2_1_A2"``) or matcher
            name (``"a5d2_1"``).
        exclude: Structures to drop, accepted in the same forms as
            ``structures``. Applied last.

    Returns:
        The matching structures, in ``LABEL_TRANSLATIONS`` declaration order.

    Raises:
        ValueError: If a value matches nothing, or if the combination of filters
            selects no structure at all.
    """
    catalogue = all_structures()

    group_keys = [k for k in LEVEL_DESCRIPTIONS if len(k) == 2]
    category_keys = [k for k in LEVEL_DESCRIPTIONS if len(k) > 2]
    available_levels = {s.level for s in catalogue}

    wanted_groups = _as_list(groups)
    wanted_categories = _as_list(categories)
    wanted_levels = _as_list(levels)
    wanted_structures = _as_list(structures)
    unwanted = _as_list(exclude)

    group_filter = _resolve_keys(wanted_groups, group_keys, "group") if wanted_groups else None
    category_filter = (
        _resolve_keys(wanted_categories, category_keys, "category")
        if wanted_categories else None
    )
    level_filter = _resolve_levels(wanted_levels, available_levels) if wanted_levels else None
    structure_filter = (
        _resolve_structures(wanted_structures, catalogue, "structure")
        if wanted_structures else None
    )
    exclude_filter = (
        _resolve_structures(unwanted, catalogue, "excluded structure")
        if unwanted else set()
    )

    selected = [
        s for s in catalogue
        if (group_filter is None or s.group in group_filter)
        and (category_filter is None or s.category in category_filter)
        and (level_filter is None or s.level in level_filter)
        and (structure_filter is None or s.structure in structure_filter)
        and s.structure not in exclude_filter
    ]

    if not selected:
        raise ValueError(
            "No structures match the given filters. Use "
            "saber.list_structures() to see what is available."
        )
    return selected


LIST_COLUMNS = (
    "structure", "matcher", "level", "group", "group_name",
    "category", "category_name", "description",
)


def list_structures(
    groups: Selector = None,
    categories: Selector = None,
    levels: Selector = None,
    structures: Selector = None,
    exclude: Selector = None,
):
    """Return the selectable structures as a :class:`pandas.DataFrame`.

    Takes the same filters as :func:`resolve`. Useful for discovering the valid
    values of ``groups``, ``categories`` and ``levels``.
    """
    import pandas as pd

    selected = resolve(
        groups=groups, categories=categories, levels=levels,
        structures=structures, exclude=exclude,
    )
    rows = [
        {column: getattr(structure, column) for column in LIST_COLUMNS}
        for structure in selected
    ]
    return pd.DataFrame(rows, columns=list(LIST_COLUMNS))
