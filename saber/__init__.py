"""SABER -- identify European Portuguese grammatical structures in text.

SABER (Sistema de Análise e Busca de Estruturas Relevantes) locates the
grammatical structures of the `Referencial Camões
<https://www.instituto-camoes.pt/activity/centro-virtual/referencial-camoes-ple>`_
in Portuguese text and reports each one together with the CEFR level (A1-C2) at
which it is taught.

Two ways to get results::

    import saber

    # One row per occurrence, with its text and character offsets.
    spans = saber.extract("O meu irmão vai ao cinema.", mode="spans")

    # One row per document: raw frequency, frequency per 100 tokens, presence.
    features = saber.extract(Path("corpus/"), mode="features")

Both accept ``groups``, ``categories``, ``levels`` and ``structures`` filters;
:func:`list_structures` shows what those can be set to.

Run :func:`download_models` once before the first extraction.
"""

from saber.extraction import (
    Analysis,
    Match,
    analyze,
    extract,
    extract_features,
    extract_spans,
)
from saber.models import download_models
from saber.registry import (
    Structure,
    all_structures,
    list_structures,
    load_matchers,
    missing_labels,
)

__version__ = "0.1.0"

__all__ = [
    "Analysis",
    "Match",
    "Structure",
    "all_structures",
    "analyze",
    "download_models",
    "extract",
    "extract_features",
    "extract_spans",
    "list_structures",
    "load_matchers",
    "missing_labels",
    "__version__",
]
