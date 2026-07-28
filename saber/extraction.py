"""Running the matchers over documents and shaping the results into tables."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from saber import progress as progress_module
from saber import registry, sources
from saber.registry import Selector, Structure

logger = logging.getLogger(__name__)

MODES = ("spans", "features")
FEATURE_KINDS = ("count", "normalized", "presence")
ON_ERROR = ("warn", "raise", "ignore")

#: Suffix appended to a structure id for each kind of feature.
FEATURE_SUFFIXES = {"count": "_count", "normalized": "_norm", "presence": "_present"}

SPAN_COLUMNS = (
    "document", "path", "structure", "matcher", "level",
    "group", "group_name", "category", "category_name", "description",
    "text", "reconstructed_text",
    "token_start", "token_end", "char_start", "char_end", "pipeline",
)

FEATURE_META_COLUMNS = ("document", "path", "n_tokens")


@dataclass(frozen=True)
class Match:
    """One occurrence of a structure in a document."""

    structure: Structure
    text: str
    """The matched substring, sliced out of the preprocessed text."""

    reconstructed_text: str
    """The matcher's own rendering of the match (see ``reconstruct_text``)."""

    token_start: int
    token_end: int
    char_start: int
    char_end: int


@dataclass
class Analysis:
    """The result of running a set of matchers over one document."""

    document: str
    path: Optional[Path]
    text: str
    """The preprocessed text the character offsets refer to."""

    n_tokens: int
    """Tokens that are neither whitespace nor punctuation."""

    matches: List[Match] = field(default_factory=list)


def _handle_error(message, exc, on_error):
    if on_error == "raise":
        raise exc
    if on_error == "warn":
        logger.warning("%s: %s: %s", message, type(exc).__name__, exc)


def analyze(text, structures: Sequence[Structure], *, document="text", path=None,
            on_error="warn") -> Analysis:
    """Run ``structures`` over ``text`` and return the matches with offsets.

    Character offsets are relative to :attr:`Analysis.text`, the preprocessed
    text, so ``analysis.text[m.char_start:m.char_end] == m.text`` always holds.
    """
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {', '.join(ON_ERROR)}, not {on_error!r}")

    from saber import nlp  # imported here so the pipelines load lazily

    cleaned = nlp.preprocess_text(text)
    doc_stanza = nlp.nlp_stanza(cleaned)
    # Must happen before any matcher runs: several matchers call
    # nlp_stanza.pipe() internally to build their PhraseMatcher patterns, and
    # that would overwrite the single cached Stanza doc this mapping comes from.
    char_spans = nlp.stanza_char_spans(doc_stanza, nlp.nlp_stanza.tokenizer.snlp.last_doc)
    doc_spacy = nlp.nlp_small(cleaned)

    analysis = Analysis(
        document=document,
        path=path,
        text=cleaned,
        n_tokens=nlp.count_tokens(doc_stanza),
    )

    for structure in structures:
        doc = doc_spacy if structure.requires_spacy else doc_stanza
        try:
            raw_matches = structure.func(doc)
        except Exception as exc:  # a single broken matcher must not stop the run
            _handle_error(f"Matcher {structure.matcher} failed on {document}", exc, on_error)
            continue

        for raw in raw_matches or ():
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                logger.warning(
                    "Matcher %s returned an unexpected match shape on %s: %r",
                    structure.matcher, document, raw,
                )
                continue
            # The label the matcher reports is ignored in favour of the registry
            # id we selected it by; the two are one-to-one by convention.
            _label, reconstructed, token_start, token_end = raw[:4]

            if not isinstance(token_start, int) or not isinstance(token_end, int):
                logger.warning(
                    "Matcher %s returned non-integer token indices on %s: %r",
                    structure.matcher, document, raw,
                )
                continue

            # One matcher reports a single token as (i, i) rather than (i, i+1).
            if token_end <= token_start:
                token_end = token_start + 1
            if token_start < 0 or token_start >= len(doc):
                continue
            token_end = min(token_end, len(doc))

            if structure.requires_spacy:
                # nlp_small keeps original character offsets, so use them directly.
                char_start = doc[token_start].idx
                char_end = doc[token_end - 1].idx + len(doc[token_end - 1])
            else:
                start_span = char_spans[token_start]
                end_span = char_spans[token_end - 1]
                if not (start_span and end_span):
                    continue
                char_start, char_end = start_span[0], end_span[1]

            if char_start is None or char_end is None or char_end <= char_start:
                continue

            analysis.matches.append(
                Match(
                    structure=structure,
                    text=cleaned[char_start:char_end],
                    reconstructed_text=reconstructed,
                    token_start=token_start,
                    token_end=token_end,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

    return analysis


def analyze_documents(documents, structures: Sequence[Structure], *,
                      on_error="warn", progress="auto"):
    """Run ``structures`` over every :class:`~saber.sources.Document`."""
    analyses = []
    total = len(documents)
    with progress_module.bar(total, progress=progress) as bar:
        for index, document in enumerate(documents, start=1):
            logger.info("Analysing %s (%d/%d)", document.name, index, total)
            bar.describe(document.name)
            analyses.append(
                analyze(
                    document.text,
                    structures,
                    document=document.name,
                    path=document.path,
                    on_error=on_error,
                )
            )
            bar.update()
    return analyses


def spans_table(analyses):
    """One row per match, sorted by document and position."""
    import pandas as pd

    rows = []
    for analysis in analyses:
        for match in analysis.matches:
            structure = match.structure
            rows.append({
                "document": analysis.document,
                "path": str(analysis.path) if analysis.path else "",
                "structure": structure.structure,
                "matcher": structure.matcher,
                "level": structure.level,
                "group": structure.group,
                "group_name": structure.group_name,
                "category": structure.category,
                "category_name": structure.category_name,
                "description": structure.description,
                "text": match.text,
                "reconstructed_text": match.reconstructed_text,
                "token_start": match.token_start,
                "token_end": match.token_end,
                "char_start": match.char_start,
                "char_end": match.char_end,
                "pipeline": "spacy" if structure.requires_spacy else "stanza",
            })

    frame = pd.DataFrame(rows, columns=list(SPAN_COLUMNS))
    if not frame.empty:
        frame = frame.sort_values(
            ["document", "char_start", "char_end", "structure"], kind="stable"
        ).reset_index(drop=True)
    return frame


def features_table(analyses, structures: Sequence[Structure],
                   features=FEATURE_KINDS):
    """One row per document, with the requested feature columns per structure.

    Columns are emitted for every selected structure, including those that never
    matched, so feature matrices line up across documents and runs.
    """
    import pandas as pd

    kinds = [features] if isinstance(features, str) else list(features)
    unknown = [k for k in kinds if k not in FEATURE_KINDS]
    if unknown:
        raise ValueError(
            f"Unknown feature kind(s) {', '.join(map(repr, unknown))}. "
            f"Valid kinds: {', '.join(FEATURE_KINDS)}."
        )
    if not kinds:
        raise ValueError(f"features must name at least one of {', '.join(FEATURE_KINDS)}.")

    columns = list(FEATURE_META_COLUMNS)
    for structure in structures:
        for kind in kinds:
            columns.append(f"{structure.structure}{FEATURE_SUFFIXES[kind]}")

    rows = []
    for analysis in analyses:
        counts = Counter(match.structure.structure for match in analysis.matches)
        row = {
            "document": analysis.document,
            "path": str(analysis.path) if analysis.path else "",
            "n_tokens": analysis.n_tokens,
        }
        for structure in structures:
            count = counts.get(structure.structure, 0)
            for kind in kinds:
                column = f"{structure.structure}{FEATURE_SUFFIXES[kind]}"
                if kind == "count":
                    row[column] = count
                elif kind == "normalized":
                    row[column] = (
                        count / analysis.n_tokens * 100 if analysis.n_tokens else 0.0
                    )
                else:
                    row[column] = int(count > 0)
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def extract(
    source,
    mode="spans",
    *,
    groups: Selector = None,
    categories: Selector = None,
    levels: Selector = None,
    structures: Selector = None,
    exclude: Selector = None,
    features=FEATURE_KINDS,
    pattern=None,
    recursive=False,
    encoding="utf-8",
    source_type="auto",
    on_error="warn",
    progress="auto",
):
    """Identify grammatical structures in Portuguese text.

    Args:
        source: A string of text, a :class:`~pathlib.Path` to a text file, a
            :class:`~pathlib.Path` to a folder of them, or a sequence mixing
            those. Plain strings are treated as text; use
            ``source_type="path"`` to read them as paths instead.
        mode: ``"spans"`` for one row per identified structure occurrence, with
            its text and offsets; ``"features"`` for one row per document, with
            the frequency, normalized frequency and presence of each structure.
        groups: Restrict to broad groups, by key (``"a5"``) or name
            (``"Pronomes"``).
        categories: Restrict to categories, by key (``"a5d2"``) or name
            (``"Demonstrativos"``).
        levels: Restrict to CEFR levels, ``"A1"`` to ``"C2"``.
        structures: Restrict to individual structures, by id (``"a5d2_1_A2"``)
            or matcher name (``"a5d2_1"``).
        exclude: Structures to leave out, in the same forms as ``structures``.
        features: Which feature kinds to emit in ``"features"`` mode; any of
            ``"count"``, ``"normalized"``, ``"presence"``.
        pattern: Glob used when ``source`` is a folder. ``None`` (the default)
            reads every plain-text file: ``.txt`` files and files with no
            file-type extension, such as ``texto57`` or ``A1.1``.
        recursive: Search folders recursively.
        encoding: Encoding used to read files; falls back to latin-1.
        source_type: ``"auto"``, ``"text"`` or ``"path"``.
        on_error: What to do when an individual matcher raises -- ``"warn"``
            (default), ``"raise"`` or ``"ignore"``.
        progress: Show a progress bar over the documents. ``"auto"`` (default)
            shows one when there is more than one document and stderr is a
            terminal; ``True`` always shows it, ``False`` never does.

    Returns:
        A :class:`pandas.DataFrame`. See ``SPAN_COLUMNS`` for the columns of the
        spans table; the feature table starts with ``document``, ``path`` and
        ``n_tokens``, followed by ``<structure>_count``, ``<structure>_norm``
        (occurrences per 100 tokens) and ``<structure>_present`` (0/1) for each
        selected structure.

    Raises:
        ValueError: For an unknown mode, an unknown filter value, or a filter
            combination that selects no structure.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")

    selected = registry.resolve(
        groups=groups, categories=categories, levels=levels,
        structures=structures, exclude=exclude,
    )
    documents = sources.resolve_sources(
        source, pattern=pattern, recursive=recursive,
        encoding=encoding, source_type=source_type,
    )
    logger.info(
        "Extracting %d structure(s) from %d document(s)", len(selected), len(documents)
    )
    analyses = analyze_documents(
        documents, selected, on_error=on_error, progress=progress
    )

    if mode == "spans":
        return spans_table(analyses)
    return features_table(analyses, selected, features=features)


def extract_spans(source, **kwargs):
    """:func:`extract` with ``mode="spans"``."""
    return extract(source, mode="spans", **kwargs)


def extract_features(source, **kwargs):
    """:func:`extract` with ``mode="features"``."""
    return extract(source, mode="features", **kwargs)
