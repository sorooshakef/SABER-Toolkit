"""Turning the ``source`` argument into a list of named documents.

``source`` may be a raw string of text, a path to a text file, a path to a
folder of text files, or any mixture of those in a sequence. The distinction
between "a string of text" and "a path" is made by type, so it is never
ambiguous:

* :class:`str` is always treated as text,
* :class:`pathlib.Path` (or any :class:`os.PathLike`) is always treated as a path.

Pass ``source_type="path"`` to read a plain string as a path, or
``source_type="text"`` to force the opposite.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SOURCE_TYPES = ("auto", "text", "path")


@dataclass(frozen=True)
class Document:
    """A single text to analyse."""

    name: str
    """Identifier used in the output tables (file stem, or ``text``/``text_1``...)."""

    path: Optional[Path]
    """Where it was read from, or ``None`` for text passed in directly."""

    text: str


def read_text_file(path, encoding="utf-8"):
    """Read ``path``, falling back to latin-1 if it is not valid ``encoding``."""
    try:
        return Path(path).read_text(encoding=encoding)
    except UnicodeDecodeError:
        logger.warning("%s is not valid %s; falling back to latin-1", path, encoding)
        return Path(path).read_text(encoding="latin-1")


def _is_pathlike(value):
    return isinstance(value, os.PathLike)


def _flatten(source):
    """Yield ``(kind, value)`` pairs, where kind is ``"text"`` or ``"path"``."""
    if isinstance(source, str) or _is_pathlike(source):
        yield source
        return
    try:
        items = list(source)
    except TypeError:
        raise TypeError(
            f"source must be a string, a path, or a sequence of those, "
            f"not {type(source).__name__}"
        ) from None
    for item in items:
        if isinstance(item, str) or _is_pathlike(item):
            yield item
        else:
            raise TypeError(
                f"source sequence items must be strings or paths, "
                f"not {type(item).__name__}"
            )


def resolve_sources(
    source,
    *,
    pattern="*.txt",
    recursive=False,
    encoding="utf-8",
    source_type="auto",
):
    """Expand ``source`` into a list of :class:`Document` objects.

    Args:
        source: A string of text, a :class:`~pathlib.Path` to a file or folder,
            or a sequence mixing those.
        pattern: Glob applied inside folders.
        recursive: Search folders recursively.
        encoding: Encoding used to read files; falls back to latin-1.
        source_type: ``"auto"`` (default), ``"text"`` or ``"path"``. Overrides
            the type-based interpretation of plain strings.

    Returns:
        A list of documents, in the order given; folder contents are sorted by
        path. Document names are unique: file stems, disambiguated with a
        numeric suffix on collision, and ``text`` (or ``text_1``, ``text_2``,
        ... when there is more than one) for inline strings.
    """
    if source_type not in SOURCE_TYPES:
        raise ValueError(
            f"source_type must be one of {', '.join(SOURCE_TYPES)}, not {source_type!r}"
        )

    pending = []  # (kind, value) with kind in {"text", "path"}
    for item in _flatten(source):
        if source_type == "text":
            kind = "text"
        elif source_type == "path" or _is_pathlike(item):
            kind = "path"
        else:
            kind = "text"
        pending.append((kind, item))

    if not pending:
        raise ValueError("source is empty; nothing to analyse.")

    inline_total = sum(1 for kind, _ in pending if kind == "text")
    inline_seen = 0
    used_names = set()
    documents = []

    def unique(name):
        candidate = name
        suffix = 2
        while candidate in used_names:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    for kind, value in pending:
        if kind == "text":
            inline_seen += 1
            name = "text" if inline_total == 1 else f"text_{inline_seen}"
            documents.append(Document(name=unique(name), path=None, text=str(value)))
            continue

        path = Path(value).expanduser()
        if path.is_dir():
            globber = path.rglob if recursive else path.glob
            files = sorted(p for p in globber(pattern) if p.is_file())
            if not files:
                raise ValueError(
                    f"No files matching {pattern!r} in {path}"
                    f"{' (searched recursively)' if recursive else ''}."
                )
            for file_path in files:
                documents.append(
                    Document(
                        name=unique(file_path.stem),
                        path=file_path,
                        text=read_text_file(file_path, encoding),
                    )
                )
        elif path.is_file():
            documents.append(
                Document(
                    name=unique(path.stem),
                    path=path,
                    text=read_text_file(path, encoding),
                )
            )
        else:
            raise FileNotFoundError(f"No such file or folder: {path}")

    return documents
