"""Turning the ``source`` argument into a list of named documents.

``source`` may be a raw string of text, a path to a text file, a path to a
folder of text files, or any mixture of those in a sequence. The distinction
between "a string of text" and "a path" is made by type, so it is never
ambiguous:

* :class:`str` is always treated as text,
* :class:`pathlib.Path` (or any :class:`os.PathLike`) is always treated as a path.

Pass ``source_type="path"`` to read a plain string as a path, or
``source_type="text"`` to force the opposite.

Inside a folder, the default selection is every plain-text file: those with a
``.txt`` extension and those with no file-type extension at all -- including
corpus names such as ``A1.1``, where the trailing ``.1`` is part of the name
rather than an extension. Pass an explicit ``pattern`` glob to override that.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SOURCE_TYPES = ("auto", "text", "path")

DEFAULT_SUFFIXES = ("", ".txt")
"""File suffixes picked up in a folder when no ``pattern`` is given."""

_EXTENSION_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")
"""What counts as a file-type extension: a letter, then a few alphanumerics."""

_SNIFF_BYTES = 8192
"""How much of a file is inspected to decide whether it is text or binary."""


@dataclass(frozen=True)
class Document:
    """A single text to analyse."""

    name: str
    """Identifier used in the output tables (file name, or ``text``/``text_1``...)."""

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


def _has_extension(path):
    """Whether ``path`` ends in something that looks like a file-type extension.

    :attr:`~pathlib.Path.suffix` calls anything after the last dot an extension,
    which makes corpus filenames such as ``A1.1`` or ``texto.2`` look like
    ``.1``/``.2`` files and so hides them from the default selection. A real
    extension starts with a letter (see :data:`_EXTENSION_RE`), so a numeric one
    is read as part of the name instead.
    """
    return bool(_EXTENSION_RE.search(path.name))


def _document_name(path):
    """Name a document after its file, dropping only a real extension.

    ``texto57.txt`` becomes ``texto57``, and so does ``texto57``. ``A1.1`` keeps
    its number, since :func:`_has_extension` does not count ``.1`` as an
    extension -- taking the stem there would name ``A1.1`` through ``A1.6`` all
    ``A1`` and leave the collision numbering to tell them apart.
    """
    return path.stem if _has_extension(path) else path.name


def _looks_like_text(path):
    """Whether ``path`` holds text, judged by the absence of a NUL byte.

    Guards the extensionless half of the default selection: without a suffix to
    go on, a binary file is only recognisable by its contents.
    """
    try:
        with path.open("rb") as handle:
            return b"\x00" not in handle.read(_SNIFF_BYTES)
    except OSError as error:
        logger.warning("Cannot read %s: %s", path, error)
        return False


def _is_default_text_file(path):
    """Whether ``path`` belongs to the default, patternless folder selection.

    That is every plain-text file: one with a ``.txt`` extension, or one with no
    file-type extension at all whose contents look like text. Hidden files are
    skipped, since their leading dot would otherwise make ``.DS_Store`` and
    friends look extensionless.
    """
    if path.name.startswith("."):
        return False
    if _has_extension(path):
        return path.suffix.lower() in DEFAULT_SUFFIXES
    return _looks_like_text(path)


def _folder_files(folder, *, pattern, recursive):
    """List the files to read inside ``folder``, sorted by path.

    With ``pattern=None`` every plain-text file is picked up, as decided by
    :func:`_is_default_text_file`; anything left out is logged. An explicit
    ``pattern`` is used as a glob, as given.
    """
    globber = folder.rglob if recursive else folder.glob
    if pattern is not None:
        return sorted(p for p in globber(pattern) if p.is_file())

    files, skipped = [], []
    for path in sorted(p for p in globber("*") if p.is_file()):
        (files if _is_default_text_file(path) else skipped).append(path)
    if skipped:
        logger.info(
            "Skipped %d of %d file(s) in %s as not plain text: %s",
            len(skipped),
            len(skipped) + len(files),
            folder,
            _summarise(skipped),
        )
    return files


def _summarise(paths, limit=5):
    """Join up to ``limit`` file names, noting how many more there are."""
    names = [p.name for p in paths[:limit]]
    if len(paths) > limit:
        names.append(f"... and {len(paths) - limit} more")
    return ", ".join(names)


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
    pattern=None,
    recursive=False,
    encoding="utf-8",
    source_type="auto",
):
    """Expand ``source`` into a list of :class:`Document` objects.

    Args:
        source: A string of text, a :class:`~pathlib.Path` to a file or folder,
            or a sequence mixing those.
        pattern: Glob applied inside folders. ``None`` (the default) reads every
            plain-text file: ``.txt`` files and files with no file-type
            extension, such as ``texto57`` or ``A1.1``.
        recursive: Search folders recursively.
        encoding: Encoding used to read files; falls back to latin-1.
        source_type: ``"auto"`` (default), ``"text"`` or ``"path"``. Overrides
            the type-based interpretation of plain strings.

    Returns:
        A list of documents, in the order given; folder contents are sorted by
        path. Document names are unique: file names without their extension,
        disambiguated with a numeric suffix on collision, and ``text`` (or
        ``text_1``, ``text_2``, ... when there is more than one) for inline
        strings.
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
            files = _folder_files(path, pattern=pattern, recursive=recursive)
            if not files:
                what = (
                    "No plain-text files"
                    if pattern is None
                    else f"No files matching {pattern!r}"
                )
                raise ValueError(
                    f"{what} in {path}"
                    f"{' (searched recursively)' if recursive else ''}."
                )
            for file_path in files:
                documents.append(
                    Document(
                        name=unique(_document_name(file_path)),
                        path=file_path,
                        text=read_text_file(file_path, encoding),
                    )
                )
        elif path.is_file():
            documents.append(
                Document(
                    name=unique(_document_name(path)),
                    path=path,
                    text=read_text_file(path, encoding),
                )
            )
        else:
            raise FileNotFoundError(f"No such file or folder: {path}")

    return documents
