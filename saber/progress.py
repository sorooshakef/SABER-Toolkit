"""The progress bar shown while a corpus is being analysed.

Extraction is slow enough (a few seconds per text) that a multi-document run
needs to show that it is making progress. Per-document ``logging.INFO`` lines
still do that for scripts and log files; this module is the interactive
counterpart, a single self-overwriting line on stderr.
"""

import contextlib
import logging
import sys

logger = logging.getLogger(__name__)

PROGRESS = ("auto", True, False)


class _NullBar:
    """No-op stand-in with the two methods :func:`bar` promises."""

    def update(self, n=1):
        pass

    def describe(self, text):
        pass


class _TqdmBar:
    def __init__(self, tqdm):
        self._tqdm = tqdm

    def update(self, n=1):
        self._tqdm.update(n)

    def describe(self, text):
        # refresh=False: the description is set just before a step that takes
        # seconds, and the following update() redraws the line anyway.
        self._tqdm.set_description(text, refresh=False)


def _in_notebook():
    """True inside a Jupyter kernel, where stderr is not a terminal but tqdm
    still renders a bar (``tqdm.auto`` switches to the widget version)."""
    ipython = sys.modules.get("IPython")
    if ipython is None:  # IPython is not even imported: not a notebook
        return False
    get_ipython = getattr(ipython, "get_ipython", None)
    shell = get_ipython() if get_ipython else None
    # ZMQInteractiveShell is Jupyter/qtconsole; TerminalInteractiveShell is the
    # plain `ipython` REPL, which is a terminal and gets caught by isatty().
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _enabled(progress, total):
    if progress not in PROGRESS:
        raise ValueError(
            f"progress must be one of 'auto', True, False, not {progress!r}"
        )
    if progress is False:
        return False
    if progress is True:
        return True
    # "auto": a bar is only useful when there is more than one step to watch,
    # and only when someone is watching -- writing carriage returns into a
    # redirected stderr or a log file just produces noise.
    if total <= 1:
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)()) or _in_notebook()


@contextlib.contextmanager
def bar(total, *, progress="auto", unit="text"):
    """Yield a progress bar over ``total`` steps.

    The bar exposes ``update(n=1)`` and ``describe(text)``. It is a no-op when
    ``progress`` is ``False``, when ``progress`` is ``"auto"`` and stderr is not
    a terminal or there is only one step, or when tqdm is not installed.

    Args:
        total: Number of steps the bar counts up to.
        progress: ``"auto"`` (default), ``True`` to force the bar on, ``False``
            to force it off.
        unit: Noun for one step, shown in the rate.
    """
    if not _enabled(progress, total):
        yield _NullBar()
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:
        # tqdm is a declared dependency, so this only happens in an environment
        # that was assembled by hand. Losing the bar is not worth an error.
        logger.debug("tqdm is not installed; progress bar disabled")
        yield _NullBar()
        return

    # leave=True: the finished bar stays on screen as a record of how long the
    # run took, and closing it emits the newline that keeps whatever the caller
    # prints next off the bar's line.
    with tqdm(total=total, unit=unit, leave=True,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as t:
        yield _TqdmBar(t)
