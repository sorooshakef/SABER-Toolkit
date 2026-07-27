"""Make the matchers' absolute imports resolve inside this package.

The modules in ``saber/matchers`` are copied verbatim from the upstream SABER
application, where they live next to their dependencies at the top level of the
project. They therefore contain absolute imports such as::

    from process_and_display import nlp_stanza
    from text_reconstruction import reconstruct_text

Rewriting those lines would break the drop-in contract (the folder is re-copied
whenever the matchers are updated upstream), so instead the three modules they
may refer to are registered in ``sys.modules`` under their bare names as well as
their proper ``saber.*`` names. Both names point at the same module object, so
``nlp_stanza`` stays a singleton either way.

:func:`install` must run before any matcher module is imported.
:mod:`saber.registry` calls it.
"""

import importlib
import sys

#: Modules the drop-in matchers may import by bare name.
ALIASED_MODULES = ("process_and_display", "text_reconstruction", "label_translation")


def install():
    """Register the ``saber.*`` support modules under their bare names too."""
    for name in ALIASED_MODULES:
        if name not in sys.modules:
            sys.modules[name] = importlib.import_module(f"saber.{name}")
