"""Compatibility module -- do not rename, and do not put real code here.

Every module in ``saber/matchers`` does ``from process_and_display import
nlp_stanza`` (and/or ``nlp_small``), because those files are maintained in the
upstream SABER application and are copied into this package verbatim. Renaming
this module would break that drop-in contract.

``saber._compat`` registers this module under the bare name
``process_and_display`` in ``sys.modules`` so those absolute imports resolve.
The pipelines themselves live in :mod:`saber.nlp`.
"""

from saber.nlp import (  # noqa: F401
    nlp_small,
    nlp_stanza,
    preprocess_text,
    remove_diacritics,
    stanza_char_spans,
)
