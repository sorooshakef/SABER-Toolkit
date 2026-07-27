"""The NLP models SABER needs, and a helper to fetch them.

Kept separate from :mod:`saber.nlp` so that naming and downloading the models
does not require loading them.
"""

import logging

logger = logging.getLogger(__name__)

#: Stanza processors loaded for Portuguese. ``constituency`` is deliberately
#: absent: it is one of the heaviest models and no matcher uses it.
STANZA_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"

#: Stanza language code.
STANZA_LANG = "pt"

#: spaCy pipeline used by the matchers flagged ``REQUIRES_SPACY``.
SPACY_MODEL = "pt_core_news_md"


def download_models(stanza=True, spacy=True):
    """Download the Portuguese models, once per machine.

    The Stanza pipeline is built with ``download_method=None`` -- it never
    fetches anything itself -- so this has to be run before the first
    extraction::

        import saber
        saber.download_models()

    Args:
        stanza: Download the Stanza Portuguese models.
        spacy: Download the ``pt_core_news_md`` spaCy pipeline.
    """
    if stanza:
        import stanza as stanza_lib

        logger.info("Downloading Stanza models for %r (%s)", STANZA_LANG, STANZA_PROCESSORS)
        stanza_lib.download(STANZA_LANG, processors=STANZA_PROCESSORS)

    if spacy:
        try:
            import spacy as spacy_lib
        except ModuleNotFoundError as exc:
            # spaCy's top-level __init__ imports its CLI, which pulls in click
            # (via typer). Those arrive with spaCy in a clean install but are
            # sometimes absent in mixed conda/pip environments, where the
            # failure reads as a SABER bug rather than a missing dependency.
            raise ModuleNotFoundError(
                f"Importing spaCy failed: no module named {exc.name!r}. This is an "
                f"incomplete spaCy installation, not a SABER problem -- {exc.name} is "
                "one of spaCy's own dependencies. Reinstall SABER's requirements into "
                'the active environment with: pip install --force-reinstall "spacy==3.8.7"'
            ) from exc

        if spacy_lib.util.is_package(SPACY_MODEL):
            logger.info("spaCy pipeline %s is already installed", SPACY_MODEL)
        else:
            logger.info("Downloading spaCy pipeline %s", SPACY_MODEL)
            spacy_lib.cli.download(SPACY_MODEL)
