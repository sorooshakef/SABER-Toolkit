"""The two Portuguese NLP pipelines the matchers run on, plus the text
preprocessing and token-to-character mapping helpers they need.

Importing this module loads both pipelines, which takes a few seconds and a few
hundred megabytes of memory. Nothing else in ``saber`` imports it at package
import time -- it is pulled in lazily the first time structures are extracted.
"""

import re

import spacy
import spacy_stanza
from spacy.language import Language
from spacy.tokens import Token
from unidecode import unidecode

from saber.models import SPACY_MODEL, STANZA_LANG, STANZA_PROCESSORS

# Note on the GPU: Apple Silicon (M-series) GPUs are used via PyTorch's MPS
# backend, not CUDA, so Stanza's `use_gpu` flag (which only checks for CUDA)
# never sees them. But MPS is not worth using here anyway -- Stanza's models are
# small, token-sequential networks, and benchmarking showed the MPS backend
# running ~2x SLOWER than the CPU because of per-op GPU overhead. So we use the
# CPU on Apple Silicon and only reach for a real CUDA GPU if present.
import torch

_stanza_device = "cuda" if torch.cuda.is_available() else "cpu"

# Only load the processors the matchers actually use. The `constituency` parser
# is one of the heaviest models and is referenced by no matcher, so it is
# omitted; `depparse` is kept because several matchers use dependency relations.
nlp_stanza = spacy_stanza.load_pipeline(
    name=STANZA_LANG,
    processors=STANZA_PROCESSORS,
    device=_stanza_device,
    download_method=None,
)


class _SnlpCache:
    """Wrap the underlying Stanza pipeline so the most recent Stanza Doc is kept
    accessible.

    spacy-stanza rebuilds ``doc.text`` from space-joined, MWT-expanded syntactic
    words whenever a text contains a multi-word token (any contraction such as
    "no" -> "em o"), which discards every token's original character offset. The
    raw Stanza Doc, however, always keeps correct ``start_char``/``end_char`` on
    each surface token, so caching it lets us recover an exact token->character
    mapping (see ``stanza_char_spans``). This runs the pipeline only once."""

    def __init__(self, snlp):
        self._snlp = snlp
        self.last_doc = None

    def __call__(self, text):
        self.last_doc = self._snlp(text)
        return self.last_doc

    def __getattr__(self, name):
        return getattr(self._snlp, name)


nlp_stanza.tokenizer.snlp = _SnlpCache(nlp_stanza.tokenizer.snlp)

# Load the spaCy pipeline for Portuguese
nlp_small = spacy.load(SPACY_MODEL)

# Add custom extension attribute
Token.set_extension("past_in_sentence", default=False, force=True)


@Language.component("remove_diacritics")
def remove_diacritics(doc):
    for token in doc:
        token.norm_ = unidecode(token.text.lower())
    return doc


# Registered on the Stanza pipeline only. Matchers that key on the NORM
# attribute rely on this, which is part of why a matcher cannot simply be run on
# whichever pipeline happens to be convenient.
nlp_stanza.add_pipe("remove_diacritics", name="remove_diacritics_component", last=True)


def preprocess_text(text):
    """
    Preprocess text to remove noisy characters while preserving valid punctuation.
    Specifically handles dialogue dashes that appear after punctuation marks or at the beginning.
    """
    # Remove dialogue dashes at the beginning of text
    cleaned_text = re.sub(r'^[\s\-—–]+', '', text)

    # Remove dialogue dashes that appear after punctuation marks
    # Pattern: punctuation + optional space + dash + optional space
    cleaned_text = re.sub(r'([.!?,:;])\s*[-—–]\s*', r'\1 ', cleaned_text)

    # Clean up multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text.strip()


def stanza_char_spans(doc_stanza, snlp_doc):
    """Map every ``doc_stanza`` token to the ``(start_char, end_char)`` span of
    its parent Stanza token in the original text (``None`` for inserted
    whitespace tokens).

    The spaCy Doc's tokens are the Stanza *words* flattened in reading order
    (sentence -> token -> words), 1:1 with the doc's non-space tokens, and each
    word's parent Stanza *token* always keeps a correct character span into the
    original text -- even when spacy-stanza has discarded ``token.idx`` because
    of multi-word-token expansion. This yields an exact, unambiguous
    token->character mapping, so a matched token span can be located without
    re-searching the text for its surface string."""
    # snlp_doc is None for empty/whitespace-only input, which spacy-stanza
    # tokenizes without running the pipeline; such a doc has no real tokens.
    word_spans = [
        (tok.start_char, tok.end_char)
        for sent in snlp_doc.sentences
        for tok in sent.tokens
        for _ in tok.words
    ] if snlp_doc is not None else []
    spans = []
    wi = 0
    for token in doc_stanza:
        if token.is_space:
            spans.append(None)
        else:
            spans.append(word_spans[wi] if wi < len(word_spans) else None)
            wi += 1
    return spans


# Tokens that do not count towards text length when normalizing structure
# frequencies.
PUNCT_UPOS = {"PUNCT", "SYM"}


def count_tokens(doc):
    """Number of tokens in ``doc`` that are neither whitespace nor punctuation.

    This is the denominator used for normalized structure frequencies.
    """
    return sum(1 for token in doc if not token.is_space and token.pos_ not in PUNCT_UPOS)
