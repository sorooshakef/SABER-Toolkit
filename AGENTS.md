# Repository Guidelines

SABER is a Python library that identifies European Portuguese grammatical
structures in text and maps them to the CEFR level at which they are taught
(Referencial Camões). See `README.md` for user-facing documentation.

## Project structure

```
saber/
  __init__.py             public API (extract, list_structures, download_models, ...)
  extraction.py           runs matchers over documents; builds the spans / feature tables
  registry.py             matcher discovery, the structure taxonomy, filter resolution
  sources.py              turns a string / file / folder into named documents
  nlp.py                  the two NLP pipelines, preprocess_text, stanza_char_spans
  progress.py             the per-document progress bar (tqdm, optional)
  models.py              model names and download_models()
  process_and_display.py  compat shim -- see below
  text_reconstruction.py  upstream, verbatim
  label_translation.py    upstream, verbatim
  matchers/               upstream, verbatim (no __init__.py on purpose)
tests/
  tests.yaml              positive/negative example sentences per matcher
  test_suite.py           precision/recall harness
  test_results.csv        reference results from the last full run
```

**`saber/matchers/` and `saber/label_translation.py` are copied verbatim from the
upstream SABER application. Do not edit them here** — they are replaced wholesale
whenever the matchers are updated, so any local change is lost. Everything else
in `saber/` is this project's own code.

Two consequences of that contract:

- `saber/matchers/` has no `__init__.py`; it is an implicit namespace package so
  that replacing the folder cannot delete a file the project needs.
  `pyproject.toml` lists `saber.matchers` explicitly for packaging.
- The matchers import their dependencies by bare name
  (`from process_and_display import nlp_stanza`,
  `from text_reconstruction import reconstruct_text`). `saber/_compat.py`
  registers those names in `sys.modules`, and `registry.load_matchers()` calls it
  before importing any matcher module. `saber/process_and_display.py` exists only
  to keep that name resolvable; the pipelines live in `saber/nlp.py`.

## Development commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import saber; saber.download_models()"    # once per machine
python tests/test_suite.py                           # precision/recall over tests.yaml
```

To inspect a single matcher's raw hits:

```bash
python -c "from saber.matchers.b5_2_subordination import b5d2_6; from saber.nlp import nlp_stanza; print(b5d2_6(nlp_stanza('Fiquei em casa porque estava a chover.')))"
```

## Conventions

- PEP 8, 4-space indentation, docstrings that cite the CEFR reference where
  relevant.
- Matcher ids follow the Referencial Camões scheme, e.g. `a3d1_36_B1`. The
  matcher function is named after the id minus the level suffix (`a3d1_36`), takes
  a spaCy `Doc`, and returns `(label, text, token_start, token_end)` tuples with
  an exclusive end index. `<function>.REQUIRES_SPACY = True` routes it to the
  `pt_core_news_md` parse instead of the Stanza one.
- A structure is only available if it has both a `LABEL_TRANSLATIONS` entry and a
  matcher function; commenting out a label disables the structure.
- Library code logs through `logging` (`logger = logging.getLogger(__name__)`).
  Never `print()` — matcher modules still contain a few stray debug prints from
  upstream, which show up as noise in library output and should be removed
  upstream.
- Character offsets always refer to the *preprocessed* text
  (`saber.nlp.preprocess_text`). In `extraction.analyze`, `stanza_char_spans` must
  be called immediately after `nlp_stanza(text)` and before any matcher runs:
  some matchers call `nlp_stanza.pipe()` internally, which overwrites the single
  cached Stanza doc that the offset mapping is derived from.

## Testing

`tests/test_suite.py` is the regression check that matters: it runs every matcher
against `tests.yaml` and reports precision and recall, both with all patterns
active and with only the first pattern registered. Run it after any change to the
pipelines, the registry, or the matchers, and compare against the committed
`tests/test_results.csv`.

When adding a matcher upstream, add positive and negative sentences for it to
`tests.yaml` in the same commit.

## Commit & pull request guidelines

Short, capitalised, present-tense subjects, e.g. "Fix active-passive agreement
edge case". Mention the CEFR code when it helps. In a PR, summarise the change,
note how it was validated (precision/recall deltas from `test_suite.py` are ideal),
and link any related issues.
