# SABER

**SABER** (*Sistema de Análise e Busca de Estruturas Relevantes*) identifies
European Portuguese grammatical structures in text and reports the CEFR level
(A1–C2) at which each one is taught, following the
[**Referencial Camões**](https://www.instituto-camoes.pt/activity/centro-virtual/referencial-camoes-ple).

It gives you two things:

- **Spans** — every occurrence of every structure, with its text, its token and
  character offsets, and its CEFR level. One row per occurrence.
- **Features** — per document, the raw frequency, the frequency per 100 tokens,
  and the presence (0/1) of each structure. One row per document, ready to feed
  into a model or a statistical analysis.

252 structures are covered, described in Portuguese and organised into 11 broad
groups (nouns, adjectives, verbs, adverbs, pronouns, determiners, quantifiers,
relations between constituents, sentence types, sentence polarity, relations
between sentences) and 31 finer categories. You can extract all of them, or
filter by group, category, CEFR level, or individual structure.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input: strings, files, folders](#input-strings-files-folders)
- [Choosing which structures to extract](#choosing-which-structures-to-extract)
- [Output columns](#output-columns)
- [How it works](#how-it-works)
- [Performance](#performance)
- [Working on the matchers](#working-on-the-matchers)
- [Testing the matchers](#testing-the-matchers)
- [Licence and attribution](#licence-and-attribution)

## Installation

Python 3.9 or newer, but not above 3.13.

From a clone:

```bash
git clone https://github.com/sorooshakef/SABER-Toolkit.git
cd SABER-Toolkit
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

Then download the Portuguese models **once per machine** (~1 GB, mostly Stanza):

```bash
python -c "import saber; saber.download_models()"
```

> **Note on PyTorch.** `torch` is pinned to 2.5.1 because Stanza's model loading
> breaks with newer releases. If another package upgrades it, reinstall with
> `pip install torch==2.5.1`.

## Quick start

```python
import saber

spans = saber.extract("Ele foi ao cinema no domingo com a minha irmã.", mode="spans")
print(spans[["structure", "level", "text", "description"]])
```

```
    structure level          text                                        description
0  a3d1_12_A2    A2           foi  Pretérito perfeito simples do indicativo - ver...
1   a6d1_6_A1    A1     ao cinema  Artigo definido - concordância - em género e n...
2   a1d1_7_A2    A2        cinema                Nomes masculinos terminados em "a"
3  a6d1_10_A2    A2            no             Artigo definido - contração com prep...
4   a6d1_6_A1    A1    no domingo  Artigo definido - concordância - em género e n...
5   a6d1_5_A1    A1       a minha  Artigo definido - uso/valor - antes de determi...
6   a6d1_6_A1    A1  a minha irmã  Artigo definido - concordância - em género e n...
7   a5d3_1_A1    A1    minha irmã  Pronomes Possessivos - variação em pessoa, gén...
```

And the feature matrix for a folder of texts:

```python
from pathlib import Path

features = saber.extract(Path("corpus/"), mode="features")
features.to_csv("features.csv", index=False)
```

`extract` always returns a `pandas.DataFrame`. `saber.extract_spans(...)` and
`saber.extract_features(...)` are aliases for the two modes.

## Input: strings, files, folders

The first argument accepts text, a file, a folder, or a mix of them. The
distinction between "a string of text" and "a path" is made by **type**, so it
is never guessed:

```python
saber.extract("Um texto qualquer.")                 # str  -> raw text
saber.extract(Path("texto.txt"))                    # Path -> one file
saber.extract(Path("corpus/"))                      # Path -> every .txt in the folder
saber.extract(["Primeiro texto.", "Segundo."])      # several texts
saber.extract([Path("a.txt"), Path("corpus/")])     # several paths
```

To read a plain string as a path, pass `source_type="path"`.

Folder options: `pattern="*.txt"` (which files to pick up), `recursive=False`,
`encoding="utf-8"` (files that are not valid UTF-8 fall back to latin-1 with a
warning).

Each document is named after its file stem; inline strings become `text`, or
`text_1`, `text_2`, … when there are several.

## Choosing which structures to extract

Four independent filters. Values within one filter are OR-ed, and the filters
are AND-ed together. Matching ignores case and diacritics.

```python
# A broad group: by key or by its Portuguese name
saber.extract(text, groups="a5")
saber.extract(text, groups="Pronomes")            # the same 29 structures

# A finer category
saber.extract(text, categories="a5d2")            # Pronomes > Demonstrativos

# A CEFR level, or several
saber.extract(text, levels="B1")
saber.extract(text, levels=["A1", "A2"])

# Individual structures, by id or by matcher name
saber.extract(text, structures=["a5d2_1_A2", "b5d2_6"])

# Combined: subjunctive-level pronoun structures only
saber.extract(text, groups="a5", levels="B2")

# Everything except a few structures
saber.extract(text, exclude=["a1d1_7_A2"])
```

To see what the valid values are:

```python
saber.list_structures()                # all 252, as a DataFrame
saber.list_structures(groups="a7")     # just the quantifiers
```

| structure | matcher | level | group | group_name | category | category_name | description |
|---|---|---|---|---|---|---|---|
| a5d2_1_A2 | a5d2_1 | A2 | a5 | Pronomes | a5d2 | Demonstrativos | Pronomes - Demonstrativos - contração com preposições |

An unknown filter value raises `ValueError` listing the valid options, and so
does a combination that selects nothing.

Structure ids encode the taxonomy, so you can also filter the output frame
directly: `a5d2_1_A2` → group `a5`, category `a5d2`, matcher `a5d2_1`, level
`A2`.

## Output columns

### `mode="spans"` — one row per occurrence

| Column | Meaning |
|---|---|
| `document`, `path` | Which text the match came from (`path` is empty for inline strings) |
| `structure` | Full structure id, e.g. `a5d2_1_A2` |
| `matcher` | Name of the matcher function, e.g. `a5d2_1` |
| `level` | CEFR level, `A1`–`C2` |
| `group`, `group_name` | Broad group key and Portuguese name |
| `category`, `category_name` | Category key and Portuguese name |
| `description` | Portuguese description of the structure |
| `text` | The matched substring, sliced out of the preprocessed text |
| `reconstructed_text` | The matcher's own rendering of the match |
| `token_start`, `token_end` | Token offsets, end-exclusive |
| `char_start`, `char_end` | Character offsets into the preprocessed text |
| `pipeline` | `stanza` or `spacy`, depending on which parse the matcher needs |

Rows are sorted by document and position. Structures overlap freely — a single
phrase commonly matches several — so occurrences are *not* mutually exclusive.

`text` is the authoritative one: it is cut from the text by character offset, so
`analysis.text[char_start:char_end] == text` always holds and contractions come
out as written (`ao`, `no`, `daquele`). `reconstructed_text` comes from the
matcher's own `reconstruct_text` helper and is normally identical, but it is
rebuilt from tokens rather than sliced, so keep to `text` when exactness matters.

To get offsets against your own untouched string, note that they refer to the
*preprocessed* text — see [How it works](#how-it-works). `saber.analyze()`
returns that text alongside the matches if you need it:

```python
analysis = saber.analyze("Ele foi ao cinema.", saber.all_structures())
analysis.text                # the preprocessed text the offsets refer to
analysis.n_tokens            # tokens excluding punctuation
analysis.matches             # list of Match objects
```

### `mode="features"` — one row per document

| Column | Meaning |
|---|---|
| `document`, `path` | Which text the row describes |
| `n_tokens` | Tokens excluding whitespace and punctuation |
| `<structure>_count` | Raw number of occurrences |
| `<structure>_norm` | Occurrences per 100 tokens |
| `<structure>_present` | `1` if the structure occurs at all, else `0` |

Columns are emitted for **every** selected structure, including those that never
matched, so feature matrices line up across documents and across runs. Their
order follows the declaration order of `LABEL_TRANSLATIONS` in
`saber/label_translation.py`, which is curated taxonomically.

Selecting all 252 structures therefore gives `3 × 252 + 3 = 759` columns. Use
`features=` to keep only what you need:

```python
saber.extract(corpus, mode="features", features=["normalized", "presence"])
```

## How it works

1. **Preprocessing.** Dialogue dashes are stripped and whitespace collapsed. All
   offsets refer to this preprocessed text, which `saber.analyze()` exposes as
   `analysis.text`.
2. **Parsing.** The text is parsed twice: by
   [spaCy-Stanza](https://github.com/explosion/spacy-stanza) (`tokenize`, `mwt`,
   `pos`, `lemma`, `depparse`) for morphology and dependencies, and by spaCy's
   `pt_core_news_md` for the handful of matchers that need named entities or
   spaCy's own tokenisation. A matcher declares which parse it needs; you never
   choose.
3. **Matching.** Each selected structure's matcher runs over the appropriate
   parse using spaCy's `Matcher`, `PhraseMatcher` and `DependencyMatcher`, and
   returns token spans.
4. **Offsets.** Token spans are converted to character offsets. Stanza expands
   multi-word tokens (`no` → `em o`), which destroys spaCy's own character
   offsets, so they are recovered from the underlying Stanza document — this is
   why `text` reads `no` and not `em o`.
5. **Tabulating.** Occurrences become the spans table, or are counted and
   normalised into the feature table.

A matcher that raises is reported as a warning and skipped, so one broken
pattern cannot abort a corpus run. Use `on_error="raise"` while debugging, or
`on_error="ignore"` to silence it.

## Performance

Expect **a few seconds per short text**, and add ~30 s of one-off model loading
to the first extraction in a process. Two things dominate: the Stanza parse, and
the fact that each matcher rebuilds its spaCy `Matcher` object on every call.

Practical consequences:

- Extract once for all the structures you might want, then filter the resulting
  DataFrame — that is much cheaper than re-running with different filters.
- Restricting `groups`/`categories`/`levels` up front does cut the matching cost
  roughly in proportion, but not the parsing cost.
- Progress is logged per document. Turn it on with
  `logging.basicConfig(level=logging.INFO)`.

## Working on the matchers

`saber/matchers/` and `saber/label_translation.py` are maintained in the
upstream SABER application and copied into this package **verbatim**. Do not
edit them here; replace them wholesale and they keep working. Two consequences
worth knowing:

- `saber/matchers/` has no `__init__.py`, on purpose — it is an implicit
  namespace package, so replacing the folder cannot delete a file this project
  needs.
- The matchers import their dependencies by bare name
  (`from process_and_display import nlp_stanza`,
  `from text_reconstruction import reconstruct_text`). `saber/_compat.py`
  registers those names in `sys.modules` before any matcher is imported, and
  `saber/process_and_display.py` is a thin shim re-exporting the pipelines from
  `saber/nlp.py`. That is why the module keeps its otherwise odd name.

A matcher function is named after its structure minus the CEFR suffix
(`a5d2_1` → `a5d2_1_A2`), takes a spaCy `Doc`, and returns a list of
`(label, text, token_start, token_end)` tuples. Setting
`<function>.REQUIRES_SPACY = True` after the definition routes it to the
`pt_core_news_md` parse instead of the Stanza one.

A structure is available only if it has **both** an entry in
`LABEL_TRANSLATIONS` and a matcher function, so commenting out a label in
`saber/label_translation.py` disables the structure.
`saber.missing_labels()` lists labels with no matcher.

## Testing the matchers

`tests/tests.yaml` holds positive and negative example sentences per matcher.
The harness measures precision and recall for each, both with all patterns
active and with only the first pattern registered:

```bash
pip install -e ".[dev]"
python tests/test_suite.py
```

Results are written to `tests/test_results.csv`; the committed copy is the
reference from the last full run.

## Licence and attribution

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
[LICENSE](LICENSE). You may share and adapt this work, including commercially,
provided you give appropriate credit.

The structure inventory and its CEFR mapping derive from the
[Referencial Camões PLE](https://www.instituto-camoes.pt/activity/centro-virtual/referencial-camoes-ple)
(Instituto Camões).
