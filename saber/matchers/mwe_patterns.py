"""Dependency-based multiword-expression pattern extractors for SABER.

Copy this module to ``saber/matchers/``. The functions return SABER's usual
``(structure, text, start, end)`` tuples and use the Universal Dependencies
parse exposed by SABER's Stanza-backed spaCy document.
"""

from collections import defaultdict

from spacy.matcher import DependencyMatcher

from text_reconstruction import reconstruct_text


def _as_saber_results(doc, matches):
    """Turn DependencyMatcher matches into de-duplicated SABER result tuples."""
    results = []
    seen = set()

    for match_id, token_ids in matches:
        start = min(token_ids)
        end = max(token_ids) + 1
        key = (match_id, start, end)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            (
                doc.vocab.strings[match_id],
                reconstruct_text(doc[start:end]),
                start,
                end,
            )
        )

    return sorted(results, key=lambda result: (result[2], result[3], result[0]))


def b2d4_1(doc):
    """Extract adjective--noun expressions with an ``amod`` relation.

    Matches adjective-before-noun and noun-before-adjective order, including
    intervening material: ``grande casa``, ``casa grande``, and
    ``casa muito grande``.
    """
    matcher = DependencyMatcher(doc.vocab)
    pattern = [
        {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}},
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "adjective",
            "RIGHT_ATTRS": {"POS": "ADJ", "DEP": "amod"},
            "REL_OP": ">",
        },
    ]

    matcher.add("b2d4_1_MWE", [pattern])
    return _as_saber_results(doc, matcher(doc))


def b2d4_2(doc):
    """Extract noun + preposition + noun nominal groups.

    The second noun must be an ``nmod`` child of the first, and the
    preposition must be the dependent noun's ``case`` child. Determiners and
    other material between these lexical anchors are therefore permitted.

    Example: ``vontade de ferro``.
    """
    matcher = DependencyMatcher(doc.vocab)
    pattern = [
        {"RIGHT_ID": "head_noun", "RIGHT_ATTRS": {"POS": "NOUN"}},
        {
            "LEFT_ID": "head_noun",
            "RIGHT_ID": "dependent_noun",
            "RIGHT_ATTRS": {"POS": "NOUN", "DEP": "nmod"},
            "REL_OP": ">",
        },
        {
            "LEFT_ID": "dependent_noun",
            "RIGHT_ID": "preposition",
            "RIGHT_ATTRS": {"POS": "ADP", "DEP": "case"},
            "REL_OP": ">",
        },
    ]

    matcher.add("b2d4_2_MWE", [pattern])
    return _as_saber_results(doc, matcher(doc))


def b2d4_3(doc):
    """Extract verbal object patterns: verb + optional determiner + noun.

    The lexical anchors are a verb and its nominal direct object
    (``VERB --obj--> NOUN``). Objects which govern an adposition as a ``case``
    child are excluded: they are prepositional complements that the parser has
    labelled ``obj``, rather than the bare noun-object pattern targeted here.
    A determiner, adjective, or other non-adpositional material in the object
    noun phrase is permitted and kept in the returned span.

    Examples: ``dar importância`` and ``dar muita importância``.
    """
    matcher = DependencyMatcher(doc.vocab)
    pattern = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "object",
            "RIGHT_ATTRS": {"POS": "NOUN", "DEP": "obj"},
            "REL_OP": ">",
        },
    ]

    matcher.add("b2d4_3_MWE", [pattern])
    matches_without_case_adposition = []
    for match_id, token_ids in matcher(doc):
        # ``token_ids`` follows the RIGHT_ID order in the pattern: verb, object.
        object_token = doc[token_ids[1]]
        has_case_adposition = any(
            child.pos_ == "ADP" and child.dep_.split(":", 1)[0] == "case"
            for child in object_token.children
        )
        if not has_case_adposition:
            matches_without_case_adposition.append((match_id, token_ids))

    return _as_saber_results(doc, matches_without_case_adposition)

def b2d4_4(doc):
    """Extract verb + preposition + optional determiner + noun patterns.

    This is the UD configuration ``VERB --obl--> NOUN`` together with
    ``NOUN --case--> ADP``. The returned span includes a determiner or other
    intervening material in the oblique noun phrase.

    Examples: ``pôr em prática`` and ``levar ao fim``.
    """
    matcher = DependencyMatcher(doc.vocab)
    pattern = [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "oblique_noun",
            "RIGHT_ATTRS": {"POS": "NOUN", "DEP": "obl"},
            "REL_OP": ">",
        },
        {
            "LEFT_ID": "oblique_noun",
            "RIGHT_ID": "preposition",
            "RIGHT_ATTRS": {"POS": "ADP", "DEP": "case"},
            "REL_OP": ">",
        },
    ]

    matcher.add("b2d4_4_MWE", [pattern])
    return _as_saber_results(doc, matcher(doc))


def b2d4_5(doc):
    """Extract functional MWEs connected by one or more ``fixed`` relations.

    The ``fixed`` relation can form a chain, so a direct dependency pattern is
    insufficient for expressions such as ``em cima de``. This function finds
    each connected component made exclusively from ``fixed`` edges and returns
    the complete token span. It supports both a direct ``fixed`` relation (for
    example, ``para que``) and arbitrarily long chains.
    """
    adjacency = defaultdict(set)
    for token in doc:
        # Accept an enhanced subtype too, while retaining the UD base relation.
        if token.dep_.split(":", 1)[0] == "fixed" and token.head != token:
            adjacency[token.i].add(token.head.i)
            adjacency[token.head.i].add(token.i)

    results = []
    seen = set()
    for start_node in adjacency:
        if start_node in seen:
            continue

        component = set()
        pending = [start_node]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(adjacency[node] - component)

        seen.update(component)
        start = min(component)
        end = max(component) + 1
        results.append(
            (
                "b2d4_5_MWE",
                reconstruct_text(doc[start:end]),
                start,
                end,
            )
        )

    return sorted(results, key=lambda result: (result[2], result[3]))
