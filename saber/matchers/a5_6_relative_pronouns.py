from spacy.matcher import Matcher, PhraseMatcher
from text_reconstruction import reconstruct_text
from process_and_display import nlp_small

from spacy.matcher import Matcher

def a5d6_1(doc):
    """
    A5.6-1: Relativos - forma - invariáveis
    e.g., que, quem, [advérbio relativo] onde
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    # --- Patterns that should return the LAST token of the match ---
    pattern = [
        # To prevent "Com quem e onde viste o Rui?" from matching
        {"LOWER": {"NOT_IN": ["o", "com", "e", "a", "desde"]}},
        {"LOWER": {"IN": ["que", "quem", "onde"]}, "MORPH": {"IS_SUPERSET": ["PronType=Rel"]}}
    ]
    
    pattern_onde = [
        {"POS": "ADV"},
        {"LOWER": "onde"}
    ]
    
    # --- Pattern that should return the FIRST token of the match ---
    pattern_sent_start = [
        {"LOWER": {"IN": ["que", "quem", "onde"]}, "MORPH": {"IS_SUPERSET": ["PronType=Rel"]}, "IS_SENT_START": True},
        {"LOWER": {"NOT_IN": [".", "?", "!"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": [".", "!"]}}
    ]

    # Add the pattern groups with unique names to identify them later
    matcher.add("A5D6_1_LAST_TOKEN", [pattern, pattern_onde])
    matcher.add("A5D6_1_FIRST_TOKEN", [pattern_sent_start])

    matches = matcher(doc)

    # The final ID string we want to return for all matches
    final_match_id = "a5d6_1_B1"
    results = []

    for match_id, start, end in matches:
        # Get the string name of the rule that was matched
        rule_name = doc.vocab.strings[match_id]

        if rule_name == "A5D6_1_FIRST_TOKEN":
            # If the match came from pattern_sent_start, get the FIRST token
            token_index = start
            token = doc[token_index]
            # FIX: The end index must be exclusive (token.i + 1)
            results.append((final_match_id, token.text, token.i, token.i + 1))
        else:
            # Otherwise (for pattern and pattern_onde), get the LAST token
            token_index = end - 1
            token = doc[token_index]
            # FIX: The end index must be exclusive (token.i + 1)
            results.append((final_match_id, token.text, token.i, token.i + 1))

    return results

# def a5d6_5(doc):
#     """
#     A5.6-5: Relativos - forma - variação em género e número
#     e.g., o qual, a qual, os quais, as quais
#     Level: B2
#     """
#     matcher = PhraseMatcher(doc.vocab, attr="LOWER")

#     patterns = list(nlp_small.pipe(["o qual", "a qual", "os quais", "as quais"]))

#     matcher.add("a5d6_5_B2", patterns)

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
# a5d6_5.REQUIRES_SPACY = True

def a5d6_5(doc):
    """
     A5.6-5: Pronomes - Relativos - forma - variação em género e número
    e.g., o qual, a qual, os quais, as quais
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    # The article can be preceded by a preposition that forms a contraction
    # with it (e.g., de+o -> "do qual", em+a -> "na qual", por+o -> "pelo qual",
    # a+o -> "ao qual"). The spaCy-Stanza pipeline splits these contractions
    # into two tokens (preposition + article), so we capture the optional
    # leading preposition to include it in the highlight.
    pattern = [
                {"LEMMA": {"IN": ["de", "em", "por", "a"]}, "POS": "ADP", "OP": "?"},
                {"MORPH": {"IS_SUPERSET": ["PronType=Art"]}, "IS_SENT_START": False},
                {"LEMMA": "qual"}
            ]

    # greedy="LONGEST" keeps the contraction-inclusive match instead of also
    # returning the shorter "o qual" match at the same position.
    matcher.add("a5d6_5_B2", [pattern], greedy="LONGEST")

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a5d6_6(doc):
    """
    A5.6-6: Pronomes - Relativos - forma - invariáveis
    e.g., o que
    Level: B2

    This function matches patterns like "do que" and excludes matches
    if the last token ("que") is in the same dependency tree as a token
    with lower form "mais" or "menos" (e.g., in comparative sentences).

    Não entendo nada do que ele disse.
    But not "Ele é mais alto do que eu."
    """
    from spacy.matcher import Matcher

    matcher = Matcher(doc.vocab)

    # Pattern to match "do que" where "que" is a relative pronoun.
    # The preceding preposition is optional and, when present, must be one of
    # the relevant prepositions.
    pattern = [
        {"LOWER": {"IN": ["em", "de", "por", "a", "o"]}, "OP": "?"},
        {"LOWER": "o"},
        {"LOWER": "que", "MORPH": {"IS_SUPERSET": ["PronType=Rel"]}}
    ]

    matcher.add("a5d6_6_B2", [pattern], greedy="LONGEST")
    matches = matcher(doc)
    valid_matches = []

    for match_id, start, end in matches:
        current_start = start
        current_end = end

        # Identify the last token ("que") of the match
        token_que = doc[current_end - 1]

        # Find the root of the dependency tree containing token_que.
        dep_root = token_que
        while dep_root.head != dep_root:
            dep_root = dep_root.head

        # Check if any token in the dependency tree (except token_que itself)
        # has a lower form of "mais" or "menos".
        if any(t.lower_ in ["mais", "menos", "melhor", "pior", "maior"] for t in dep_root.subtree if t.i != token_que.i):
            # Skip this match because it's likely part of a comparative structure.
            continue

        valid_matches.append(
            (doc.vocab.strings[match_id], reconstruct_text(doc[current_start:current_end]), current_start, current_end)
        )

    return valid_matches
