from spacy.matcher import PhraseMatcher, Matcher, DependencyMatcher
from text_reconstruction import reconstruct_text
from process_and_display import nlp_small

def a5d2_01(doc):
    """
    A5.2-01: Demonstrativos - forma - variação em género e número
    e.g., este, esta, estes, estas; esse, essa, esses, essas; aquele, aquela, aqueles, aquelas
    Level: A1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ['este', 'esta', 'estes', 'estas',
                                  'esse', 'essa', 'esses', 'essas',
                                  'aquele', 'aquela', 'aqueles', 'aquelas']},
                "POS": "PRON"}
            ]

    matcher.add("a5d2_01_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a5d2_02(doc):
    """
    A5.2-02: Demonstrativos - forma - invariáveis
    e.g., isto, isso, aquilo
    Level: A1
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    patterns = list(nlp_small.pipe(["isso", "isto", "aquilo"]))

    matcher.add("a5d2_02_A1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a5d2_02.REQUIRES_SPACY = True

def a5d2_1(doc):
    """
    A5.2-1: Demonstrativos - forma - contração com preposições
    e.g., de - deste, desse, daquele (e variantes); em - este, nesse, naquele (e variantes); a - àquele (e variantes)
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["deste", "desse", "daquele",
                                    "desta", "dessa", "daquela",
                                    "destes", "desses", "daqueles",
                                    "destas", "dessas", "daquelas",
                                    "neste", "nesse", "naquele",
                                    "nesta", "nessa", "naquela",
                                    "nestes", "nesses", "naqueles",
                                    "nestas", "nessas", "naquelas",
                                    "àquele", "àquela", "àqueles", "àquelas",
                                    "àquilo", "àquela"]},
                "POS": "PRON"}
            ]

    matcher.add("a5d2_1_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a5d2_1.REQUIRES_SPACY = True

def a5d2_2(doc):
    """
    A5.2-2: Demonstrativos - uso / valor - uso deítico (reforço com advérbios)
    e.g., este aqui, aquele ali
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {}, # For contractions
                {"LEMMA": {"IN": ["este", "esse", "aquele"]}, "IS_SENT_START": False},
                {"LEMMA": {"IN": ["aqui", "ali", "lá", "aí"]}}
            ]
    
    pattern_alt = [
                {"LEMMA": {"IN": ["este", "esse", "aquele"]}, "IS_SENT_START": True},
                {"LEMMA": {"IN": ["aqui", "ali", "lá", "aí"]}}
            ]

    matcher.add("a5d2_2_A2", [pattern, pattern_alt])

    matches = matcher(doc)

    filtered_matches = []
    for match_id, start, end in matches:
        # Check if the first token is a relevant preposition or matches the expected first relevant token
        first_token = doc[start]
        if (first_token.lower_ in ["em", "de", "por", "a", "este", "esse", "aquele"] or 
            first_token.lemma_ in ["este", "esse", "aquele"]):
            # Keep the full match
            filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
        else:
            # Exclude the first token from the match
            if end - start > 1:  # Ensure we have more than one token
                filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start+1:end]), start+1, end))
    
    return filtered_matches

def a5d2_3(doc):
    """
    A5.2-3: Demonstrativos - uso / valor - uso deítico (reforço com mesmo)
    e.g., Isso mesmo.
    Level: B1
    """
    matcher = DependencyMatcher(doc.vocab)

    pattern = [
    {
        "RIGHT_ID": "pronoun",
        "RIGHT_ATTRS": {"LOWER": {"IN": ['este', 'esta', 'estes', 'estas',
                                  'esse', 'essa', 'esses', 'essas',
                                  'aquele', 'aquela', 'aqueles', 'aquelas',
                                  "isso", "isto", "aquilo"]}}
    },
    {
        "LEFT_ID": "pronoun",
        "RIGHT_ID": "mesmo",
        "RIGHT_ATTRS": {"LEMMA": "mesmo"},
        "REL_OP": "<"
    }
    ]

    pattern_reversed = [
    {
        "RIGHT_ID": "pronoun",
        "RIGHT_ATTRS": {"LOWER": {"IN": ['este', 'esta', 'estes', 'estas',
                                  'esse', 'essa', 'esses', 'essas',
                                  'aquele', 'aquela', 'aqueles', 'aquelas',
                                  "isso", "isto", "aquilo"]}}
    },
    {
        "LEFT_ID": "pronoun",
        "RIGHT_ID": "mesmo",
        "RIGHT_ATTRS": {"LEMMA": "mesmo"},
        "REL_OP": ">"
    }
    ]

    matcher.add("a5d2_3_B1", [pattern, pattern_reversed])

    matches = matcher(doc)

    demonstratives = {'este', 'esta', 'estes', 'estas',
                      'esse', 'essa', 'esses', 'essas',
                      'aquele', 'aquela', 'aqueles', 'aquelas',
                      'isso', 'isto', 'aquilo'}

    results = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1

        # When the demonstrative is a contracted form, Stanza splits it into a
        # preposition token + pronoun token (e.g. "naquele" -> "em" + "aquele").
        # If the pronoun starts the span, that preposition sits just before it and
        # outside the span, so reconstruct_text can't rebuild the contraction and
        # would return "aquele mesmo" instead of "naquele mesmo". Pull the leading
        # preposition into the span so the contraction is reconstructed correctly.
        if doc[start].lower_ in demonstratives and start > 0 and \
                doc[start - 1].pos_ == "ADP" and doc[start - 1].lower_ in {"de", "em", "a"}:
            start -= 1

        tokens_in_span = doc[start:end]

        text = reconstruct_text(tokens_in_span)

        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def a5d2_4(doc):
    """
    A5.2-4: "Demonstrativos - uso / valor - uso anafórico - referente catafórico"
    e.g., Ouve isto: vão aumentar os impostos.
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {},
                {"LOWER": {"IN": ["isto", "este"]}},
                {"LOWER": ":"}
            ]

    matcher.add("a5d2_4_B2", [pattern])

    matches = matcher(doc)

    filtered_matches = []
    for match_id, start, end in matches:
        # Check if the first token is a relevant preposition or matches the expected first relevant token
        first_token = doc[start]
        if first_token.lower_ in ["em", "de", "por", "a", "isto"]:
            # Keep the full match
            filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
        else:
            # Exclude the first token from the match
            if end - start > 1:  # Ensure we have more than one token
                filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start+1:end]), start+1, end))
    
    return filtered_matches


def a5d2_5(doc):
    """
    A5.2-5: Demonstrativos - forma - outras formas demonstrativas
    e.g., mesmo, mesma, mesmos, mesmas; tal, tal, tais, tais
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": {"IN": ["tal", "tais"]}, "POS": {"IN": ["ADJ", "DET"]}}
                # "Mesmo" was excluded because it wasn't deemed too difficult "Ao mesmo tempo"
            ]

    matcher.add("a5d2_5_C1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]