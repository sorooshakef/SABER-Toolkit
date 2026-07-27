from spacy.matcher import Matcher, DependencyMatcher
from text_reconstruction import reconstruct_text

def a6d2_3(doc):
    """
    A6.2-3: Demonstrativos - concordância - em género e número com o nome
    e.g., este livro, esta caneta, estes livros, estas canetas
    Level: A1
    """
    matcher = Matcher(doc.vocab)

    pattern_este = [
                {},
                {"LOWER": {"IN": ["este", "esse", "aquele"]}, "POS": "DET", "IS_SENT_START": False},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Sing"}
            ]
    
    pattern_esta = [
                {},
                {"LOWER": {"IN": ["esta", "essa", "aquela"]}, "POS": "DET", "IS_SENT_START": False},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Sing"}
            ]
    
    pattern_estes = [
                {},
                {"LOWER": {"IN": ["estes", "esses", "aqueles"]}, "POS": "DET", "IS_SENT_START": False},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Plur"}
            ]
    
    pattern_estas = [
                {},
                {"LOWER": {"IN": ["estas", "essas", "aquelas"]}, "POS": "DET", "IS_SENT_START": False},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Plur"}
            ]
    
    pattern_este_alt = [
                {"LOWER": {"IN": ["este", "esse", "aquele"]}, "POS": "DET", "IS_SENT_START": True},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Sing"}
            ]
    
    pattern_esta_alt = [
                {"LOWER": {"IN": ["esta", "essa", "aquela"]}, "POS": "DET", "IS_SENT_START": True},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Sing"}
            ]
    
    pattern_estes_alt = [
                {"LOWER": {"IN": ["estes", "esses", "aqueles"]}, "POS": "DET", "IS_SENT_START": True},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Plur"}
            ]
    
    pattern_estas_alt = [
                {"LOWER": {"IN": ["estas", "essas", "aquelas"]}, "POS": "DET", "IS_SENT_START": True},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Plur"}
            ]

    matcher.add("a6d2_3_A1", [pattern_este, pattern_esta, pattern_estes, pattern_estas,
                              pattern_este_alt, pattern_esta_alt, pattern_estes_alt, pattern_estas_alt])

    matches = matcher(doc)

    filtered_matches = []
    for match_id, start, end in matches:
        # Check if the first token is a relevant preposition or matches the expected first relevant token
        first_token = doc[start]
        if (first_token.lower_ in ["em", "de", "por", "a", "este", "esse", "aquele", "esta", "essa", "aquela", "estes", "esses", "aqueles", "estas", "essas", "aquelas"] or 
            first_token.pos_ == "DET"):
            # Keep the full match
            filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
        else:
            # Exclude the first token from the match
            if end - start > 1:  # Ensure we have more than one token
                filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start+1:end]), start+1, end))
    
    return filtered_matches

def a6d2_4(doc):
    """
    A6.2-4: Demonstrativos - forma - contração com preposições
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
                "POS": "ADP"}
            ]

    matcher.add("a6d2_4_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a6d2_4.REQUIRES_SPACY = True


def a6d2_13(doc):
    """
    A6.2-13: Demonstrativos - posição - depois do nome - em relativas apositivas
    e.g., A pobreza é um grave problema mundial, problema este que diz respeito a todos nós.
    Level: C2
    """
    matcher = DependencyMatcher(doc.vocab)

    pattern = [
        {
            "RIGHT_ID": "problema",
            "RIGHT_ATTRS": {
                "DEP": "appos"
                }
        },
        {
            "LEFT_ID": "problema",
            "RIGHT_ID": "este",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["PronType=Dem"]},
                "DEP": "det"
                },
            "REL_OP": ">+"             
        },
        {
            "LEFT_ID": "problema",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": "acl:relcl"
                },
            "REL_OP": ">++"             
        }
    ]

    pattern_alt = [
        {
            "RIGHT_ID": "problema",
            "RIGHT_ATTRS": {
                "DEP": "appos"
                }
        },
        {
            "LEFT_ID": "problema",
            "RIGHT_ID": "este",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["PronType=Dem"]},
                "DEP": "det"},
            "REL_OP": ">+"             
        },
        {
            "LEFT_ID": "este",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": "acl:relcl"
                },
            "REL_OP": ">++"
        }
    ]

    # Variants where the demonstrative ("este") is itself parsed as the
    # appositive, and its head is simply a NOUN.
    pattern_det = [
        {
            "RIGHT_ID": "este",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["PronType=Dem"]},
                "DEP": "appos"
                }
        },
        {
            "LEFT_ID": "este",
            "RIGHT_ID": "problema",
            "RIGHT_ATTRS": {
                "POS": "NOUN"
                },
            "REL_OP": "<"
        },
        {
            "LEFT_ID": "problema",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": "acl:relcl"
                },
            "REL_OP": ">++"
        }
    ]

    pattern_det_alt = [
        {
            "RIGHT_ID": "este",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["PronType=Dem"]},
                "DEP": "appos"
                }
        },
        {
            "LEFT_ID": "este",
            "RIGHT_ID": "problema",
            "RIGHT_ATTRS": {
                "POS": "NOUN"
                },
            "REL_OP": "<"
        },
        {
            "LEFT_ID": "este",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": "acl:relcl"
                },
            "REL_OP": ">++"
        }
    ]

    pattern_cujo = [
        {
            "RIGHT_ID": "problema",
            "RIGHT_ATTRS": {
                "POS": "NOUN"
                }
        },
        {
            "LEFT_ID": "problema",
            "RIGHT_ID": "este",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["PronType=Dem"]},
                "DEP": "det"
                },
            "REL_OP": "."             
        },
        {
            "LEFT_ID": "este",
            "RIGHT_ID": "cujo",
            "RIGHT_ATTRS": {
                "LEMMA": "cujo"
                },
            "REL_OP": "$+"             
        }
    ]


    matcher.add("a6d2_13_C2", [pattern, pattern_alt, pattern_det, pattern_det_alt, pattern_cujo])

    matches = matcher(doc)
    
    results = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]
        
        text = reconstruct_text(tokens_in_span)

        results.append((doc.vocab.strings[match_id], text, start, end))

    return results