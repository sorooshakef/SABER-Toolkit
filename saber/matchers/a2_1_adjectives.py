from spacy.matcher import Matcher, DependencyMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def a2d1_4(doc):
    """
    A2.1-4: género - adjetivos invariáveis quanto ao género - e.g., pobre, amável, jovem, simples, feliz
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    adj_list = ["pobre", "jovem", "simples", "feliz", "especial", "infeliz", "hipócrita", "hipócrito", "indigina", "lamecha", "lorpa",
                "otimista", "palerma", "simplista", "brilhante", "forte", "quente", "torpe", "incolor", "inferior", "regular",
                "superior", "banal", "cruel", "final", "normal", "ruim", "choné", "gagá", "doente", "inocente", "presente", "internacional",
                "nacional", "influente", "grande"]
    # "hipócrito" is added due to the error of the lemmatizer.

    adj_recog_noun = ["indiferente", "diferente", "independente", "dependente", "carente", "ausente", "contente",
                      "competente", "incompetente", "inconsciente", "consciente", "incoerente", "coerente", "inerte", "competente",
                      "incolor"]
    
    pattern_noun_but_adj = [
        {"LEMMA": {"IN": adj_recog_noun}},
    ]

    pattern_vel = [
        {"POS": "ADJ", "LEMMA": {"REGEX": r"(vel)$"}}
    ]

    pattern = [
    {
        "LEMMA": {"IN": adj_list}, "POS": "ADJ"
    }
    ]
    matcher.add("a2d1_4_A2", [pattern, pattern_vel, pattern_noun_but_adj])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a2d1_5(doc):
    """
    A2.1-5: número - adjetivos invariáveis quanto ao número - e.g., simples, piegas
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    adj_list = ["simples", "piegas"]

    pattern = [
    {
        "LOWER": {"IN": adj_list}, "POS": "ADJ"
    }
    ]

    matcher.add("a2d1_5_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a2d1_6(doc):
    """
    A2.1-6: grau - comparativo - e.g., mais alto (do) que; tão alto como / quanto;  menos alto (do) que
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        # "por" so that we don't match "Esses teus argumentos, por mais convincentes que pareçam, não me demovem da minha opinião."
        {"LOWER": {"NOT_IN": ["o", "a", "por"]}},
        {"LOWER": {"IN": ["mais", "menos"]}},
        {"POS": "ADJ"},
        {"LOWER": {"NOT_IN": ["que", ","]}, "IS_SENT_START": False, "OP": "{,5}"},
        {"LOWER": "que", "IS_SENT_START": False},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "VERB", "PROPN", "PUNCT"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"POS": {"IN": ["NOUN", "PRON", "VERB", "PROPN"]}, "IS_SENT_START": False}
    ]

    pattern_2 = [
        {"LOWER": "tão"},
        {"POS": "ADJ"},
        {"LOWER": {"IN": ["como", "quanto"]}},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "VERB", "PROPN", "PUNCT"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"POS": {"IN": ["NOUN", "PRON", "VERB", "PROPN"]}, "IS_SENT_START": False}
    ]

    # Patterns 3 and 4 are similar to above but with a participial adjective.

    pattern_3 = [
        {"LOWER": {"NOT_IN": ["o", "a"]}},
        {"LOWER": {"IN": ["mais", "menos"]}},
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
        {"LOWER": {"NOT_IN": ["que", ","]}, "IS_SENT_START": False, "OP": "{,5}"},
        {"LOWER": "que", "IS_SENT_START": False},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "VERB", "PROPN", "PUNCT"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"POS": {"IN": ["NOUN", "PRON", "VERB", "PROPN"]}, "IS_SENT_START": False}
    ]

    pattern_4 = [
        {"LOWER": "tão"},
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
        {"LOWER": {"IN": ["como", "quanto"]}},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "VERB", "PROPN", "PUNCT"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"POS": {"IN": ["NOUN", "PRON", "VERB", "PROPN"]}, "IS_SENT_START": False}
    ]

    matcher.add("a2d1_6_A2", [pattern_1, pattern_2, pattern_3, pattern_4])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a2d1_7(doc):
    """
    A2.1-7: grau - superlativo relativo - e.g., o mais alto de; o menos alto de
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        {}, # For possible contractions
        {"LOWER": {"IN": ["o", "a"]}, "IS_SENT_START": False},
        {"POS": "NOUN", "OP": "?", "IS_SENT_START": False},
        {"LOWER": {"IN": ["mais", "menos"]}, "IS_SENT_START": False},
        {"POS": "ADJ", "IS_SENT_START": False},
        {"LOWER": {"NOT_IN": ["que"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"LOWER": "de", "IS_SENT_START": False},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "PROPN"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "IS_SENT_START": False}
    ]

    pattern_2 = [
        {},
        {"LOWER": {"IN": ["o", "a"]}, "IS_SENT_START": False},
        {"LOWER": {"IN": ["mais", "menos"]}, "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "IS_SENT_START": False},
        {"LOWER": {"NOT_IN": ["que"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"LOWER": "de", "IS_SENT_START": False},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "PROPN"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "IS_SENT_START": False}
    ]

    pattern_3 = [
        {"LOWER": {"IN": ["o", "a"]}, "IS_SENT_START": True},
        {"POS": "NOUN", "OP": "?", "IS_SENT_START": False},
        {"LOWER": {"IN": ["mais", "menos"]}, "IS_SENT_START": False},
        {"POS": "ADJ", "IS_SENT_START": False},
        {"LOWER": {"NOT_IN": ["que"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"LOWER": "de", "IS_SENT_START": False},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "PROPN"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "IS_SENT_START": False}
    ]

    pattern_4 = [
        {"LOWER": {"IN": ["o", "a"]}, "IS_SENT_START": True},
        {"LOWER": {"IN": ["mais", "menos"]}, "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "IS_SENT_START": False},
        {"LOWER": {"NOT_IN": ["que"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"LOWER": "de", "IS_SENT_START": False},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "PROPN"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "IS_SENT_START": False}
    ]

    matcher.add("a2d1_7_A2", [pattern_1, pattern_2, pattern_3, pattern_4])

    matches = matcher(doc)

    filtered_matches = []
    for match_id, start, end in matches:
        # Check if the first token is a relevant preposition or matches the expected first relevant token
        first_token = doc[start]
        if first_token.lower_ in ["em", "de", "por", "a", "o"]:
            # Keep the full match
            filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
        else:
            # Exclude the first token from the match
            if end - start > 1:  # Ensure we have more than one token
                filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start+1:end]), start+1, end))
    
    return filtered_matches

def a2d1_8(doc):
    """
    A2.1-8: grau - comparativos e superlativos irregulares frequentes - bom, mau, grande, pequeno, alto, baixo; Este livro é bom / melhor (*mais bom) / ótimo.
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    # For some adjectives, for example "pior," the lemma of the singular form is different from that of the
    # plural form, which makes us resort to including both forms in the list.

    adj_list = ["maior", "maiores", "melhor", "melhores", "pior", "piores", "menor", "menores", "superior", "superiores",
                "inferior", "inferiores"]

    pattern = [
        {"LOWER": {"IN": adj_list}, "POS": "ADJ"},
    ]

    matcher.add("a2d1_8_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a2d1_9(doc):
    """
    A2.1-9: género - feminino de adjetivos compostos - e.g. luso-africano, luso-africana
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "TEXT": {"REGEX": "[A-Za-z]+(-[A-Za-z]+){1,2}"},
                    "MORPH": {"IS_SUPERSET": ["Gender=Fem"]},
                    "POS": "ADJ"
                }
            ]

    matcher.add("a2d1_9_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a2d1_10(doc):
    """
    A2.1-10: número - plural de adjetivos compostos - verde-claro, verde-claros
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "TEXT": {"REGEX": "[A-Za-z]+(-[A-Za-z]+){1,2}"},
                    "MORPH": {"IS_SUPERSET": ["Number=Plur"]},
                    "POS": "ADJ"
                }
            ]

    pattern_1 = [
                {
                    "POS": "NOUN",
                    "MORPH": {"IS_SUPERSET": ["Number=Plur"]}
                },
                {
                    "TEXT": {"REGEX": "[A-Za-z]+(-[A-Za-z]+){1,2}"},
                    "POS": "ADJ"
                }
            ]

    matcher.add("a2d1_10_B1", [pattern, pattern_1])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a2d1_11(doc):
    """
    A2.1-11: grua - superlativo (-íssimo) - e.g., altíssimo
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "NORM": {"REGEX": "\\b[A-Za-z]+(?:issimo|issima|cilimo|cilima)(s?)\\b"},
                    "POS": "ADJ"
                }
            ]

    matcher.add("a2d1_11_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a2d1_12(doc):
    """
    A2.1-12: grau - outros processos
    e.g., um rapaz deveras inteligente; uma festa super animada; a casa toda desarrumada
    Level: B2
    """

    matcher = DependencyMatcher(nlp_stanza.vocab)

    word_list = ["deveras", "super", "toda", "todo", "hiper",
                  "ultra", "mega", "bem", "imensamente", "extremamente",
                  "bastante", "demasiadamente", "excessivamente", "incrivelmente", "notavelmente"]
    
    black_list = ["contigo", "connosco", "convosco"]

    pattern = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": word_list}
            }
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
                "POS": "ADJ",
                "LOWER": {"NOT_IN": black_list}
            },
        "REL_OP": "<"
        }
    ]

    pattern_alt = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": word_list}
            }
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]} ,
                "LOWER": {"NOT_IN": black_list}
            },
        "REL_OP": "<"
        }
    ]
    
    matcher.add("a2d1_12_B2", [pattern, pattern_alt])

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
