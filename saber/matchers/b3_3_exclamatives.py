from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text

def b3d3_5(doc):
    """
    B3.3-5: exclamativa parcial - elemento exclamativo + expressões nominais e adjetivais
    e.g. Que medo! Que grande!
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": {"IN": ["que", "quanta", "quanto"]}, "IS_SENT_START": True},
        {"POS": {"IN": ["NOUN", "ADJ"]}},
        {"LOWER": {"NOT_IN": ["que"]}, "OP": "*", "IS_SENT_START": False, "POS": {"NOT_IN": ["VERB", "AUX"]}},
        {"LOWER": {"IN": ["!", "."]}}
        ]
    
    pattern_e = [
        {"POS": "CCONJ", "IS_SENT_START": True},
        {"LOWER": {"IN": ["que", "quanta", "quanto"]}},
        {"POS": {"IN": ["NOUN", "ADJ"]}},
        {"LOWER": {"NOT_IN": ["que"]}, "OP": "*", "IS_SENT_START": False, "POS": {"NOT_IN": ["VERB", "AUX"]}},
        {"LOWER": {"IN": ["!", "."]}}
        ]

    matcher.add("b3d3_5_A2", [pattern, pattern_e])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d3_6(doc):
    """
    B3.3-6: exclamativa parcial - com tanto / tão + que
    e.g. Tão bem que ela canta! Tanta fome que eu tenho!
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": {"IN": ["tanto", "tanta", "tão"]}, "IS_SENT_START": True},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV"]}},
        {"LOWER": "que"}
        ]
    
    pattern_e = [
        {"POS": "CCONJ", "IS_SENT_START": True},
        {"LOWER": {"IN": ["tanto", "tanta", "tão"]}},
        {"POS": {"IN": ["NOUN", "ADJ", "ADV"]}},
        {"LOWER": "que"}
        ]

    matcher.add("b3d3_6_B1", [pattern, pattern_e])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d3_8(doc):
    """
    B3.3-8: exclamativa parcial - elemento exclamativo + que + oração
    e.g. Que bom que foi encontrar-te!
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": {"IN": ["que", "quanta", "quanto"]}, "IS_SENT_START": True},
        {"POS": "ADJ", "OP": "?"},
        {"POS": {"IN": ["NOUN", "ADJ"]}},
        {"LOWER": "que"}
        ]
    
    pattern_e = [
        {"POS": "CCONJ", "IS_SENT_START": True},
        {"LOWER": {"IN": ["que", "quanta", "quanto"]}},
        {"POS": "ADJ", "OP": "?"},
        {"POS": {"IN": ["NOUN", "ADJ"]}},
        {"LOWER": "que"}
        ]
    
    pattern_adv = [
        {"POS": "ADV", "IS_SENT_START": True},
        {"LOWER": ",", "OP": "?"},
        {"LOWER": {"IN": ["que", "quanta", "quanto"]}},
        {"POS": "ADJ", "OP": "?"},
        {"POS": {"IN": ["NOUN", "ADJ"]}},
        {"LOWER": "que"}
        ]

    matcher.add("b3d3_8_B1", [pattern, pattern_e, pattern_adv])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d3_9(doc):
    """
    B3.3-9: elemento exclamativo + oração
    e.g. Como ele corre depressa! Quantas mentiras a Ana inventa!
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LEMMA": {"IN": ["como", "quanto", "que"]}, "IS_SENT_START": True},
        {"LOWER": {"NOT_IN": ["que"]}, "DEP": {"NOT_IN": ["nsubj"]}, "OP": "*", "IS_SENT_START": False},
        {"DEP": "nsubj", "IS_SENT_START": False},
        {"LOWER": {"NOT_IN": ["que"]}, "POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False},
        {"POS": {"IN": ["VERB", "AUX"]}, "DEP": "root", "IS_SENT_START": False},
        {"LOWER": {"NOT_IN": ["?"]}, "OP": "*", "IS_SENT_START": False},
        {"LOWER": {"IN": ["!", "."]}}
        ]

    matcher.add("b3d3_9_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d3_10(doc):
    """
    B3.3-10: exclamativa total - com diferentes marcadores - é que
    e.g. Isso é que era bom!
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": {"NOT_IN": ["que", "onde", "quando", "quem", "como"]}},
        {"LOWER": "é", "MORPH": {"IS_SUPERSET": ["ExtPos=INTJ"]}},
        {"LOWER": "que", "DEP": "fixed"}
        ]
    
    pattern_alt = [
        {"LOWER": {"NOT_IN": ["que", "onde", "quando", "quanto", "quanta", "quantas", "quantos", "quem", "como"]}},
        {"LOWER": "é", "DEP": "fixed"},
        {"LOWER": "que", "DEP": "fixed"}
        ]

    matcher.add("b3d3_10_B2", [pattern, pattern_alt])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d3_11(doc):
    """
    B3.3-11: exclamativa total - se condicional + imperf. conjuntivo
    e.g. Se ele trabalhasse mais!
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": "se", "IS_SENT_START": True},
        {"DEP": {"NOT_IN": ["root", "cop"]}, "IS_SENT_START": False, "OP": "*", "NORM": {"NOT_IN": ["nao"]}},
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"},
            "DEP": {"IN":["root", "cop"]}
        },
        {"POS": {"NOT_IN": ["PUNCT"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["!", "."]}}
        ]
    
    pattern_cconj = [
        {"POS": "CCONJ", "IS_SENT_START": True},
        {"LOWER": "se", "IS_SENT_START": False},
        {"DEP": {"NOT_IN": ["root", "cop"]}, "IS_SENT_START": False, "OP": "*", "NORM": {"NOT_IN": ["nao"]}},
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"},
            "DEP": {"IN":["root", "cop"]}
        },
        {"POS": {"NOT_IN": ["PUNCT"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["!", "."]}}
        ]
    

    pattern_ad_hoc = [
        {"LOWER": "se", "IS_SENT_START": True},
        {"DEP": {"NOT_IN": ["root", "cop"]}, "IS_SENT_START": False, "OP": "*", "NORM": {"NOT_IN": ["nao"]}},
        {"NORM": {"IN": ["perdessemos", "ouvisses", "interrompesses", "ladrasse"]}},
        {"POS": {"NOT_IN": ["PUNCT"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["!", "."]}}
        ]
    
    pattern_cconj_ad_hoc = [
        {"POS": "CCONJ", "IS_SENT_START": True},
        {"LOWER": "se", "IS_SENT_START": False},
        {"DEP": {"NOT_IN": ["root", "cop"]}, "IS_SENT_START": False, "OP": "*", "NORM": {"NOT_IN": ["nao"]}},
        {"NORM": {"IN": ["perdessemos", "ouvisses", "interrompesses", "ladrasse"]}},
        {"POS": {"NOT_IN": ["PUNCT"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["!", "."]}}
        ]

    matcher.add("b3d3_11_B2", [pattern, pattern_cconj,
                               pattern_ad_hoc, pattern_cconj_ad_hoc])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d3_15(doc):
    """
    B3.3-15: exclamativa parcial - se condicional + não com valor de reforço de afirmação
    e.g. Se ela não chegasse sempre atrasada!
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": "se", "IS_SENT_START": True},
        {"NORM": {"NOT_IN": ["nao"]}, "OP": "*", "IS_SENT_START": False},
        {"NORM": "nao"},
        {"DEP": {"NOT_IN": ["root", "cop"]}, "IS_SENT_START": False, "OP": "*"},
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"},
            "DEP": {"IN": ["root", "cop"]}
        },
        {"POS": {"NOT_IN": ["PUNCT"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["!", "."]}}
        ]
    

    pattern_cconj = [
        {"POS": "CCONJ", "IS_SENT_START": True},
        {"LOWER": "se", "IS_SENT_START": False},
        {"NORM": {"NOT_IN": ["nao"]}, "OP": "*", "IS_SENT_START": False},
        {"NORM": "nao"},
        {"DEP": {"NOT_IN": ["root", "cop"]}, "IS_SENT_START": False, "OP": "*"},
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"},
            "DEP": {"IN": ["root", "cop"]}
        },
        {"POS": {"NOT_IN": ["PUNCT"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["!", "."]}}
        ]
    

    pattern_ad_hoc = [
        {"LOWER": "se", "IS_SENT_START": True},
        {"NORM": {"NOT_IN": ["nao"]}, "OP": "*", "IS_SENT_START": False},
        {"NORM": "nao"},
        {"DEP": {"NOT_IN": ["root", "cop"]}, "IS_SENT_START": False, "OP": "*"},
        {"NORM": {"IN": ["perdessemos", "ouvisses", "interrompesses", "ladrasse"]}},
        {"POS": {"NOT_IN": ["PUNCT"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["!", "."]}}
        ]
    

    pattern_cconj_ad_hoc = [
        {"POS": "CCONJ", "IS_SENT_START": True},
        {"LOWER": "se", "IS_SENT_START": False},
        {"NORM": {"NOT_IN": ["nao"]}, "OP": "*", "IS_SENT_START": False},
        {"NORM": "nao"},
        {"DEP": {"NOT_IN": ["root", "cop"]}, "IS_SENT_START": False, "OP": "*"},
        {"NORM": {"IN": ["perdessemos", "ouvisses", "interrompesses", "ladrasse"]}},
        {"POS": {"NOT_IN": ["PUNCT"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["!", "."]}}
        ]

    matcher.add("b3d3_15_C1", [pattern, pattern_cconj,
                               pattern_ad_hoc, pattern_cconj_ad_hoc])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]



