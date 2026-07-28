from spacy.matcher import Matcher, PhraseMatcher
from text_reconstruction import reconstruct_text
from process_and_display import nlp_stanza

def b3d4_2(doc):
    """
    B3.4-2: imperativas atenuadas / intensificadas - por favor / se faz favor
    e.g. Dá-me esse livro, por favor.
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"POS": {"IN": ["VERB", "AUX"]}, "DEP": {"IN": ["root", "cop"]}},
        {"LOWER": {"NOT_IN": ["por", "faz"]}, "IS_SENT_START": False, "OP": "*"},
        {"LOWER": {"IN": ["por", "faz"]}, "IS_SENT_START": False},
        {"LOWER": "favor"}
        ]
    
    pattern_reverse = [
        {"LOWER": {"IN": ["por", "faz"]}},
        {"LOWER": "favor"},
        {"LOWER": ","},
        {"POS": {"IN": ["VERB", "AUX"]}, "DEP": {"IN": ["root", "cop"]}}
    ]

    matcher.add("b3d4_2_A2", [pattern, pattern_reverse])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d4_3(doc):
    """
    B3.4_3: imperativa - particípio passado
    e.g., Sentados!
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
            "IS_SENT_START": True
        },
        {"LOWER": "!"}
    ]

    pattern_adj = [
        {
            "POS": "ADJ",
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
            "IS_SENT_START": True
        },
        {"LOWER": "!"}
    ]

    pattern_adv = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
            "IS_SENT_START": True
        },
        {"LOWER": ","},
        {"POS": "ADV"},
        {"LOWER": "!"}
    ]

    pattern_adj_adv = [
        {
            "POS": "ADJ",
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
            "IS_SENT_START": True
        },
        {"LOWER": ","},
        {"POS": "ADV"},
        {"LOWER": "!"}
    ]

    pattern_favor = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"}
        },
        {"LOWER": {"NOT_IN": ["faz", "por"]}, "OP": "*", "IS_SENT_START": False},
        {"LOWER": {"IN": ["faz", "por"]}, "IS_SENT_START": False},
        {"LOWER": "favor", "IS_SENT_START": False}
    ]

    pattern_adj_favor = [
        {
            "POS": "ADJ",
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"}
        },
        {"LOWER": {"NOT_IN": ["faz", "por"]}, "OP": "*", "IS_SENT_START": False},
        {"LOWER": {"IN": ["faz", "por"]}, "IS_SENT_START": False},
        {"LOWER": "favor", "IS_SENT_START": False}
    ]

    matcher.add("b3d4_3_B1", [pattern, pattern_adj, pattern_adv, pattern_adj_adv,
                              pattern_favor, pattern_adj_favor])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d4_5(doc):
    """
    B3.4-5: imperativas atenuadas / intensificadas - se não se importa
    e.g., Traga-me uma garrafa de água, se não se importa.
    Level: B1
    """
    matcher = PhraseMatcher(doc.vocab, attr="NORM")

    patterns = list(nlp_stanza.pipe(["se nao se importa"]))

    matcher.add("b3d4_5_B1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d4_10(doc):
    """
    B3.4_10: imperativa direta - gerúndio
    e.g., Andando!
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": "VerbForm=Ger",
            "IS_SENT_START": True
        },
        {"LOWER": "!"}
    ]


    pattern_adv = [
        {
            "MORPH": "VerbForm=Ger",
            "IS_SENT_START": True
        },
        {"LOWER": ","},
        {"POS": "ADV"},
        {"LOWER": "!"}
    ]

    matcher.add("b3d4_10_C1", [pattern, pattern_adv])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]