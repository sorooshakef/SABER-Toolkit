from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text

def a6d3_6(doc):
    """
    A6.3-6: "Possessivos - posição/distribuição - obrigatoriedade de ocorrência - sem artigo"
    e.g., Meu amigo, como estás?
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"MORPH": {"IS_SUPERSET": ["PronType=Prs"]}, "IS_SENT_START": True},
                {"POS": {"IN": ["NOUN", "PROPN"]}}
            ]

    matcher.add("a6d3_6_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a6d3_7(doc):
    """
    A6.3-7: "Possessivos - posição - depois do nome - em contextos indefinidos"
    e.g., Um amigo meu já me falou deste filme.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": "um"},
                {"POS": "NOUN"},
                {"MORPH": {"IS_SUPERSET": ["PronType=Prs"]}}
            ]
    
    pattern_de = [
                {"LEMMA": "um"},
                {"POS": "NOUN"},
                {"LOWER": "de"},
                {"MORPH": {"IS_SUPERSET": ["PronType=Prs"]}}
            ]

    matcher.add("a6d3_7_B1", [pattern, pattern_de])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

# The two constructions below are classified as B2, but that doesn't seem consistent with other similar constructions
# which are taught at level A2, nor with my own intuition as a learner of Portuguese.

# def a6d3_8(doc):
#     """
#     A6.3-8: "Possessivos - distribuição - possibilidade de ocorrência com outros determinantes - quantificador + demonstrativo"
#     e.g., Todas estas ideias são boas.
#     Level: B2
#     """
#     matcher = Matcher(doc.vocab)

#     pattern = [
#                 {"POS": {"IN": ["DET", "PRON"]}}, # "PRON" to accommodate "mas só algumas destas propostas avançaram para a fase seguinte."
#                 {"LOWER": "de", "OP": "?"},
#                 {"MORPH": {"IS_SUPERSET": ["PronType=Dem"]}},
#                 {"POS": "NOUN"}
#             ]
    
#     matcher.add("a6d3_8_B2", [pattern])

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a6d3_9(doc):
    """
    A6.3-9: "Possessivos - distribuição - possibilidade de ocorrência com outros determinantes - demonstrativo + possessivo"
    e.g., Este meu cão é muito meigo.
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"MORPH": {"IS_SUPERSET": ["PronType=Dem"]}, "DEP": "det"},
                {"MORPH": {"IS_SUPERSET": ["PronType=Prs"]}, "DEP": "det"}
            ]

    matcher.add("a6d3_9_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]