from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text

def a6d4_1(doc):
    """
    A6.4-1: Indefinidos - forma - variação em género e número
    e.g., ooutro, outra, outros, outras; certo, certa, certos, certas
    Level: B1

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": {"IN": ["outro", "certo", "certos"]}, "POS": "DET"}
            ]

    matcher.add("a6d4_1_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

# The two constructions below are classified at level B2, but that's inconsistent with other similar constructions and
# with my intuition as a learner.

# def a6d4_4(doc):
#     """
#     A6.4-4: "Indefinidos - distribuição - possibilidade de ocorrência com outros determinantes - quantificador + artigo + possessivo"
#     e.g., todas as vossas ideias
#     Level: B2

#     """
#     matcher = Matcher(doc.vocab)

#     pattern = [
#                 {"POS": {"IN": ["DET", "PRON"]}},  # "PRON" added to accommodate "alguns dos seus conselhos"
#                 {"LOWER": "de", "OP": "?"},
#                 {"MORPH": {"IS_SUPERSET": ["PronType=Art"]}},
#                 {"MORPH": {"IS_SUPERSET": ["PronType=Prs"]}},
#                 {"POS": "NOUN"}
#             ]

#     matcher.add("a6d4_4_B2", [pattern])

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


# def a6d4_5(doc):
#     """
#     A6.4-5: "Indefinidos - distribuição - possibilidade de ocorrência com outros determinantes - artigo + possessivo + quantificador"
#     e.g., os meus três primos; a sua pouca experiência
#     Level: B2

#     """
#     matcher = Matcher(doc.vocab)

#     pattern = [
#                 {"MORPH": {"IS_SUPERSET": ["PronType=Art"]}},
#                 {"MORPH": {"IS_SUPERSET": ["PronType=Prs"]}},
#                 {"POS": {"IN": ["DET", "NUM"]}},
#                 {"POS": "NOUN"}
#             ]

#     matcher.add("a6d4_5_B2", [pattern])

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]