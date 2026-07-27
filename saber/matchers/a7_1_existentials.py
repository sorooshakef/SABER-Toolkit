from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text

def a7d1_1(doc):
    """
    A7.1-1: Existenciais - forma - variação em género e número - A2
    e.g., muito, muita, muitos, muitas; pouco, pouca, poucos, poucas; algum, alguma, alguns, algumas
    Level: A2

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"POS": "DET", "LEMMA": {"IN": ["muito", "pouco", "algum"]}},
            ]

    matcher.add("a7d1_1_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a7d1_5(doc):
    """
    A7.1-5: Existenciais - forma - variação em género e número - B1
    e.g., tanto, tanta, tantos, tantas; vários, várias
    Level: B1

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"POS": "DET", "NORM": {"IN": ["tantos", "tanto", "tanta", "tantas", "varios", "varias"]}},
            ]

    matcher.add("a7d1_5_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d1_9(doc):
    """
    A7.1-9: Existenciais - distribuição - possibilidade de ocorrência com artigos, com alteração de valor - pouco, tanto
    e.g., Li poucos livros para fazer este trabalho / Li uns poucos de livros para fazer este trabalho.
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"MORPH": {"IS_SUPERSET": ["PronType=Art"]}},
                {"NORM": {"IN": ["tantos", "tantas", "tanto", "tanta",
                                "poucos", "poucas", "pouco", "pouca",]}},
                {"LOWER": "de"},
                {"POS": "NOUN"}
            ]
    
    pattern_pron = [
                {"MORPH": {"IS_SUPERSET": ["PronType=Art"]}},
                {"NORM": {"IN": ["tantos", "tantas", "tanto", "tanta",
                                "poucos", "poucas", "pouco", "pouca",]}},
                {"POS": "PRON"}
            ]
    

    matcher.add("a7d1_9_B2", [pattern, pattern_pron])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


