from spacy.matcher import Matcher, PhraseMatcher
from text_reconstruction import reconstruct_text
from process_and_display import nlp_small

def a5d4_1(doc):
    """
    A5.4-1: Indefinidos - forma - variação em género e número
    e.g., muito, muita, muitos, muitas; pouco, pouca, poucos, poucas; todo, toda, todos, toda; algum, alguma, alguns, algumas; nenhum, nenhuma, nenhuns, nenhumas
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": {"IN": ["muito", "pouco", "algum", "nenhum", "todo"]}, "POS": "PRON"},
            ]

    matcher.add("a5d4_1_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a5d4_2(doc):
    """
    A5.4-2: Indefinidos - forma - invariáveis
    e.g., tudo, nada, alguém, ninguém
    Level: A2
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    patterns = list(nlp_small.pipe(["tudo", "nada", "alguém", "ninguém"]))

    matcher.add("a5d4_2_A2", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a5d4_2.REQUIRES_SPACY = True

def a5d4_6(doc):
    """
    A5.4-1: Indefinidos - forma - variação em género e número
    e.g., outro, outra, outros, outras; tanto, tanta, tantos, tantas; vários, várias; qualquer, quaisquer
    Level: B1

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": {"IN": ["outro", "tanto", "vários", "qualquer", "várias"]}, "POS": "PRON"}
            ]

    matcher.add("a5d4_6_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a5d4_8(doc):
    """
    A5.4-8: Pronomes - Indefinidos - forma - variação em género e número
    e.g., quanto, quanta, quantos, quantas  
    Level: B2

    Not matching: Too restrictive, but it is necessary to avoid false positives with "quanto" as an interrogative pronoun.
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"NOT_IN": ["a", "de", "com", "por", "para"]}},
                {"LEMMA": "quanto", "MORPH": {"IS_SUPERSET": ["PronType=Rel"]}, "IS_SENT_START": False}
            ]

    matcher.add("a5d4_8_B2", [pattern])

    matches = matcher(doc)

     # Return the last token of every match
    return [(doc.vocab.strings[match_id], doc[end-1].text, end-1, end-1) for match_id, start, end in matches]

def a5d4_9(doc):
    """
    A5.4-9: Indefinidos - forma - invariáveis
    e.g., algo, outrem
    Level: B2
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    patterns = list(nlp_small.pipe(["algo", "outrem"]))

    matcher.add("a5d4_9_B2", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a5d4_9.REQUIRES_SPACY = True