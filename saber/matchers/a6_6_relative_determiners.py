from spacy.matcher import Matcher, PhraseMatcher
from text_reconstruction import reconstruct_text
from process_and_display import nlp_small

def a6d6_1(doc):
    """
    A6.6-1: Relativos - forma - variação em género e número
    e.g., cujo, cuja, cujos, cujas
    Level: B1
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    patterns = list(nlp_small.pipe(["cujo", "cuja", "cujos", "cujas"]))

    matcher.add("a6d6_1_B1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a6d6_1.REQUIRES_SPACY = True


def a6d6_3(doc):
    """
    A6.6-3: Relativos - forma - variação em género e número - com preposições - de cujo, a cujo, com cujo, para cujo
    e.g., Este rapaz, de cujo nome não me recordo, vive na minha rua.
    Level: B2

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["de", "a", "com", "para"]}},
                {"LEMMA": "cujo"}
            ]

    matcher.add("a6d6_3_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]