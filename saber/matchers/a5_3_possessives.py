from spacy.matcher import Matcher, PhraseMatcher, DependencyMatcher
from text_reconstruction import reconstruct_text
from process_and_display import nlp_small

def a5d3_1(doc):
    """
    A5.3-1: "Possessivos - forma - variação em pessoa, género e número - um / vários possuidores"
    e.g., meu, minha, meus, minhas; nosso, nossa, nossos, nossas; teu, tua, teus, tuas; vosso, vossa, vossos, vossas; seu, sua, seus, suas
    Level: A1
    """
    matcher = DependencyMatcher(doc.vocab)

    possessives = ['meu', 'minha', 'meus', 'minhas',
                   'nosso', 'nossa', 'nossos', 'nossas',
                   'teu', 'tua', 'teus', 'tuas',
                   'vosso', 'vossa', 'vossos', 'vossas',
                   'seu', 'sua', 'seus', 'suas']

    # Match a possessive pronoun modifying a noun only when the two agree in
    # gender and number. One pattern per gender x number combination, anchored
    # on the noun (the head) with the possessive pronoun as its child.
    pattern_masc_sing = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": "NOUN",
                "MORPH": {"IS_SUPERSET": ["Gender=Masc", "Number=Sing"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "possessive",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": possessives},
                "MORPH": {"IS_SUPERSET": ["Gender=Masc", "Number=Sing"]}
                },
            "REL_OP": ">"
        }
    ]

    pattern_fem_sing = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": "NOUN",
                "MORPH": {"IS_SUPERSET": ["Gender=Fem", "Number=Sing"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "possessive",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": possessives},
                "MORPH": {"IS_SUPERSET": ["Gender=Fem", "Number=Sing"]}
                },
            "REL_OP": ">"
        }
    ]

    pattern_masc_plur = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": "NOUN",
                "MORPH": {"IS_SUPERSET": ["Gender=Masc", "Number=Plur"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "possessive",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": possessives},
                "MORPH": {"IS_SUPERSET": ["Gender=Masc", "Number=Plur"]}
                },
            "REL_OP": ">"
        }
    ]

    pattern_fem_plur = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": "NOUN",
                "MORPH": {"IS_SUPERSET": ["Gender=Fem", "Number=Plur"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "possessive",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": possessives},
                "MORPH": {"IS_SUPERSET": ["Gender=Fem", "Number=Plur"]}
                },
            "REL_OP": ">"
        }
    ]

    matcher.add("a5d3_1_A1", [pattern_masc_sing, pattern_fem_sing,
                              pattern_masc_plur, pattern_fem_plur])

    matches = matcher(doc)

    results = []
    for match_id, token_ids in matches:
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]

        text = reconstruct_text(tokens_in_span)
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def a5d3_3(doc):
    """
    A5.3-3: Possessivos - forma - formas de desambiguação da 3.ª pessoa
    e.g., dele, dela, deles, delas; Este carro é seu / dele.
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": "de"},
                {"LEMMA": {"IN": ["ele", "eles", "ela", "elas"]}}
            ]

    matcher.add("a5d3_3_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a5d3_5(doc):
    """
    A5.3-5: Possessivos - uso / valor - expressões com possessivos
    e.g., O seu a seu dono. É a cara dele. Cada macaco no seu galho.
    Level: B2
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    expressions = [
        "o seu a seu dono",
        "a cara dele",
        "a seu tempo",
        "cada coisa a seu tempo",
        "cada um a seu tempo",
        "cada macaco no seu galho",
        "cada um no seu canto",
        "cada qual com a sua",
        "por sua conta e risco",
        "ser dono do seu nariz",
        "à sua maneira",
        "fazer das suas",
        "levar a sua avante",
        "ir à sua vida",
        "não é da minha conta",
        "não é da sua conta",
        "ter as suas coisas",
        "dar o seu melhor",
        "fazer a sua vontade",
        "seguir o seu caminho",
    ]
    patterns = list(nlp_small.pipe(expressions))

    matcher.add("a5d3_5_B2", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end)
            for match_id, start, end in matches]
a5d3_5.REQUIRES_SPACY = True