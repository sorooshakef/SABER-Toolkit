from spacy.matcher import Matcher, PhraseMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def a1d1_6(doc):
    """
    A1.1-6: género - masculinos terminados em -ão, e.g. irmão
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {
        "POS": "DET",
        "MORPH": {"IS_SUPERSET": ["Gender=Masc"]}
    },
    {
        "LOWER": {"REGEX": "ão$"},
        "POS": "NOUN",
        "MORPH": {"IS_SUPERSET": ["Gender=Masc"]}
    }
    ]
    matcher.add("a1d1_6_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a1d1_7(doc):
    """
    A1.1-7: género - masculinos terminados em -a, e.g. dia, cinema, problema
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)


    pattern = [
        {
            "LOWER": {"REGEX": "a$"},
            "POS": "NOUN",
            "MORPH": {"IS_SUPERSET": ["Gender=Masc"]}
        }
    ]
    matcher.add("a1d1_7_A2", [pattern])

    matches = matcher(doc)

    filtered_matches = {}
    for match in matches:
        match_id, start, end = match
        # If we haven't seen this end index, or if this match starts later than
        # the currently stored one, update the filtered matches.
        if end not in filtered_matches or start < filtered_matches[end][1]:
            filtered_matches[end] = (match_id, start, end)
    
    # Get the final list of matches from the dictionary values.
    final_matches = list(filtered_matches.values())
    
    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) 
            for match_id, start, end in final_matches]


def a1d1_8(doc):
    """
    A1.1-8: género - nomes invariáveis quanto ao género, e.g. o/a estudante; o/a artista; a criança; a pessoa
    Level: A2
    Requires nlp_stanza

    The idea here is to capture cases that are particularly tricky, for example "o colega," but not cases that are
    more straightforward, like "a colega." We also consider cases like "pessoa" which is feminine regardless of the
    gender of the person as tricky feminine cases.
    """
    matcher = Matcher(doc.vocab)

    tricky_masc_words = [
        # Original
        "artista", "dentista", "jornalista", "motorista", "ciclista", "poeta", "taxista", "atleta",
        "camarada", "colega", "guarda", "cliente", "gerente", "parente", "estudante", "presidente", "guia", "especialista",
        
        # Added: Comuns de dois gêneros ending in -a
        "pianista", "recepcionista", "eletricista", "psiquiatra", "pediatra", "astronauta", "idiota", "terapeuta", "entusiasta",
        
        # Added: Comuns de dois gêneros lacking standard morphological gender markers (-e, -m, consonants)
        "adolescente", "paciente", "jovem", "habitante", "viajante", "sobrevivente", "líder", "intérprete",
        
        # Added: Sobrecomuns (fixed masculine gender representing any biological sex)
        "indivíduo", "ídolo", "anjo", "gênio", "cônjuge"
    ]

    tricky_fem_words = [
        # Original
        "estudante", "presidente", "vítima", "cliente", "gerente", "parente", "pessoa", "criança",
        
        # Added: Comuns de dois gêneros ending in -o (highly counterintuitive with feminine determinants)
        "modelo", "piloto",
        
        # Added: Comuns de dois gêneros lacking standard morphological gender markers (-e, -m, consonants)
        "adolescente", "paciente", "jovem", "habitante", "viajante", "sobrevivente", "líder", "intérprete",
        
        # Added: Sobrecomuns (fixed feminine gender representing any biological sex)
        "testemunha", "criatura"
    ]
    
    # Patterns for words with feminine article (if they do not adhere to standard -a suffix paradigms)
    pattern_fem = [
        {
            "POS": "DET",
            "MORPH": {"IS_SUPERSET": ["Gender=Fem"]}
        },
        {"LOWER": {"IN": tricky_fem_words}}
    ]

    # Patterns for words with masculine article (if they do not adhere to standard -o suffix paradigms)
    pattern_masc = [
        {
            "POS": "DET",
            "MORPH": {"IS_SUPERSET": ["Gender=Masc"]}
        },
        {"LOWER": {"IN": tricky_masc_words}}
    ]

    matcher.add("a1d1_8_A2", [pattern_masc, pattern_fem])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a1d1_9(doc):
    """
    A1.1-9: número - masculinos terminados em -ão, e.g. leão, leões; pão, pães; irmão, irmãos
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)


    pattern = [
    {
        "POS": "NOUN",
        "MORPH": {"IS_SUPERSET": ["Number=Plur", "Gender=Masc"]},
        "LEMMA": {"REGEX": "ão$"}
    }
    ]

    matcher.add("a1d1_9_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a1d1_10(doc):
    """
    A1.1-10: número - nomes invariáveis quanto ao número, e.g. o/os lápis; os óculos; as calças;
    Level: A2
    Requires nlp_stanza
    """
    matcher = PhraseMatcher(doc.vocab, attr="NORM")

    patterns = list(nlp_stanza.pipe(["lapis", "oculos", "calcas"]))

    matcher.add("a1d1_10_A2", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a1d1_11(doc):
    """
    A1.1-11: género - outros casos de flexão - e.g. europeu, europeia; ator, atriz; crocodilo macho; crocodilo fêmea
    Level: B1
    Need a more comprehensive list of what these words are.
    """
    matcher = Matcher(doc.vocab)

    word_list = ["europeu", "europeia", "ator", "atriz"]

    pattern_1 = [
        {
            "NORM": {"IN": word_list}
        }
    ]

    pattern_2 = [
        {
            "POS": "NOUN"
        },
        {
            "LEMMA": {"IN": ["macho", "fêmea", "fêmeo"]}
        }
    ]
    matcher.add("a1d1_11_B1", [pattern_1, pattern_2])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a1d1_13(doc):
    """
    A1.1-13: número - plural de nomes compostos - e.g. girassóis; navios-escola; porta-vozes; pães-de-ló; amores-perfeitos
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "TEXT": {"REGEX": "[A-Za-z]+(-[A-Za-z]+){1,2}"},
                    "MORPH": {"IS_SUPERSET": ["Number=Plur"]},
                    "POS": "NOUN"
                }
            ]

    matcher.add("a1d1_13_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a1d1_14(doc):
    """
    A1.1-14: grau - diminutivos e aumentaivos frequentes (-inho, -ito, -zinho, -zito) - e.g.,carrinho/carrito, irmãozinho/irmãozito; carrão, casarão
    Level: B1
    When we are confident there are no other Portuguese words that end with these characters, we use a regular expression.
    For words ending with "ão" or "ito," however, we need to use word lists.
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
                {
                    "NORM": {"REGEX": "\\b[A-Za-z]+(?:inho|inha|ito|ita|inhos|inhas|itos|itas)\\b"},
                    "POS": "NOUN", # Removed "PROPN" because I don't remember why I'd added it, and we don't want to match "Rita"!
                    "LEMMA": {"NOT_IN": ["caminho", "palito", "cozinha", "vizinho", "farinha", "linha",
                                         "vizinha", "vinho", "moinho", "carinho", "propósito", "mérito",
                                         "inquérito", "incógnita", "visita", "golfinho", "âmbito", "direito",
                                         "trânsito", "suspeito", "respeito"]}
                }
            ]
    
    pattern_2 = [
        {
            "NORM": {"IN": ["carrito", "carrao", "casarao", "cachorrao", "tartezao",
                            "livrao", "maozao", "copao", "peixao", "olhao", "pezao",
                            "jogao", "narigao", "piscinao", "mocao",
                            "cavalao", "farolao", "bolao",
                            "coracaozao", "mochilao", "colchaozao"]}
        }
    ]

    matcher.add("a1d1_14_B1", [pattern_1, pattern_2])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]