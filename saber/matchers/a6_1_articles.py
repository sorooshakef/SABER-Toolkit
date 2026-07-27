from spacy.matcher import Matcher, DependencyMatcher
from text_reconstruction import reconstruct_text

def a6d1_3(doc):
    """
    A6.1-3: artigo definido - uso/valor - individualização/determinação do ser designado pelo nome - com nomes próprios (primeiro nome)
    e.g., o Joaquim
    Level: A1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["a", "o"]}, "POS": "DET"},
                {"POS": "PROPN", "ENT_TYPE": "PER"}
            ]

    matcher.add("a6d1_3_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a6d1_3.REQUIRES_SPACY = True

def a6d1_4(doc):
    """
    A6.1-4: artigo definido - uso/valor - individualização/determinação do ser designado pelo nome - com a maioria dos nomes de países , regiões e cidades/localidades
    e.g., a Alemanha/a Bavária; o Porto/a Guarda
    Level: A1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["a", "o"]}, "POS": "DET"},
                {"POS": "PROPN", "ENT_TYPE": "LOC"}
            ]

    matcher.add("a6d1_4_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a6d1_4.REQUIRES_SPACY = True

def a6d1_5(doc):
    """
    A6.1-5: artigo definido - uso/valor - individualização/determinação do ser designado pelo nome - antes de determinante possessivo
    e.g., o meu irmão
    Level: A1
    """
    matcher = Matcher(doc.vocab)

    pattern_o = [
                {},
                {"LOWER": "o", "POS": "DET", "IS_SENT_START": False},
                {"LOWER": {"IN": ['meu', 'nosso', 'teu', 'vosso', 'seu']}}
            ]
    
    pattern_a = [
                {},
                {"LOWER": "a", "POS": "DET", "IS_SENT_START": False},
                {"LOWER": {"IN": ['minha', 'nossa', 'tua', 'vossa', 'sua']}}
            ]
    
    pattern_os = [
                {},
                {"LOWER": "os", "POS": "DET", "IS_SENT_START": False},
                {"LOWER": {"IN": ['meus', 'nossos', 'teus', 'vossos', 'seus']}}
            ]
    
    pattern_as = [
                {},
                {"LOWER": "as", "POS": "DET", "IS_SENT_START": False},
                {"LOWER": {"IN": ['minhas', 'nossas', 'tuas', 'vossas', 'suas']}}
            ]
    
    pattern_o_alt = [
                {"LOWER": "o", "POS": "DET", "IS_SENT_START": True},
                {"LOWER": {"IN": ['meu', 'nosso', 'teu', 'vosso', 'seu']}}
            ]
    
    pattern_a_alt = [
                {"LOWER": "a", "POS": "DET", "IS_SENT_START": True},
                {"LOWER": {"IN": ['minha', 'nossa', 'tua', 'vossa', 'sua']}}
            ]
    
    pattern_os_alt = [
                {"LOWER": "os", "POS": "DET", "IS_SENT_START": True},
                {"LOWER": {"IN": ['meus', 'nossos', 'teus', 'vossos', 'seus']}}
            ]
    
    pattern_as_alt = [
                {"LOWER": "as", "POS": "DET", "IS_SENT_START": True},
                {"LOWER": {"IN": ['minhas', 'nossas', 'tuas', 'vossas', 'suas']}}
            ]

    matcher.add("a6d1_5_A1", [pattern_a, pattern_o, pattern_as, pattern_os,
                              pattern_o_alt, pattern_a_alt, pattern_os_alt, pattern_as_alt])

    matches = matcher(doc)

    filtered_matches = []
    for match_id, start, end in matches:
        # Check if the first token is a relevant preposition or matches the expected first relevant token
        first_token = doc[start]
        if first_token.lower_ in ["em", "de", "por", "a", "o", "as", "os"]:
            # Keep the full match
            filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
        else:
            # Exclude the first token from the match
            if end - start > 1:  # Ensure we have more than one token
                filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start+1:end]), start+1, end))
    
    return filtered_matches

def a6d1_6(doc):
    """
    A6.1-6: artigo definido - concordância - em género e número com o nome
    e.g., o livro/os livros; a revista/as revistas
    Level: A1
    """
    matcher = Matcher(doc.vocab)

    pattern_o = [
                {},
                {"LOWER": "o", "POS": "DET", "IS_SENT_START": False},
                {"LOWER": {"IN": ['meu', 'nosso', 'teu', 'vosso', 'seu']}, "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Sing"}
            ]
    
    pattern_a = [
                {},
                {"LOWER": "a", "POS": "DET", "IS_SENT_START": False},
                {"LOWER": {"IN": ['minha', 'nossa', 'tua', 'vossa', 'sua']}, "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Sing"}
            ]
    
    pattern_os = [
                {},
                {"LOWER": "os", "POS": "DET", "IS_SENT_START": False},
                {"LOWER": {"IN": ['meus', 'nossos', 'teus', 'vossos', 'seus']}, "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Plur"}
            ]
    
    pattern_as = [
                {},
                {"LOWER": "as", "POS": "DET", "IS_SENT_START": False},
                {"LOWER": {"IN": ['minhas', 'nossas', 'tuas', 'vossas', 'suas']}, "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Plur"}
            ]
    
    pattern_o_alt = [
                {"LOWER": "o", "POS": "DET", "IS_SENT_START": True},
                {"LOWER": {"IN": ['meu', 'nosso', 'teu', 'vosso', 'seu']}, "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Sing"}
            ]
    
    pattern_a_alt = [
                {"LOWER": "a", "POS": "DET", "IS_SENT_START": True},
                {"LOWER": {"IN": ['minha', 'nossa', 'tua', 'vossa', 'sua']}, "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Sing"}
            ]
    
    pattern_os_alt = [
                {"LOWER": "os", "POS": "DET", "IS_SENT_START": True},
                {"LOWER": {"IN": ['meus', 'nossos', 'teus', 'vossos', 'seus']}, "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Plur"}
            ]
    
    pattern_as_alt = [
                {"LOWER": "as", "POS": "DET", "IS_SENT_START": True},
                {"LOWER": {"IN": ['minhas', 'nossas', 'tuas', 'vossas', 'suas']}, "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Plur"}
            ]

    matcher.add("a6d1_6_A1", [pattern_a, pattern_o, pattern_as, pattern_os,
                              pattern_a_alt, pattern_o_alt, pattern_as_alt, pattern_os_alt])

    matches = matcher(doc)

    filtered_matches = []
    for match_id, start, end in matches:
        # Check if the first token is a relevant preposition or matches the expected first relevant token
        first_token = doc[start]
        if first_token.lower_ in ["em", "de", "por", "a", "o", "as", "os"]:
            # Keep the full match
            filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
        else:
            # Exclude the first token from the match
            if end - start > 1:  # Ensure we have more than one token
                filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start+1:end]), start+1, end))
    
    return filtered_matches


def a6d1_9(doc):
    """
    A6.1-9: artigo indefinido - concordância - em género e número com o nome
    e.g., um livro/uns livros; uma revista/umas revistas
    Level: A1
    """
    matcher = Matcher(doc.vocab)

    pattern_um = [
                {},
                {"LOWER": "um", "POS": "DET", "IS_SENT_START": False},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Sing", "POS": "NOUN"}
            ]
    
    pattern_uma = [
                {},
                {"LOWER": "uma", "POS": "DET", "IS_SENT_START": False},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Sing", "POS": "NOUN"}
            ]
    
    pattern_uns = [
                {},
                {"LOWER": "uns", "POS": "DET", "IS_SENT_START": False},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Plur", "POS": "NOUN"}
            ]
    
    pattern_umas = [
                {},
                {"LOWER": "umas", "POS": "DET", "IS_SENT_START": False},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Plur", "POS": "NOUN"}
            ]
    
    pattern_um_alt = [
                {},
                {"LOWER": "um", "POS": "DET", "IS_SENT_START": True},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Sing", "POS": "NOUN"}
            ]
    
    pattern_uma_alt = [
                {},
                {"LOWER": "uma", "POS": "DET", "IS_SENT_START": True},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Sing", "POS": "NOUN"}
            ]
    
    pattern_uns_alt = [
                {},
                {"LOWER": "uns", "POS": "DET", "IS_SENT_START": True},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Masc|Number=Plur", "POS": "NOUN"}
            ]
    
    pattern_umas_alt = [
                {},
                {"LOWER": "umas", "POS": "DET", "IS_SENT_START": True},
                {"POS": "ADJ", "OP": "?"},
                {"MORPH": "Gender=Fem|Number=Plur", "POS": "NOUN"}
            ]

    matcher.add("a6d1_9_A1", [pattern_um, pattern_uma, pattern_uns, pattern_umas,
                              pattern_um_alt, pattern_uma_alt, pattern_uns_alt, pattern_umas_alt])

    matches = matcher(doc)

    filtered_matches = []
    for match_id, start, end in matches:
        # Check if the first token is a relevant preposition or matches the expected first relevant token
        first_token = doc[start]
        if first_token.lower_ in ["em", "de", "por", "a", "um", "uma", "uns", "umas"]:
            # Keep the full match
            filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
        else:
            # Exclude the first token from the match
            if end - start > 1:  # Ensure we have more than one token
                filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start+1:end]), start+1, end))
    
    return filtered_matches


def a6d1_10(doc):
    """
    A6.1-10: artigo definido - forma - contração com preposições
    e.g., de - do/da/dos/das; em - no/na/nos/nas
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["de", "em", "por"]}},
                {"LOWER": {"IN": ["o", "a", "os", "as"]}, "POS": "DET"}
            ]

    matcher.add("a6d1_10_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a6d1_11(doc):
    """
    A6.1-11: artigo definido - uso/valor - com nomes próprios (apelido)
    e.g., os Ribeiro
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["os", "as"]}, "POS": "DET"},
                {"POS": "PROPN", "ENT_TYPE": "PER"}
            ]

    matcher.add("a6d1_11_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a6d1_11.REQUIRES_SPACY = True

def a6d1_15(doc):
    """
    A6.1-15: artigo indefinido - uso/valor - antes de numeral, indica aproximação numérica
    e.g., ficava a uns 10km da sua casa/demora uns 20 minutos
    Level: A2
    """
    matcher = DependencyMatcher(doc.vocab)

    pattern = [
        {
            "RIGHT_ID": "uns",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["uns", "umas"]}
            }
        },
        {
            "LEFT_ID": "uns",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {
                "POS": "NUM",
                "MORPH": "NumType=Card"
            },
            "REL_OP": ".*"  # So that we can match "Segundo o mecânico, o reboque demoraria uns, vá lá, 30 minutos a chegar, 
                           # mas acabou por aparecer passados uns bons 45 minutos."            
        }
    ]
    
    matcher.add("a6d1_15_A2", [pattern])
    matches = matcher(doc)
    
    # Use a dictionary to keep the best (shortest span) match for each start index
    best_matches = {}
    for match_id, token_ids in matches:
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]
        text = reconstruct_text(tokens_in_span)
        print(text)
        
        # If there's already a match with the same start, keep the one with the smaller end index
        if start in best_matches:
            if end < best_matches[start][3]:
                best_matches[start] = (doc.vocab.strings[match_id], text, start, end)
        else:
            best_matches[start] = (doc.vocab.strings[match_id], text, start, end)

    # Return the best matches as a list
    results = list(best_matches.values())
    return results

def a6d1_19(doc):
    """
    A6.1-19: artigo definido - uso/valor - individualização/determinação do ser designado pelo nome  - antes de datas festivas ou célebres
    e.g., O 1 de maio é feriado.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": "o"},
                {"POS": "NUM"},
                {"LOWER": "de"},
                {"NORM": {"IN": ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]},}
            ]
    
    # To match "O 25 de Abril"
    pattern_reg = [
                {"LOWER": "o"},
                {"TEXT": {"REGEX": r"^\d+$"}},
                {"LOWER": "de"},
                {"NORM": {"IN": ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]},}
            ]
    # To match "O dia do Trabalhador"
    pattern_fest = [
                {"LOWER": "o"},
                {"LOWER": "dia"},
                {"LEMMA": "de"},
                {"LEMMA": "o", "OP": "?"},
                {"POS": {"IN": ["PROPN", "NOUN"]}}
    ]

    pattern_common = [
                {"LOWER": {"IN": ["o", "a"]}},
                {"NORM": {"IN": ["natal", "pascoa", "carnaval", "corpus christi", "corpus christo"]}}
    ]

    matcher.add("a6d1_19_B1", [pattern, pattern_reg, pattern_fest, pattern_common])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a6d1_20(doc):
    """
    A6.1-20: artigo definido - uso/valor - individualização/determinação do ser designado pelo nome  - com 'todo/a/os/as'
    e.g., todos os dias; toda a semana ( ou a semana toda)
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["todo", "todos", "toda", "todas"]}},
                {"LOWER": {"IN": ["o", "os", "a", "as"]}},
                {"POS": "ADJ", "OP": "?"},
                {"POS": "NOUN"}
            ]
    
    pattern_alt = [
                {"LOWER": {"IN": ["o", "os", "a", "as"]}},
                {"POS": "NOUN"},
                {"LEMMA": {"IN": ["todo"]}}
    ]
    
    # An ad-hoc fix for a common construction. I didn't want to further expand the first pattern out of fear of mismatches.
    pattern_gente = [
                {"LOWER": "toda"},
                {"LOWER": "a"},
                {"LOWER": "gente"}
            ]
    matcher.add("a6d1_20_B1", [pattern, pattern_alt, pattern_gente])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a6d1_21(doc):
    """
    A6.1-21: artigo definido - uso/valor - individualização/determinação do ser designado pelo nome  - com 'ambos/as'
    e.g., Acenava com ambas as mãos.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": "ambos"},
                {"LOWER": {"IN": ["os", "as"]}},
                {"POS": "NOUN"}
            ]
    
    matcher.add("a6d1_21_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a6d1_28(doc):
    """
    A6.1-28: "artigo definido - uso/valor - individualização/determinação do ser designado pelo nome - para substantivar palavras que pertencem a outras classes"
    e.g., O olhar dele dava pena! O sim deles foi imediato.
    Level: B2
    """
    matcher = DependencyMatcher(doc.vocab)

    pattern = [
        {
            "RIGHT_ID": "o",
            "RIGHT_ATTRS": {
                "LOWER": "o",
                "POS": "DET"}
        },
        {
            "LEFT_ID": "o",
            "RIGHT_ID": "word",
            "RIGHT_ATTRS": {
                "POS": {"NOT_IN": ["NOUN", "PROPN"]},
                "LOWER": {"NOT_IN": ["mais", "menos", "pior", "melhor", "quê", "que"]}
                },
            "REL_OP": "<+"             
        }
    ]
    
    matcher.add("a6d1_28_B2", [pattern])

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
a6d1_28.REQUIRES_SPACY = True


def a6d1_34(doc):
    """
    A6.1-34: artigo indefinido - uso/valor - usa-se com nomes de pessoas (primeiro nome e apelido), para marcar indefinição de sujeito
    e.g., Acho que ele referiu um Francisco.
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["um", "uma"]}},
                {"LOWER": "tal", "OP": "?"},
                {"LOWER": "de", "OP": "?"},
                {"POS": "PROPN", "ENT_TYPE": "PER",
                 "LOWER": {"NOT_IN": ["amigo"]}} # Ad-hoc fix
            ]
    
    matcher.add("a6d1_34_C1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a6d1_34.REQUIRES_SPACY = True

def a6d1_35(doc):
    """
    A6.1-35: artigo indefinido - uso/valor - pode ocorrer com nomes geográficos, se estes aparecem qualificados
    e.g., um Portugal melhor; numa Europa devastada
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["um", "uma", "num", "numa"]}},
                {"POS": "PROPN", "ENT_TYPE": "LOC"},
                {"POS": "ADV", "OP": "*"},
                {"POS": "ADJ"}
            ]
    
    pattern_verb = [
                {"LOWER": {"IN": ["um", "uma", "num", "numa"]}},
                {"POS": "PROPN", "ENT_TYPE": "LOC"},
                {"POS": "ADV", "OP": "*"},
                {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}}
            ]
    
    # To match "um Brasil em reconstrução"
    pattern_noun = [
                {"LOWER": {"IN": ["um", "uma", "num", "numa"]}},
                {"POS": "PROPN", "ENT_TYPE": "LOC"},
                {"LOWER": "em"},
                {"POS": "NOUN"}
            ]
    
    matcher.add("a6d1_35_C1", [pattern, pattern_verb, pattern_noun])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a6d1_35.REQUIRES_SPACY = True