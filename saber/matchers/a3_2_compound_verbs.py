from spacy.matcher import DependencyMatcher, Matcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def a3d2_1(doc):
    """
    A3.2-1: estar (pres. ind.) a + inf. (presente momentâneo)
    e.g., (Neste momento/Agora) Eles estão a estudar na biblioteca.
    Level: A1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "estar",
            "RIGHT_ATTRS": {
                "NORM": {"IN": ["esta", "estas", "estamos", "estao"]}
                }
        },
        {
            "LEFT_ID": "estar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "estar",
            "RIGHT_ATTRS": {
                "NORM": {"IN": ["esta", "estas", "estamos", "estao"]}
                }
        },
        {
            "LEFT_ID": "estar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_1_A1", [pattern, pattern_reverse])

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


def a3d2_2(doc):
    """
    A3.2-2: costumar (pres. ind.) + inf. (presente frequentativo)
    e.g., Ela costuma almoçar na cantina.
    Level: A1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "costumar",
            "RIGHT_ATTRS": {
                "LEMMA": "costumar",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "costumar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "costumar",
            "RIGHT_ATTRS": {
                "LEMMA": "costumar",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "costumar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        }
    ]
    
    matcher.add("a3d2_2_A1", [pattern, pattern_reverse])

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

def a3d2_3(doc):
    """
    A3.2-3: ir + inf . (futuro próximo)
    e.g., Amanhã/Na próxima semana vou visitar o Mosteiro dos Jerónimos.
    Level: A1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "ir",
            "RIGHT_ATTRS": {
                "LEMMA": "ir",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "ir",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "ir",
            "RIGHT_ATTRS": {
                "LEMMA": "ir",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "ir",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        }
    ]
    
    matcher.add("a3d2_3_A1", [pattern, pattern_reverse])

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

def a3d2_4(doc):
    """
    A3.2-4: começar a + inf. - início de ação
    e.g., Começo a trabalhar às 14h.
    Level: A1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "começar",
            "RIGHT_ATTRS": {
                "LEMMA": "começar",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "começar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "começar",
            "RIGHT_ATTRS": {
                "LEMMA": "começar",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "começar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_4_A1", [pattern, pattern_reverse])

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

def a3d2_5(doc):
    """
    A3.2-5: acabar de + inf. - fim da ação
    e.g., Todos os dias, acabo de jantar e vejo televisão.
    Level: A1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "acabar",
            "RIGHT_ATTRS": {
                "LEMMA": "acabar",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "acabar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "acabar",
            "RIGHT_ATTRS": {
                "LEMMA": "acabar",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "acabar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_5_A1", [pattern, pattern_reverse])

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

def a3d2_8(doc):
    """
    A3.2-8: ter de + inf. (obrigação/necessidade)
    e.g., Tenho de carregar o passe.
    Level: A1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "LEMMA": "ter",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "LEMMA": "ter",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_8_A1", [pattern, pattern_reverse])

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

def a3d2_9(doc):
    """
    A3.2-9: estar (p.p.s. ind.) a + inf. - ação contínua num momento delimitado do passado.
    e.g., Estiveram a estudar toda a tarde. Estive a falar com a tua irmã durante muito tempo.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "estar",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["estive", "estiveste", "esteve", "estivemos", "estiveram"]}
                }
        },
        {
            "LEFT_ID": "estar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    # To match "Mais tarde, quando cheguei a casa, percebi que o meu irmão esteve, certamente, a jogar videojogos durante horas, pois a consola ainda estava ligada e os comandos largados no sofá."
    pattern_reverse = [
        {
            "RIGHT_ID": "estar",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["estive", "estiveste", "esteve", "estivemos", "estiveram"]}
                }
        },
        {
            "LEFT_ID": "estar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_9_A2", [pattern, pattern_reverse])

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

def a3d2_10(doc):
    """
    A3.2-10: estar (imperf. ind.) a + inf . - ação em progresso num momento preciso do passado, simultânea ou não de outra ação.
    e.g., Às 21h elas ainda estavam a trabalhar. Ela estava a atravessar a rua quando me viu.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "estar",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["estava", "estavas", "estava", "estávamos", "estavam"]}
                }
        },
        {
            "LEFT_ID": "estar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    # To match "Mais tarde, quando cheguei a casa, percebi que o meu irmão esteve, certamente, a jogar videojogos durante horas, pois a consola ainda estava ligada e os comandos largados no sofá."
    pattern_reverse = [
        {
            "RIGHT_ID": "estar",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["estava", "estavas", "estava", "estávamos", "estavam"]}
                }
        },
        {
            "LEFT_ID": "estar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_10_A2", [pattern, pattern_reverse])

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


def a3d2_11(doc):
    """
    A3.2-11: andar (pres. ind.) a + inf. (ação em desenvolvimento)
    e.g., Andamos a aprender português.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "andar",
            "RIGHT_ATTRS": {
                "LEMMA": "andar",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "andar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "andar",
            "RIGHT_ATTRS": {
                "LEMMA": "andar",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres"]}
                }
        },
        {
            "LEFT_ID": "andar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_11_A2", [pattern, pattern_reverse])

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

def a3d2_14(doc):
    """
    A3.2-14: dever + inf. (certeza/probabilidade)
    e.g., O João devia estar em casa.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "dever",
            "RIGHT_ATTRS": {
                "LEMMA": "dever"
                }
        },
        {
            "LEFT_ID": "dever",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        }
    ]

    # To match "Este restaurante deve ser bom, porque está sempre cheio de clientes."
    pattern_2 = [
        {
            "RIGHT_ID": "dever",
            "RIGHT_ATTRS": {
                "LEMMA": "dever"
                }
        },
        {
            "LEFT_ID": "dever",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "."             
        }
    ]

    matcher.add("a3d2_14_A2", [pattern, pattern_2])

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


def a3d2_15(doc):
    """
    A3.2-15: haver (pres. ind.) de + inf.
    resolução de ação ou intenção de ação futura"
    e.g., Hei de descobrir o que se passou.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "haver",
            "RIGHT_ATTRS": {
                "NORM": {"IN": ["hei", "has", "ha", "havemos", "hemos", "hao", "haveis", "heis"]},
                }
        },
        {
            "LEFT_ID": "haver",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "haver",
            "RIGHT_ATTRS": {
                "NORM": {"IN": ["hei", "has", "ha", "havemos", "hemos", "hao"]},
                }
        },
        {
            "LEFT_ID": "haver",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_15_B1", [pattern, pattern_reverse])

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


def a3d2_16(doc):
    """
    A3.2-16: continuar a + inf. - ação ou estado contínuo ou permanente
    e.g., Eu entrei e eles continuaram a ver televisão.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "continuar",
            "RIGHT_ATTRS": {
                "LEMMA": "continuar"
                }
        },
        {
            "LEFT_ID": "continuar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]


    pattern_reverse = [
        {
            "RIGHT_ID": "continuar",
            "RIGHT_ATTRS": {
                "LEMMA": "continuar"
                }
        },
        {
            "LEFT_ID": "continuar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_16_B1", [pattern, pattern_reverse])

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


def a3d2_17(doc):
    """
    A3.2-17: acabar de + inf. - ação terminada num passado muito próximo do presente
    Acabei de falar com o Fernando. Ele também vai ao jantar.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "acabar",
            "RIGHT_ATTRS": {
                "MORPH": {"INTERSECTS": ["Tense=Past", "Tense=Imp", "VerbForm=Part"]},
                "LEMMA": 'acabar'
                }
        },
        {
            "LEFT_ID": "acabar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "acabar",
            "RIGHT_ATTRS": {
                "LEMMA": "acabar"
                }
        },
        {
            "LEFT_ID": "acabar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_17_B1", [pattern, pattern_reverse])

    matches = matcher(doc)
    
    results = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]

        # Exclude the match if the last token in the match is the parent of a token with dep "conj".
        # To avoid matching "O Pedro e a Carla acabaram de almoçar e decidiram dar um passeio antes de voltarem ao trabalho."
        last_token = doc[max(token_ids)]
        if any(child.dep_ == "conj" for child in last_token.children):
            continue
        
        text = reconstruct_text(tokens_in_span)
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def a3d2_18(doc):
    """
    A3.2-18: deixar (pret. perf. ind.) de + inf. - paragem, supressão ou desistência
    e.g. Ela deixou de jogar andebol aos 26 anos./ Eles deixaram de fumar há muito tempo.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "deixar",
            "RIGHT_ATTRS": {
                "LEMMA": "deixar",
                "MORPH": {"IS_SUPERSET": ["Tense=Past", "Mood=Ind"]}
                }
        },
        {
            "LEFT_ID": "deixar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "deixar",
            "RIGHT_ATTRS": {
                "LEMMA": "deixar",
                "MORPH": {"IS_SUPERSET": ["Tense=Past", "Mood=Ind"]}
                }
        },
        {
            "LEFT_ID": "deixar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_18_B1", [pattern, pattern_reverse])

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


def a3d2_19(doc):
    """
    A3.2-19: "haver de (imperf. ind.) + inf. -
    expressão de conveniência/necessidade/dever"
    e.g., Havíamos de chegar ao teatro com antecedência.
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "haver",
            "RIGHT_ATTRS": {
                "LEMMA": "haver",
                "MORPH": {"IS_SUPERSET": ["Tense=Imp", "Mood=Ind"]}
                }
        },
        {
            "LEFT_ID": "haver",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "haver",
            "RIGHT_ATTRS": {
                "LEMMA": "haver",
                "MORPH": {"IS_SUPERSET": ["Tense=Imp", "Mood=Ind"]}
                }
        },
        {
            "LEFT_ID": "haver",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_19_B2", [pattern, pattern_reverse])

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

def a3d2_20(doc):
    """
    A3.2-20: "deixar de + inf. - eventualidade de não concretização de ação habitual ou futura"
    e.g. Não deixes de fazer o que te dá prazer!
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "deixar",
            "RIGHT_ATTRS": {
                            "LEMMA": "deixar",
                            "POS": "VERB"
                            }
        },
        {
            "LEFT_ID": "deixar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "deixar",
            "RIGHT_ATTRS": {
                            "LEMMA": "deixar",
                            "POS": "VERB"
                            }
        },
        {
            "LEFT_ID": "deixar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_20_B2", [pattern, pattern_reverse])

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


def a3d2_21(doc):
    """
    A3.2-21: "estar para + inf. - proximidade de realização da ação"
    e.g., Ela está para casar! / Eles estavam para mudar de casa, não sei se já mudaram...
    Level: B2

    We're still not matching "O Rui estava, segundo me disseram, para aceitar aquela
    proposta de trabalho, mas parece que mudou de ideias." But that's because the dependency
    parser gets the dependency relations wrong, and being more flexible can result in unexpected
    false positives here.
    """
    # matcher = DependencyMatcher(nlp_stanza.vocab,)

    # pattern = [
    #     {
    #         "RIGHT_ID": "estar",
    #         "RIGHT_ATTRS": {
    #             "LEMMA": "estar"
    #             }
    #     },
    #     {
    #         "LEFT_ID": "estar",
    #         "RIGHT_ID": "verb",
    #         "RIGHT_ATTRS": {
    #             "POS": "VERB",
    #             "MORPH": "VerbForm=Inf"
    #             },
    #         "REL_OP": ">++"             
    #     },
    #     {
    #         "LEFT_ID": "estar",
    #         "RIGHT_ID": "para",
    #         "RIGHT_ATTRS": {
    #             "LOWER": "para",
    #             "POS": "SCONJ"
    #             },
    #         "REL_OP": ">"             
    #     }
    # ]

    # # To match "Mais tarde, quando cheguei a casa, percebi que o meu irmão esteve, certamente, a jogar videojogos durante horas, pois a consola ainda estava ligada e os comandos largados no sofá."
    # pattern_reverse = [
    #     {
    #         "RIGHT_ID": "estar",
    #         "RIGHT_ATTRS": {
    #             "LEMMA": "estar"
    #             }
    #     },
    #     {
    #         "LEFT_ID": "estar",
    #         "RIGHT_ID": "verb",
    #         "RIGHT_ATTRS": {
    #             "POS": "VERB",
    #             "MORPH": "VerbForm=Inf"
    #             },
    #         "REL_OP": "<++"             
    #     },
    #     {
    #         "LEFT_ID": "verb",
    #         "RIGHT_ID": "para",
    #         "RIGHT_ATTRS": {
    #             "LOWER": "para",
    #             "POS": "SCONJ"
    #             },
    #         "REL_OP": ">"             
    #     }
    # ]

    # pattern_composto = [
    #     {
    #         "RIGHT_ID": "estar",
    #         "RIGHT_ATTRS": {
    #             "LEMMA": "estar"
    #             }
    #     },
    #     {
    #         "LEFT_ID": "estar",
    #         "RIGHT_ID": "pp",
    #         "RIGHT_ATTRS": {
    #             "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}
    #         },
    #         "REL_OP": ">++"
    #     },
    #     {
    #         "LEFT_ID": "pp",
    #         "RIGHT_ID": "verb",
    #         "RIGHT_ATTRS": {
    #             "POS": "AUX",
    #             "MORPH": "VerbForm=Inf"
    #             },
    #         "REL_OP": ">-"             
    #     },
    #     {
    #         "LEFT_ID": "pp",
    #         "RIGHT_ID": "para",
    #         "RIGHT_ATTRS": {
    #             "LOWER": "para",
    #             "POS": "SCONJ"
    #             },
    #         "REL_OP": ">--"             
    #     }
    # ]
    
    # matcher.add("a3d2_21_B2", [pattern, pattern_reverse, pattern_composto])

    # matches = matcher(doc)
    
    # results = []
    # for match_id, token_ids in matches:
    #     # Calculate start and end indices
    #     start = min(token_ids)
    #     end = max(token_ids) + 1
    #     tokens_in_span = doc[start:end]
        
    #     text = reconstruct_text(tokens_in_span)

    #     results.append((doc.vocab.strings[match_id], text, start, end))

    # return results

    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": "estar"},
                {"LOWER": "para"},
                {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
            ]

    matcher.add("a3d2_21_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d2_22(doc):
    """
    A3.2-22: "ir a + inf. - ação apenas iniciada"
    e.g., Ela ia a dizer qualquer coisa, mas arrependeu-se!
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "ir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir", "ír"]},
                # Ias a responder-lhe, mas ficaste calado. 
                "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}
                }
        },
        {
            "LEFT_ID": "ir",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]


    pattern_reverse = [
        {
            "RIGHT_ID": "ir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir", "ír"]},
                # "Nós íamos a provar o bolo"
                "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}
                }
        },
        {
            "LEFT_ID": "ir",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_22_B2", [pattern, pattern_reverse])

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


def a3d2_23(doc):
    """
    A3.2-23: vir a + inf. - resultado final da ação
    e.g., Viemos a descobrir que eles tinham sido colegas na escola!
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "vir",
            "RIGHT_ATTRS": {
                "LEMMA": "vir"
                }
        },
        {
            "LEFT_ID": "vir",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]


    pattern_reverse = [
        {
            "RIGHT_ID": "vir",
            "RIGHT_ATTRS": {
                "LEMMA": "vir"
                }
        },
        {
            "LEFT_ID": "vir",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": "VerbForm=Inf"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": "SCONJ"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("a3d2_23_B2", [pattern, pattern_reverse])

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


def a3d2_24(doc):
    """
    A3.2-24: "ir/vir ter com/a - encontro com ideia de deslocação de uma das partes"
    e.g., Vou ter contigo por volta das 17h! / Vens ter à escola mais logo?
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "ir-vir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir", "vir", "ser"]}
                # Also including "ser" for cases where the preprocessor might misidentify the lemma.
                }
        },
        {
            "LEFT_ID": "ir-vir",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "LOWER": "ter"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "com",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["contigo", "consigo", "comigo", "conosco",
                                 "connosco", "convosco", "com"]},
                "POS": {"IN": ["ADP", "ADV", "DET"]}
                },
            "REL_OP": "."             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "ir-vir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir", "vir", "ser"]}
                # Also including "ser" for cases where the preprocessor might misidentify the lemma.
                }
        },
        {
            "LEFT_ID": "ir-vir",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "LOWER": "ter"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "com",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["contigo", "consigo", "comigo", "conosco",
                                 "connosco", "convosco", "com"]},
                "POS": {"IN": ["ADP", "ADV"]}
                },
            "REL_OP": "."             
        }
    ]

    # For cases where the a contracted article follows the pattern such as in
    # "Vou ter ao cinema."

    pattern_w_article = [
        {
            "RIGHT_ID": "ir-vir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir", "vir", "ser"]}
                # Also including "ser" for cases where the preprocessor might misidentify the lemma.
                }
        },
        {
            "LEFT_ID": "ir-vir",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "LOWER": "ter"
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "com",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": {"IN": ["ADP", "ADV", "DET"]}
                },
            "REL_OP": "."             
        },
        {
            "LEFT_ID": "com",
            "RIGHT_ID": "det",
            "RIGHT_ATTRS": {
                "POS": "DET"
                },
            "REL_OP": "."
        }
    ]

    pattern_reverse_w_article = [
        {
            "RIGHT_ID": "ir-vir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir", "vir", "ser"]}
                # Also including "ser" for cases where the preprocessor might misidentify the lemma.
                }
        },
        {
            "LEFT_ID": "ir-vir",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "LOWER": "ter"
                },
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "com",
            "RIGHT_ATTRS": {
                "LOWER": "a",
                "POS": {"IN": ["ADP", "ADV"]}
                },
            "REL_OP": "."             
        },
        {
            "LEFT_ID": "com",
            "RIGHT_ID": "det",
            "RIGHT_ATTRS": {
                "POS": "DET"
                },
            "REL_OP": "."
        }
    ]

    
    matcher.add("a3d2_24_B2", [pattern, pattern_reverse,
                               pattern_w_article, pattern_reverse_w_article])

    matches = matcher(doc)
    
    print(matches)
    
    results = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]
        
        text = reconstruct_text(tokens_in_span)
        print(text)
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def a3d2_25(doc):
    """
    A3.2-25: "ir + gerúndio - ação durativa, que se desenvolve lentamente (em simultâneo ou em direção a um ponto temporal determinado)"
    e.g., Vai pondo a mesa (enquanto acabo o jantar). / Ela foi preparando tudo (ao longo da tarde).
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "ir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir", "ser"]},
                # Also including "ser" for cases where the preprocessor might misidentify the lemma.
                "MORPH": {"INTERSECTS": ["Tense=Pres", "Tense=Past"]}
                }
        },
        {
            "LEFT_ID": "ir",
            "RIGHT_ID": "ger",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                },
            "REL_OP": "<+"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "ir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir", "ser"]},
                # Also including "ser" for cases where the preprocessor might misidentify the lemma.
                "MORPH": {"INTERSECTS": ["Tense=Pres", "Tense=Past"]}
                }
        },
        {
            "LEFT_ID": "ir",
            "RIGHT_ID": "ger",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                },
            "REL_OP": ">+"             
        }
    ]
    
    
    matcher.add("a3d2_25_B2", [pattern, pattern_reverse])

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


def a3d2_26(doc):
    """
    A3.2-26: "ir (imperf.) + gerúndio - ação que quase se realizou"
    e.g., Ele ia caindo, mas conseguiu segurar-se!
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "ir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir"]},
                "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}
                }
        },
        {
            "LEFT_ID": "ir",
            "RIGHT_ID": "ger",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                },
            "REL_OP": "<+"             
        }
    ]

    pattern_reverse = [
        {
            "RIGHT_ID": "ir",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ir"]},
                "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}
                }
        },
        {
            "LEFT_ID": "ir",
            "RIGHT_ID": "ger",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                },
            "REL_OP": ">+"             
        }
    ]
    
    matcher.add("a3d2_26_C1", [pattern, pattern_reverse])

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