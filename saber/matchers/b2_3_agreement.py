from spacy.matcher import DependencyMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

# def b2d3_5(doc):
#     """
#     B2.3-5: Concordância (Suj./Pred.) - entre sujeito e verbo (sujeito simples contendo complementos ou modificadores)
#     e.g., A mãe dos gémeos vive no meu prédio. / *A mãe dos gémeos vivem no meu prédio.
#     Level: A2
#     """
#     matcher = DependencyMatcher(nlp_stanza.vocab)

#     pattern_1_sing = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=1"]}
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=1"]},
#                 "POS": {"IN": ["VERB", "AUX"]}
#                 },
#             "REL_OP": "<"             
#         }
#     ]

#     pattern_2_sing = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=2"]}
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=2"]},
#                 "POS": {"IN": ["VERB", "AUX"]}
#                 },
#             "REL_OP": "<"             
#         }
#     ]

#     pattern_3_sing = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=3"]}
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=3"]},
#                 "POS": {"IN": ["VERB", "AUX"]}
#                 },
#             "REL_OP": "<"             
#         }
#     ]

#     pattern_1_plur = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=1"]}
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=1"]},
#                 "POS": {"IN": ["VERB", "AUX"]}
#                 },
#             "REL_OP": "<"             
#         }
#     ]

#     pattern_2_plur = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=2"]}
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=2"]},
#                 "POS": {"IN": ["VERB", "AUX"]}
#                 },
#             "REL_OP": "<"             
#         }
#     ]

#     pattern_3_plur = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=3"]}
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=3"]},
#                 "POS": {"IN": ["VERB", "AUX"]}
#                 },
#             "REL_OP": "<"             
#         }
#     ]

#     pattern_3_sing_masc = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "MORPH": "Gender=Masc|Number=Sing"
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=3"]},
#                 "POS": {"IN": ["VERB", "AUX"]}
#                 },
#             "REL_OP": "<"             
#         }
#     ]

#     pattern_3_sing_fem = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "MORPH": "Gender=Fem|Number=Sing"
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=3"]},
#                 "POS": {"IN": ["VERB", "AUX"]}
#                 },
#             "REL_OP": "<"             
#         }
#     ]

    
#     matcher.add("b2d3_5_A2", [pattern_3_sing, pattern_2_sing, pattern_1_sing,
#                              pattern_3_plur, pattern_2_plur, pattern_1_plur,
#                              pattern_3_sing_masc, pattern_3_sing_fem])

#     matches = matcher(doc)
    
#     results = []
#     for match_id, token_ids in matches:
        
#         subject_token = None
#         for token_index in token_ids:
#             if doc[token_index].dep_ == 'nsubj':
#                 subject_token = doc[token_index]
#                 break

#         if subject_token:
#             has_conj_child = any(child.dep_ == 'conj' for child in subject_token.children)
            
#             if not has_conj_child:
#                 start = min(doc[i].i for i in token_ids)
#                 end = max(doc[i].i for i in token_ids) + 1
#                 tokens_in_span = doc[start:end]
                
#                 text = reconstruct_text(tokens_in_span)
#                 results.append((doc.vocab.strings[match_id], text, start, end))

#     return results


# def b2d3_6(doc):
#     """
#     B2.3-6: Concordância (Suj./Pred.) - entre sujeito e verbo (sujeito composto pré-verbal com e sem 1.ª pessoa)
#     e.g., Eu e a Ana vivemos na mesma rua. Tu e Ana vivem na mesma rua.
#     Level: A2
#     """
#     matcher = DependencyMatcher(nlp_stanza.vocab)

#     pattern_com_1 = [
#         {
#             "RIGHT_ID": "eu",
#             "RIGHT_ATTRS": {
#                 "NORM": {"IN": ["eu", "nos"]}
#                 }
#         },
#         {
#             "LEFT_ID": "eu",
#             "RIGHT_ID": "conj",
#             "RIGHT_ATTRS": {
#                 "DEP": "conj",
#                 "POS": {"IN": ["NOUN", "PROPN", "PRON"]}
#                 },
#             "REL_OP": ">++"             
#         },
#         {
#             "LEFT_ID": "eu",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "POS": {"IN": ["VERB", "AUX"]},
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=1"]}
#                 },
#             "REL_OP": "<++"             
#         }
#     ]

#     pattern_sem_1 = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "NORM": {"NOT_IN": ["eu", "nos"]},
#                 "POS": {"IN": ["PROPN", "PRON"]}
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "conj",
#             "RIGHT_ATTRS": {
#                 "DEP": "conj",
#                 "POS": {"IN": ["NOUN", "PROPN", "PRON"]}
#                 },
#             "REL_OP": ">++"             
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "POS": {"IN": ["VERB", "AUX"]},
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=3"]}
#                 },
#             "REL_OP": "<++"             
#         }
#     ]

#     # To match "eu e os meus colegas temos trabalhado num novo modelo."
#     pattern_com_1_aux = [
#         {
#             "RIGHT_ID": "eu",
#             "RIGHT_ATTRS": {
#                 "NORM": {"IN": ["eu", "nos"]}
#                 }
#         },
#         {
#             "LEFT_ID": "eu",
#             "RIGHT_ID": "conj",
#             "RIGHT_ATTRS": {
#                 "DEP": "conj",
#                 "POS": {"IN": ["NOUN", "PROPN", "PRON"]}
#                 },
#             "REL_OP": ">++"             
#         },
#         {
#             "LEFT_ID": "eu",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "POS": {"IN": ["VERB", "AUX"]},
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=1"]}
#                 },
#             "REL_OP": "$++"             
#         }
#     ]

#     pattern_sem_1_aux = [
#         {
#             "RIGHT_ID": "subject",
#             "RIGHT_ATTRS": {
#                 "DEP": "nsubj",
#                 "NORM": {"NOT_IN": ["eu", "nos"]},
#                 "POS": {"IN": ["PROPN", "PRON"]}
#                 }
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "conj",
#             "RIGHT_ATTRS": {
#                 "DEP": "conj",
#                 "POS": {"IN": ["NOUN", "PROPN", "PRON"]}
#                 },
#             "REL_OP": ">++"             
#         },
#         {
#             "LEFT_ID": "subject",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "POS": {"IN": ["VERB", "AUX"]},
#                 "MORPH": {"IS_SUPERSET": ["Number=Plur", "Person=3"]}
#                 },
#             "REL_OP": "$++"             
#         }
#     ]


    
#     matcher.add("b2d3_6_A2", [pattern_com_1, pattern_sem_1,
#                               pattern_com_1_aux, pattern_sem_1_aux])

#     matches = matcher(doc)
    
#     results = []
#     for match_id, token_ids in matches:
#         # Calculate start and end indices
#         start = min(token_ids)
#         end = max(token_ids) + 1
#         tokens_in_span = doc[start:end]
        
#         text = reconstruct_text(tokens_in_span)

#         results.append((doc.vocab.strings[match_id], text, start, end))

#     return results


def b2d3_8(doc):
    """
    B2.3-8: Concordância (Suj./Pred.) - com o verbo haver com sentido de existência
    e.g., Havia muitos turistas no museu. / *Haviam muitos turistas no museu.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "haver",
            "RIGHT_ATTRS": {
                "LEMMA": "haver",
                "MORPH": {"IS_SUPERSET": ["Number=Sing", "Person=3"]}
                }
        },
        {
            "LEFT_ID": "haver",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON"]},
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]}
                },
            "REL_OP": ">++"             
        }
    ]
    
    matcher.add("b2d3_8_A2", [pattern])

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


def b2d3_13(doc):
    """
    B2.3-13: Concordância (Suj./Pred.) - com o verbo parecer
    e.g., Os alunos parecem gostar de português. / Os alunos parece que gostam da professora.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern_1_sing = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "parecer",
            "RIGHT_ATTRS": {
                "LEMMA": "parecer",
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]},
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "parecer",
            "RIGHT_ID": "inf",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
                "DEP": "xcomp"
                },
            "REL_OP": ">++"             
        }
    ]

    pattern_1_plur = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "parecer",
            "RIGHT_ATTRS": {
                "LEMMA": "parecer",
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]},
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "parecer",
            "RIGHT_ID": "inf",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
                "DEP": "xcomp"
                },
            "REL_OP": ">++"             
        }
    ]

    pattern_2_sing = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "parecer",
            "RIGHT_ATTRS": {
                "LEMMA": "parecer",
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]},
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "parecer",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Number=Sing"]},
                "DEP": "ccomp"
                },
            "REL_OP": ">++"             
        }
    ]

    pattern_2_plur = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "parecer",
            "RIGHT_ATTRS": {
                "LEMMA": "parecer",
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]},
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "parecer",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Number=Plur"]},
                "DEP": "ccomp"
                },
            "REL_OP": ">++"             
        }
    ]

    pattern_1_sing_cop = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "parecer",
            "RIGHT_ATTRS": {
                "LEMMA": "parecer",
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]},
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "parecer",
            "RIGHT_ID": "inf",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
                "DEP": {"IN": ["cop", "aux"]}
                },
            "REL_OP": "."             
        }
    ]

    pattern_1_plur_cop = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "parecer",
            "RIGHT_ATTRS": {
                "LEMMA": "parecer",
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]},
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "parecer",
            "RIGHT_ID": "inf",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
                "DEP": {"IN": ["cop", "aux"]}
                },
            "REL_OP": "."             
        }
    ]

    pattern_2_sing_cop = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "parecer",
            "RIGHT_ATTRS": {
                "LEMMA": "parecer",
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]},
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "parecer",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Number=Sing"]},
                "DEP": {"IN": ["cop", "aux"]}
                },
            "REL_OP": ".*"             
        }
    ]

    pattern_2_plur_cop = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]}
                }
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "parecer",
            "RIGHT_ATTRS": {
                "LEMMA": "parecer",
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]},
                },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "parecer",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Number=Plur"]},
                "DEP": {"IN": ["cop", "aux"]}
                },
            "REL_OP": ".*"             
        }
    ]
    
    matcher.add("b2d3_13_B1", [pattern_1_plur, pattern_1_sing,
                             pattern_2_plur, pattern_2_sing,
                             pattern_1_sing_cop, pattern_1_plur_cop,
                             pattern_2_sing_cop, pattern_2_plur_cop])

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

def b2d3_14(doc):
    """
    B2.3-14: Concordância (Suj./Pred.) - entre sujeito e particípio passado em passivas
    e.g. Os doces foram comidos pelas crianças.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj_sing = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]},
                # So that we don't match "em tempos idos, as celebrações eram organizadas de forma muito diferente."
                "DEP": "nsubj:pass"}
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da)$"}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "adj",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"LEMMA": {"IN": ["ser", "estar"]},
                            "MORPH": {"IS_SUPERSET": ["Number=Sing"]}},
            "REL_OP": ">--"
        },
    ]
    
    # In case it's identified as a verb
    pattern_verb_sing = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Sing"]},
                "DEP": "nsubj:pass"}
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part", "Number=Sing"]}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"LEMMA": {"IN": ["ser", "estar"]},
                            "MORPH": {"IS_SUPERSET": ["Number=Sing"]}},
            "REL_OP": ">--"
        },
    ]

    pattern_adj_plur = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]},
                "DEP": "nsubj:pass"}
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:dos|das)$"}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "adj",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"LEMMA": {"IN": ["ser", "estar"]},
                            "MORPH": {"IS_SUPERSET": ["Number=Plur"]}},
            "REL_OP": ">--"
        },
    ]
    
    # In case it's identified as a verb
    pattern_verb_plur = [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]},
                "MORPH": {"IS_SUPERSET": ["Number=Plur"]},
                "DEP": "nsubj:pass"}
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part", "Number=Plur"]}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"LEMMA": {"IN": ["ser", "estar"]},
                            "MORPH": {"IS_SUPERSET": ["Number=Plur"]}},
            "REL_OP": ">--"
        },
    ]
    
    matcher.add("b2d3_14_B1", [pattern_adj_plur, pattern_adj_sing,
                               pattern_verb_plur, pattern_verb_sing])

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


def b2d3_18(doc):
    """
    B2.3-18: Concordância (Suj./Pred.) - entre o verbo e o predicativo do sujeito (com o verbo ser) - com isto, isso, tudo
    e.g. Isso são disparates. Tudo eram dificuldades.
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    pattern = [
        {
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["isto", "isso", "aquilo", "tudo"]}
            }
        },
        {
            "LEFT_ID": "pron",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
            "POS": "NOUN",
            "MORPH": {"IS_SUPERSET": ["Number=Plur"]}
            },
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "ser",
            "RIGHT_ATTRS": {"LEMMA": "ser",
                            "MORPH": {"IS_SUPERSET": ["Number=Plur"]}},
            "REL_OP": ">--"
        },
    ]
    
    matcher.add("b2d3_18_B2", [pattern])

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