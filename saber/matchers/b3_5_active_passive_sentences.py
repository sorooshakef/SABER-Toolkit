from spacy.matcher import Matcher, DependencyMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def b3d5_2(doc):
    """
    B3.5-2: frase passiva - passiva adjetival - com auxiliar estar + particípio passado,
    e.g. As portas estão fechadas. O trabalho está terminado.
    Level: A2
    Requires nlp_stanza
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "estar",
            "RIGHT_ATTRS": {"LEMMA": "estar"}
        },
        {
            "LEFT_ID": "estar",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<++"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "estar",
            "RIGHT_ATTRS": {"LEMMA": "estar"}
        },
        {
            "LEFT_ID": "estar",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<++"             
        }
    ]


    matcher.add("b3d5_2_A2", [pattern_adj, pattern_verb])

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

def b3d5_3(doc):
    """
    B3.5-3: "frase passiva - passiva periferástica - o sujeito sofre a ação expressa pelo verbo
    - com auxiliar ser + particípio passado
    - com complemento agente da passiva expresso"
    e.g. Um livro foi lido às crianças pelo professor.
    Level: B1
    Requires nlp_stanza
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    pattern = [
        {
            "RIGHT_ID": "ser",
            "RIGHT_ATTRS": {
                "LEMMA": "ser"
            }
        },
        {
            "LEFT_ID": "ser",
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}
                },
            "REL_OP": "<++"                
        },
        {
            "LEFT_ID": "part",
            "RIGHT_ID": "agent",
            "RIGHT_ATTRS": {
                "DEP": "obl:agent"
            },
            "REL_OP": ">++" 
        }
    ]
    
    matcher.add("b3d5_3_B1", [pattern])

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

def b3d5_4(doc):
    """
    B3.5-4: "frase passiva - passiva perifrástica - o sujeito sofre a ação expressa pelo verbo
    - com auxiliar ser + particípio passado
    - sem complemento agente da passiva expresso
    e.g. Um livro foi lido às crianças.
    Level: B1
    Requires nlp_stanza
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    pattern = [
        {
            "RIGHT_ID": "ser",
            "RIGHT_ATTRS": {
                "LEMMA": "ser",
                # To prevent "deve ser feita" from being captured. Why not though? "Deve ser feita" is passive!
                # Removing "VerbForm=Fin"condition.
                "DEP": {"NOT_IN": ["advcl"]}  # To prevent "Se não fosse por terem seguido o plano à risca, o projeto poderia ter falhado." from matching. 
                # Adding "DEP": {"NOT_IN": ["advcl"]} instead.
            }
        },
        {
            "LEFT_ID": "ser",
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}
                },
            "REL_OP": "<++"                
        }
    ]

    # To match "Após a inspeção, algumas falhas foram identificadas e corrigidas."
    pattern_conj = [
        {
            "RIGHT_ID": "ser",
            "RIGHT_ATTRS": {
                "LEMMA": "ser",
                "DEP": {"NOT_IN": ["advcl"]}  # To prevent "Se não fosse por terem seguido o plano à risca, o projeto poderia ter falhado." from matching.
            }
        },
        {
            "LEFT_ID": "ser",
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}
                },
            "REL_OP": "<++"                
        },
        {
            "LEFT_ID": "part",
            "RIGHT_ID": "conj",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                "DEP": "conj"
                },
            "REL_OP": ">++"                
        }
    ]
    
    matcher.add("b3d5_4_B1", [pattern, pattern_conj])

    matches = matcher(doc)
    
    results = []
    for match_id, token_ids in matches:
        # Extract the part token (the second token in the pattern)
        part_token = doc[token_ids[1]]
        
        # Check if any children of the participle are agents (obl:agent)
        has_agent = any(child.dep_ == "obl:agent" for child in part_token.children)
        if has_agent:
            continue  # Skip matches that have an agent

        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]

        # Reconstruct the text without extra spaces before punctuation
        text = reconstruct_text(tokens_in_span)
        
        results.append((doc.vocab.strings[match_id], text, start, end))
    
    return results

def b3d5_5(doc):

    """
    B3.5-5: "frase passiva - passiva reflexa ***This should be impessoal
    - formada com o pronome se
    - com complemento agente da passiva indeterminado *** sujeito inditerminado
    - no singular"
    e.g. Vende-se automóvel.
    Level: B2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
        {"NORM": "se", "POS": "PRON"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "ADP", "LOWER": "de", "OP": "?"},
        {"POS": {"IN": ["DET", "ADJ", "ADV"]}, "OP": "*"},
        {"POS": "NOUN", "MORPH": {"IS_SUPERSET": ["Number=Sing"]}}
        ]
    
    pattern_neg = [
        {"NORM": "nao"},
        {"NORM": "se", "POS": "PRON"},
        {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
        {"POS": "ADV", "OP": "*"},
        {"POS": "ADP", "LOWER": "de", "OP": "?"},
        {"POS": {"IN": ["DET", "ADJ", "ADV"]}, "OP": "*"},
        {"POS": "NOUN", "MORPH": {"IS_SUPERSET": ["Number=Sing"]}}
    ]
    
    matcher.add("b3d5_5_B2", [pattern, pattern_neg])

    matches = matcher(doc)
    
    def fix_match_text(span):

        text = reconstruct_text(span)

        words = text.split()
        fixed_words = []
        for word in words:
            if word.startswith("<UNK>"):
                # Replace only the prefix "<UNK>" with "G"
                fixed_word = "G" + word[len("<UNK>"):]
                fixed_words.append(fixed_word)
            else:
                fixed_words.append(word)
        text = " ".join(fixed_words)
        return text

    fixed_matches = [
        (doc.vocab.strings[match_id], fix_match_text(doc[start:end]), start, end)
        for match_id, start, end in matches
    ]

    return fixed_matches

def b3d5_6(doc):
    """
    B3.5-6: "frase passiva - passiva reflexa
    - formada com o pronome se
    - com complemento agente da passiva indeterminado
    - no plural"
    e.g. Vendem-se automóveis.
    Level: B2
    
    Stanza somehow tokenizes "constroem-se" as "constromem" and "se", with an extra M in "constroem".
    Interestingly, this is not an issue if the alternative spelling ("construem") is used.
    It also sometimes doesn't respect capitalization and changes "Vendem-se" to "vendem" and "se".
    This causes an issue with text reconstruction.
    There is a similar issue with <UNK> being displayed instead of G because of Stanza's MWT.

    The fixes to the issue above include using the pre-MWT tokens from Stanza, but unless we preprocess each text
    twice (which can significantly increase the run-time) using the pre-MWT tokens not only breaks many of the
    existing functions, but it also causes issues for the other preprocessing steps, including dependency parsing
    and the lemmatization.

    Since the issues resulting from MWT are limited, it would be more eficient to fix them in an ad-hoc fashion.
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Number=Plur"]}},
        {"NORM": "se", "POS": "PRON"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "ADP", "LOWER": "de", "OP": "?"},
        {"POS": {"IN": ["DET", "ADJ", "ADV"]}, "OP": "*"},
        {"POS": "NOUN", "MORPH": {"IS_SUPERSET": ["Number=Plur"]}}
        ]
    
    pattern_neg = [
        {"NORM": "nao"},
        {"NORM": "se", "POS": "PRON"},
        {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
        {"POS": "ADV", "OP": "*"},
        {"POS": "ADP", "LOWER": "de", "OP": "?"},
        {"POS": {"IN": ["DET", "ADJ", "ADV"]}, "OP": "*"},
        {"POS": "NOUN", "MORPH": {"IS_SUPERSET": ["Number=Plur"]}} 
    ]
    
    matcher.add("b3d5_6_B2", [pattern, pattern_neg])

    matches = matcher(doc)

    # Alternatively, we could be more strict and force the function to convert to upper-case all
    # instances of this structure that are at the beginning of the sentence and start with a 
    # lower-case character, but I think that's too strict; unless we come to realize that the issue
    # with "vendem" was more common than we had realized.

    # def fix_match_text(span):
    # text = reconstruct_text(span)
    # # Fix the "construem" issue:
    # text = text.replace("Constromem", "Constroem").replace("constromem", "constroem")
    # # If the span is at sentence start, fix the first non-whitespace character if needed.
    # if span[0].is_sent_start:
    #     stripped = text.lstrip()
    #     leading_ws = text[:len(text) - len(stripped)]
    #     if stripped and stripped[0].islower():
    #         stripped = stripped[0].upper() + stripped[1:]
    #     text = leading_ws + stripped
    # return text


    def fix_match_text(span):

        text = reconstruct_text(span)

        words = text.split()
        fixed_words = []
        for word in words:
            if word.startswith("<UNK>"):
                # Replace only the prefix "<UNK>" with "G"
                fixed_word = "G" + word[len("<UNK>"):]
                fixed_words.append(fixed_word)
            else:
                fixed_words.append(word)
        text = " ".join(fixed_words)

        # Fix the "construem" issue:
        text = text.replace("Constromem", "Constroem").replace("constromem", "constroem")
        # If the first token of the span is "vendem" (in lowercase) and is marked as sentence-start,
        # capitalize its first letter.
        if span[0].is_sent_start and span[0].text.lower() == "vendem":
            # Preserve any leading whitespace
            stripped = text.lstrip()
            leading_ws = text[:len(text) - len(stripped)]
            if stripped.startswith("v"):
                stripped = "V" + stripped[1:]
            text = leading_ws + stripped
        return text

    fixed_matches = [
        (doc.vocab.strings[match_id], fix_match_text(doc[start:end]), start, end)
        for match_id, start, end in matches
    ]

    return fixed_matches