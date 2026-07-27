import re

def reconstruct_text(doc):
    """This function reconstructs the string based on the identified span by taking into
    account the changes made due to the tokenization process of Stanza."""
    
    # Reconstruct the text without extra spaces before punctuation,
    # using a state machine to distinguish opening and closing quotation marks.
    text = ''
    in_quote = False
    quotes = {'"', "'"}
    
    for token in doc:
        if token.is_punct and token.text in quotes:
            # Toggle state: if not already in a quote, treat as opening quote;
            # otherwise treat as closing quote.
            if not in_quote:
                # Opening quote: attach without altering preceding whitespace.
                text += token.text
                in_quote = True
            else:
                # Closing quote: remove any trailing whitespace so that the quote
                # attaches to the preceding token, then add the token.
                text = text.rstrip()
                text += token.text
                in_quote = False
                if token.whitespace_:
                    text += token.whitespace_
        elif token.is_punct:
            # For other punctuation, remove any trailing whitespace and attach the punctuation.
            text = text.rstrip()
            text += token.text_with_ws
        else:
            text += token.text_with_ws

    text = text.strip()
    
    # Replacement rules for contractions and other language-specific fixes.
    replacements = [
        (r'\ba as\b', 'às'),
        (r'\ba os\b', 'aos'),
        (r'\ba a\b', 'à'),
        (r'\ba o\b', 'ao'),
        (r'\bem o\b', 'no'),
        (r'\bem a\b', 'na'),
        (r'\bem os\b', 'nos'),
        (r'\bem as\b', 'nas'),
        (r'\bem ele', 'nele'),
        (r'\bem ela', 'nela'),
        (r'\bem eles', 'neles'),
        (r'\bem elas', 'nelas'),
        (r'\bem um', 'num'),
        (r'\bem uma', 'numa'),
        (r'\bpor o\b', 'pelo'),
        (r'\bpor a\b', 'pela'),
        (r'\bpor os\b', 'pelos'),
        (r'\bpor as\b', 'pelas'),
        (r'\bde o\b', 'do'),
        (r'\bde a\b', 'da'),
        (r'\bde os\b', 'dos'),
        (r'\bde as\b', 'das'),
        (r'\bA as\b', 'Às'),
        (r'\bA os\b', 'Àos'),
        (r'\bA a\b', 'À'),
        (r'\bA o\b', 'Ào'),
        (r'\bEm o\b', 'No'),
        (r'\bEm a\b', 'Na'),
        (r'\bEm os\b', 'Nos'),
        (r'\bEm as\b', 'Nas'),
        (r'\bEm um', 'Num'),
        (r'\bEm uma', 'Numa'),
        (r'\bPor o\b', 'Pelo'),
        (r'\bPor a\b', 'Pela'),
        (r'\bPor os\b', 'Pelos'),
        (r'\bPor as\b', 'Pelas'),
        (r'\bDe o\b', 'Do'),
        (r'\bDe a\b', 'Da'),
        (r'\bDe os\b', 'Dos'),
        (r'\bDe as\b', 'Das'),
        (r'\bde este\b', 'deste'),
        (r'\bde esse\b', 'desse'),
        (r'\bde aquele\b', 'daquele'),
        (r'\bde esta\b', 'desta'),
        (r'\bde essa\b', 'dessa'),
        (r'\bde aquela\b', 'daquela'),
        (r'\bde estes\b', 'destes'),
        (r'\bde esses\b', 'desses'),
        (r'\bde aqueles\b', 'daqueles'),
        (r'\bde estas\b', 'destas'),
        (r'\bde essas\b', 'dessas'),
        (r'\bde aquelas\b', 'daquelas'),
        (r'\bde isto\b', 'disto'),
        (r'\bde isso\b', 'disso'),
        (r'\bde ele\b', 'dele'),
        (r'\bde ela\b', 'dela'),
        (r'\bde eles\b', 'deles'),
        (r'\bde elas\b', 'delas'),
        (r'\bem este\b', 'neste'),
        (r'\bem esse\b', 'nesse'),
        (r'\bem aquele\b', 'naquele'),
        (r'\bem esta\b', 'nesta'),
        (r'\bem essa\b', 'nessa'),
        (r'\bem aquela\b', 'naquela'),
        (r'\bem estes\b', 'nestes'),
        (r'\bem esses\b', 'nesses'),
        (r'\bem aqueles\b', 'naqueles'),
        (r'\bem estas\b', 'nestas'),
        (r'\bem essas\b', 'nessas'),
        (r'\bem aquelas\b', 'naquelas'),
        (r'\ba aquele\b', 'àquele'),
        (r'\ba aquela\b', 'àquela'),
        (r'\ba aqueles\b', 'àqueles'),
        (r'\ba aquelas\b', 'àquelas'),
        (r'\ba aquilo\b', 'àquilo'),
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    # Post-processing: Insert a hyphen between a verb and personal pronouns
    clitic_pronouns = ["me", "te", "se", "o", "a", "nos", "vos", "os", "as", "lhe", "lhes", "lo", "la", "los", "las"]
    for i, token in enumerate(doc):
        if token.text.lower() in clitic_pronouns and "PronType=Prs" in token.morph:
            if i > 0 and doc[i-1].pos_ == "VERB":
                text = re.sub(rf'\b{doc[i-1].text}\s+{token.text}\b', f'{doc[i-1].text}-{token.text}', text)
      
    # Fix spaces before punctuation: Remove extra spaces before punctuation marks.
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)

    # Ensure spaces before and after em dash (—)
    text = re.sub(r'\s*—\s*', ' — ', text)
    text = re.sub(r'\s{2,}', ' ', text)  # Remove any double spaces created

    # Ensure space before '(' and no space after '('
    text = re.sub(r'(\S)\s*\(\s*', r'\1 (', text)

    # Insert a space after punctuation when needed, but do not insert extra spaces if a quote follows.
    def insert_space_after(match):
        punct = match.group(1)
        following = match.group(2)
        if following in {'"', "'"}:
            return punct + following
        if punct == '.' and following.isdigit():
            return punct + following
        return punct + ' ' + following

    text = re.sub(r'([,.!?;:])([^\s])', insert_space_after, text)
    
    return text
