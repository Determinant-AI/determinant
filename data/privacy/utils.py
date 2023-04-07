import re
import spacy

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Define a function to redact named entities (people, email addresses, phone numbers)
def redact_named_entities(text):
    # Define regular expressions for email and phone number patterns
    email_pattern = r'(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    phone_pattern = r'(?:\+\d{1,2}\s)?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,4}'

    # Redact email addresses
    text = re.sub(email_pattern, '[REDACTED EMAIL]', text)

    # Redact phone numbers
    text = re.sub(phone_pattern, '[REDACTED PHONE]', text)

    # Perform named entity recognition using SpaCy
    doc = nlp(text)

    # Redact names of people
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            text = text.replace(ent.text, '[REDACTED NAME]')

    return text


if __name__ == "__main__":
    # Example text with named entities
    text = 'Hi, my name is John Doe, and my email is johndoe@example.com. Please call me at +1 (123) 456-7890.'

    # Redact named entities from the text
    redacted_text = redact_named_entities(text)
    print(redacted_text)


# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_sm

