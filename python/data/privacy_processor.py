import re
import spacy
import uuid

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

def process_privacy_issues(text: str, config=None) -> str:
    """
    Redact and anonymize named entities and PII patterns in text.
    """
    if config is None:
        config = {
            'email': True,
            'phone': True,
            'ip_address': True,
            'url': True,
            'credit_card': True,
            'ssn': True,
            'name': True,
            'date': True,
            'organization': True
        }

    # Define regular expressions for common PII patterns
    email_pattern = r'(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    phone_pattern = r'(?:\+\d{1,2}\s)?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,4}'
    ip_address_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    credit_card_pattern = r'(?:\d{4}[-\s]?){3}\d{4}'
    ssn_pattern = r'\d{3}[-\s]?\d{2}[-\s]?\d{4}'

    # Perform named entity recognition using SpaCy
    doc = nlp(text)

    # Generate a mapping of unique IDs for anonymization
    anonymization_map = {}

    # Redact and anonymize named entities based on configuration
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and config.get('name', True):
            # Anonymize names of people
            if ent.text not in anonymization_map:
                anonymization_map[ent.text] = str(uuid.uuid4())
            text = text.replace(ent.text, anonymization_map[ent.text])

        elif ent.label_ == 'DATE' and config.get('date', True):
            # Redact dates
            text = text.replace(ent.text, '[REDACTED DATE]')

        elif ent.label_ == 'ORG' and config.get('organization', True):
            # Redact organization names
            text = text.replace(ent.text, '[REDACTED ORG]')

    # Redact PII using regex patterns based on configuration
    if config.get('email', True):
        text = re.sub(email_pattern, '[REDACTED EMAIL]', text)
    if config.get('phone', True):
        text = re.sub(phone_pattern, '[REDACTED PHONE]', text)
    if config.get('ip_address', True):
        text = re.sub(ip_address_pattern, '[REDACTED IP]', text)
    if config.get('url', True):
        text = re.sub(url_pattern, '[REDACTED URL]', text)
    if config.get('credit_card', True):
        text = re.sub(credit_card_pattern, '[REDACTED CREDIT CARD]', text)
    if config.get('ssn', True):
        text = re.sub(ssn_pattern, '[REDACTED SSN]', text)

    return text


if __name__ == "__main__":
    # Example text with named entities and PII patterns
    text = """
    Hi, my name is John Doe, and my email is johndoe@example.com.
    Please call me at +1 (123) 456-7890. My website is https://johndoe.com.
    I was born on 01/01/1990. My company is Acme Corp. My IP address is 192.168.1.1.
    My credit card number is 1234-5678-9012-3456. My SSN is 123-45-6789.
    """

    # Process text for privacy issues (default configuration)
    processed_text = process_privacy_issues(text)
    print(processed_text)

    # Process text with custom configuration (only redact email and phone)
    custom_config = {
        'email': True,
        'phone': True,
        'ip_address': False,
        'url': False,
        'credit_card': False,
        'ssn': False,
        'name': False,
        'date': False,
        'organization': False
    }
    processed_text_custom = process_privacy_issues(text, custom_config)
    print(processed_text_custom)
