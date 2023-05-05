import re
import uuid
import spacy


class PrivacyProcessor:
    """
    Redact, De-Identify, and Anonymize Personally Identifiable Information (PII) in text.
    """

    def __init__(self):
        # Load the SpaCy model
        self.nlp = spacy.load('en_core_web_sm')

    def process(self, text: str, config=None) -> str:
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
        phone_pattern = r'(?:\+?\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
        ip_address_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        credit_card_pattern = r'(?:\d{4}[- ]?){3}\d{4}'
        ssn_pattern = r'\b(\d{3}[- ]?\d{2}[- ]?\d{4})\b'

        # TODO: Define regular expressions for PHI pattern

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

        # # Check for presence of credit card numbers
        # has_credit_card = bool(re.search(credit_card_pattern, text))

        # # Perform named entity recognition using SpaCy (excluding credit card numbers if present)
        # if has_credit_card:
        #     doc = self.nlp(re.sub(credit_card_pattern, 'CREDIT CARD NUMBER', text))
        # else:
        #     doc = self.nlp(text)

        # Perform named entity recognition using SpaCy
        doc = self.nlp(text)

        # Generate a mapping of unique IDs for anonymization
        anonymization_map = {}

        # Redact and anonymize named entities based on configuration
        for ent in doc.ents:
            print(ent.text, ent.label_)
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
            print(text)

        return text
