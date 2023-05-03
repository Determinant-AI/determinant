import pytest
from data.privacy_processor import PrivacyProcessor

@pytest.fixture
def pp():
    return PrivacyProcessor()

def test_process_privacy_issues_default_config(pp):
    text = """
    Hi, my name is Alice White, and my email is alicewhite@example.com.
    Please call me at +1 (123) 456-7890. My website is https://alicewhite.com.
    I was born on 01/01/1990. My company is Rainbird Furniture Inc. My IP address is 192.168.1.1.
    My credit card number is 1234-5678-9012-3456. My SSN is 123-45-6789.
    """

    expected_output = """
    Hi, my name is [REDACTED NAME], and my email is [REDACTED EMAIL].
    Please call me at [REDACTED PHONE]. My website is [REDACTED URL].
    I was born on [REDACTED DATE]. My company is [REDACTED ORG]. My IP address is [REDACTED IP].
    My credit card number is [REDACTED CREDIT CARD]. My SSN is [REDACTED SSN].
    """

    assert pp.process_privacy_issues(text) == expected_output

def test_process_privacy_issues_custom_config(pp):
    text = """
    Hi, my name is Alice White, and my email is alicewhite@example.com.
    Please call me at +1 (123) 456-7890. My website is https://alicewhite.com.
    I was born on 01/01/1990. My company is Rainbird Furniture Inc. My IP address is 192.168.1.1.
    My credit card number is 1234-5678-9012-3456. My SSN is 123-45-6789.
    """

    expected_output = """
    Hi, my name is Alice White, and my email is [REDACTED EMAIL].
    Please call me at [REDACTED PHONE]. My website is https://alicewhite.com.
    I was born on 01/01/1990. My company is Rainbird Furniture Inc. My IP address is [REDACTED IP].
    My credit card number is 1234-5678-9012-3456. My SSN is 123-45-6789.
    """

    custom_config = {
        'email': True,
        'phone': True,
        'ip_address': True,
        'url': False,
        'credit_card': False,
        'ssn': False,
        'name': False,
        'date': False,
        'organization': True
    }

    assert pp.process_privacy_issues(text, custom_config) == expected_output
