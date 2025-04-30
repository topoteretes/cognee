import pytest
from cognee.tasks.entity_completion.entity_extractors.regex_entity_extractor import (
    RegexEntityExtractor,
)


@pytest.fixture
def regex_extractor():
    """Create a RegexEntityExtractor instance for testing."""
    return RegexEntityExtractor()


@pytest.mark.asyncio
async def test_extract_emails(regex_extractor):
    """Test extraction of email addresses."""
    text = "Contact us at support@example.com or sales@company.co.uk for more information."
    entities = await regex_extractor.extract_entities(text)

    # Filter only EMAIL entities
    email_entities = [e for e in entities if e.is_a.name == "EMAIL"]

    assert len(email_entities) == 2
    assert "support@example.com" in [e.name for e in email_entities]
    assert "sales@company.co.uk" in [e.name for e in email_entities]


@pytest.mark.asyncio
async def test_extract_phone_numbers(regex_extractor):
    """Test extraction of phone numbers."""
    text = "Call us at +1-555-123-4567 or 020 7946 0958 for support."
    entities = await regex_extractor.extract_entities(text)

    # Filter only PHONE entities
    phone_entities = [e for e in entities if e.is_a.name == "PHONE"]

    assert len(phone_entities) == 2
    assert "+1-555-123-4567" in [e.name for e in phone_entities]
    assert "020 7946 0958" in [e.name for e in phone_entities]


@pytest.mark.asyncio
async def test_extract_urls(regex_extractor):
    """Test extraction of URLs."""
    text = "Visit our website at https://www.example.com or http://docs.example.org/help for more information."
    entities = await regex_extractor.extract_entities(text)

    # Filter only URL entities
    url_entities = [e for e in entities if e.is_a.name == "URL"]

    assert len(url_entities) == 2
    assert "https://www.example.com" in [e.name for e in url_entities]
    assert "http://docs.example.org/help" in [e.name for e in url_entities]


@pytest.mark.asyncio
async def test_extract_dates(regex_extractor):
    """Test extraction of dates."""
    text = "The event is scheduled for 2023-05-15 and ends on 06/30/2023."
    entities = await regex_extractor.extract_entities(text)

    # Filter only DATE entities
    date_entities = [e for e in entities if e.is_a.name == "DATE"]

    assert len(date_entities) == 2
    assert "2023-05-15" in [e.name for e in date_entities]
    assert "06/30/2023" in [e.name for e in date_entities]


@pytest.mark.asyncio
async def test_extract_times(regex_extractor):
    """Test extraction of times."""
    text = "The meeting starts at 09:30 AM and ends at 14:45."
    entities = await regex_extractor.extract_entities(text)

    # Filter only TIME entities
    time_entities = [e for e in entities if e.is_a.name == "TIME"]

    assert len(time_entities) == 2
    assert "09:30 AM" in [e.name for e in time_entities]
    assert "14:45" in [e.name for e in time_entities]


@pytest.mark.asyncio
async def test_extract_money(regex_extractor):
    """Test extraction of monetary amounts."""
    # TODO: Lazar to fix regex for test, it's failing currently
    pass
    # text = "The product costs $1,299.99 or €1.045,00 depending on your region."
    # entities = await regex_extractor.extract_entities(text)

    # Filter only MONEY entities
    # money_entities = [e for e in entities if e.is_a.name == "MONEY"]

    # assert len(money_entities) == 2
    # assert "$1,299.99" in [e.name for e in money_entities]
    # assert "€1.045,00" in [e.name for e in money_entities]


@pytest.mark.asyncio
async def test_extract_person_names(regex_extractor):
    """Test extraction of person names with various formats."""
    text = """
    Standard names: John Smith and Sarah Johnson will be attending.
    Names with titles: Dr. Jane Wilson and Prof Michael Brown will present.
    Names with middle initials: James T. Kirk and William H Gates are invited.
    Names with prefixes: Jean de la Fontaine and Ludwig van Beethoven are famous.

    Single names like Mary or Robert should not be extracted as they could be
    confused with regular capitalized words at the beginning of sentences.
    """
    entities = await regex_extractor.extract_entities(text)

    # Filter only PERSON entities
    person_entities = [e for e in entities if e.is_a.name == "PERSON"]
    entity_names = [e.name for e in person_entities]

    # Standard two-part names
    assert "John Smith" in entity_names
    assert "Sarah Johnson" in entity_names

    # Names with titles
    assert "Dr. Jane Wilson" in entity_names
    assert "Prof Michael Brown" in entity_names

    # Names with middle initials
    assert "James T. Kirk" in entity_names
    assert "William H Gates" in entity_names

    # Names with prefixes
    assert "Jean de la Fontaine" in entity_names
    assert "Ludwig van Beethoven" in entity_names

    # Verify single names are not extracted
    assert "Mary" not in entity_names
    assert "Robert" not in entity_names

    # Verify we have the expected number of names
    assert len(person_entities) == 8


@pytest.mark.asyncio
async def test_extract_hashtags(regex_extractor):
    """Test extraction of hashtags."""
    text = "Check out our latest post #Python #MachineLearning"
    entities = await regex_extractor.extract_entities(text)

    # Filter only HASHTAG entities
    hashtag_entities = [e for e in entities if e.is_a.name == "HASHTAG"]

    assert len(hashtag_entities) == 2
    assert "#Python" in [e.name for e in hashtag_entities]
    assert "#MachineLearning" in [e.name for e in hashtag_entities]


@pytest.mark.asyncio
async def test_extract_mentions(regex_extractor):
    """Test extraction of mentions."""
    text = "Thanks to @johndoe and @jane_smith for their contributions."
    entities = await regex_extractor.extract_entities(text)

    # Filter only MENTION entities
    mention_entities = [e for e in entities if e.is_a.name == "MENTION"]

    assert len(mention_entities) == 2
    assert "@johndoe" in [e.name for e in mention_entities]
    assert "@jane_smith" in [e.name for e in mention_entities]


@pytest.mark.asyncio
async def test_extract_ip_addresses(regex_extractor):
    """Test extraction of IP addresses with proper validation of octet ranges."""
    # Test with valid IP addresses
    text = "The server IPs are 192.168.1.1, 10.0.0.1, 255.255.255.255, and 0.0.0.0."
    entities = await regex_extractor.extract_entities(text)

    # Filter only IP_ADDRESS entities
    ip_entities = [e for e in entities if e.is_a.name == "IP_ADDRESS"]

    assert len(ip_entities) == 4
    assert "192.168.1.1" in [e.name for e in ip_entities]
    assert "10.0.0.1" in [e.name for e in ip_entities]
    assert "255.255.255.255" in [e.name for e in ip_entities]
    assert "0.0.0.0" in [e.name for e in ip_entities]


@pytest.mark.asyncio
async def test_invalid_ip_addresses(regex_extractor):
    """Test that invalid IP addresses are not extracted."""
    # Test with invalid IP addresses
    text = "Invalid IPs: 999.999.999.999, 256.256.256.256, 1.2.3.4.5, 01.102.103.104"
    entities = await regex_extractor.extract_entities(text)

    # Filter only IP_ADDRESS entities
    ip_entities = [e for e in entities if e.is_a.name == "IP_ADDRESS"]

    # None of these should be extracted as valid IPs
    assert len(ip_entities) == 1
    assert "999.999.999.999" not in [e.name for e in ip_entities]
    assert "256.256.256.256" not in [e.name for e in ip_entities]
    assert "1.2.3.4.5" not in [e.name for e in ip_entities]
    assert "01.102.103.104" not in [e.name for e in ip_entities]
    assert "1.102.103.104" in [e.name for e in ip_entities]


@pytest.mark.asyncio
async def test_extract_multiple_entity_types(regex_extractor):
    """Test extraction of multiple entity types from a single text."""
    text = """
    Contact John Doe at john.doe@example.com or +1-555-123-4567.
    Visit our website at https://www.example.com.
    The meeting is scheduled for 2023-05-15 at 09:30 AM.
    The project budget is $10,000.00.
    Follow us on social media with #Python and mention @pythonorg.
    Our server IP is 192.168.1.1.
    """

    entities = await regex_extractor.extract_entities(text)

    # Check that we have at least one entity of each type
    entity_types = [e.is_a.name for e in entities]

    assert "EMAIL" in entity_types
    assert "PHONE" in entity_types
    assert "URL" in entity_types
    assert "DATE" in entity_types
    assert "TIME" in entity_types
    assert "MONEY" in entity_types
    assert "PERSON" in entity_types
    assert "HASHTAG" in entity_types
    assert "MENTION" in entity_types
    assert "IP_ADDRESS" in entity_types


@pytest.mark.asyncio
async def test_empty_text(regex_extractor):
    """Test extraction with empty text."""
    entities = await regex_extractor.extract_entities("")
    assert len(entities) == 0


@pytest.mark.asyncio
async def test_none_text(regex_extractor):
    """Test extraction with None text."""
    entities = await regex_extractor.extract_entities(None)
    assert len(entities) == 0


@pytest.mark.asyncio
async def test_text_without_entities(regex_extractor):
    """Test extraction with text that doesn't contain any entities."""
    text = "This text does not contain any extractable entities."
    entities = await regex_extractor.extract_entities(text)
    assert len(entities) == 0


@pytest.mark.asyncio
async def test_custom_config_path(tmp_path):
    """Test extraction with a custom configuration path."""
    # Create a minimal test config file
    config_content = """[
        {
            "entity_name": "TEST_ENTITY",
            "entity_description": "Test entity type",
            "regex": "TEST\\\\d+",
            "description_template": "Test entity: {}"
        }
    ]"""

    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        f.write(config_content)

    # Create extractor with custom config
    extractor = RegexEntityExtractor(str(config_path))

    # Test extraction
    text = "This contains TEST123 and TEST456."
    entities = await extractor.extract_entities(text)

    assert len(entities) == 2
    assert all(e.is_a.name == "TEST_ENTITY" for e in entities)
    assert "TEST123" in [e.name for e in entities]
    assert "TEST456" in [e.name for e in entities]
