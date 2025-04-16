import pytest
from autolevels import make_comment
from PIL import Image

RANDOM_VERSION = "3.14.15"

@pytest.fixture
def minimal_image():
    """Create a minimal PIL.Image for testing."""
    img = Image.new('RGB', (10, 10), 'white')
    return img


def test_make_comment():
    """Tests calling make_comment"""


def test_make_comment_with_no_existing_comment(minimal_image):
    cli_params = "--example"
    result = make_comment(minimal_image, RANDOM_VERSION, cli_params)

    assert result == f"autolevels {RANDOM_VERSION}, params: --example"

def test_make_comment_with_existing_comment(minimal_image):
    minimal_image.info['comment'] = b"Existing comment"

    cli_params = "--example"
    result = make_comment(minimal_image, RANDOM_VERSION, cli_params)

    assert result == f"Existing comment\nautolevels {RANDOM_VERSION}, params: --example"

def test_make_comment_with_non_decodable_comment(minimal_image):
    minimal_image.info['comment'] = b"\x80\x81\x82"

    cli_params = "--example"
    result = make_comment(minimal_image, RANDOM_VERSION, cli_params)

    assert result == f"autolevels {RANDOM_VERSION}, params: --example"
