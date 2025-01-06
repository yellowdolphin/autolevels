import pytest
from autolevels import make_comment
from PIL import Image


@pytest.fixture
def minimal_image():
    """Create a minimal PIL.Image for testing."""
    img = Image.new('RGB', (10, 10), 'white')
    return img


def test_make_comment():
    """Tests calling make_comment"""


def test_make_comment_with_no_existing_comment(minimal_image):
    version = "1.0.0"
    cli_params = "--example"
    result = make_comment(minimal_image, version, cli_params)

    assert result == "autolevels 1.0.0, params: --example"

def test_make_comment_with_existing_comment(minimal_image):
    minimal_image.info['comment'] = b"Existing comment"

    version = "1.0.0"
    cli_params = "--example"
    result = make_comment(minimal_image, version, cli_params)

    assert result == "Existing comment\nautolevels 1.0.0, params: --example"

def test_make_comment_with_non_decodable_comment(minimal_image):
    minimal_image.info['comment'] = b"\x80\x81\x82"

    version = "1.0.0"
    cli_params = "--example"
    result = make_comment(minimal_image, version, cli_params)

    assert result == "autolevels 1.0.0, params: --example"
