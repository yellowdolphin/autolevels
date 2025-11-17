import pytest
from autolevels import make_comment
from autolevels.export import (fit_rgb_curves, compute_monotone_hermite_slopes, hermite_eval, create_basic_xmp, 
                               check_missing, append_rgbcurve_history_item, local_name, check_darktable_version)
from autolevels import process_channel
from PIL import Image
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


RANDOM_VERSION = "3.14.15"
namespaces = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'darktable': 'http://darktable.sf.net/',
    'xmpMM': 'http://ns.adobe.com/xap/1.0/mm/',
}


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


def get_support_curves():
    curves = np.zeros((3, 256), dtype=np.float32)
    gammas = 1.8, 0.9, 0.2
    for c, gamma in enumerate(gammas):
        for i in range(256):
            curves[c, i] = np.clip((i / 240.0) ** gamma - 4/256, 0, 1)
        # print(sum(curves[c] == 0), sum(curves[c] == 1))
    support = np.linspace(0, 1, 256, dtype=np.float32)
    return support, curves


def test_fit_rgb_curves_linear():
    support, curves = get_support_curves()
    MAX_POINTS = (10, 30)
    MAX_ERROR = 2e-4
    for max_points in MAX_POINTS:
        results = fit_rgb_curves(curves, max_points=max_points, max_error=MAX_ERROR)
        assert len(results) == 3  # channels
        for ch, ch_result in enumerate(results):
            assert isinstance(ch_result, np.ndarray)
            assert ch_result.shape[1] == 2  # x, y

            # calculate max fit error
            # with given MAX_POINTS, one max_err should be below, one above MAX_ERROR
            x, y = ch_result[:, 0], ch_result[:, 1]
            m = compute_monotone_hermite_slopes(x, y)
            y_fit = hermite_eval(support, x, y, m)
            assert len(y_fit) == 256
            max_err = (np.abs(y_fit - curves[ch])).max()
            assert (max_err <= MAX_ERROR) or (ch_result.shape[0] == max_points)
            if max_points == 30:
                assert max_err <= MAX_ERROR, f'max_points: {max_points}, max_err: {max_err}'
            else:
                assert max_err > MAX_ERROR, f'max_points: {max_points}, max_err: {max_err}'


def test_create_basic_xmp_writes_file(tmp_path):
    xmp_file = tmp_path / "test.xmp"
    with Image.open("images/lübeck.jpg") as pil_img:
        create_basic_xmp(xmp_file, pil_img)

        assert xmp_file.exists()

        # Verify proper UTF-8 encoding
        with open(xmp_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify content of basic XMP
        tree = ET.parse(xmp_file)
        root = tree.getroot()
        description = root.find('.//rdf:Description', namespaces)

        mtime = datetime.fromtimestamp(Path("images/lübeck.jpg").stat().st_mtime, tz=timezone.utc).strftime("%Y:%m:%d %H:%M:%S.%f")[:-3]
        assert description.get('{http://ns.adobe.com/exif/1.0/}DateTimeOriginal') == mtime
        assert description.get('{http://ns.adobe.com/xap/1.0/mm/}DerivedFrom') == Path(pil_img.filename).name
        assert description.get('{http://darktable.sf.net/}import_timestamp') != '-1'

        missing, rgbcurve_num, history_end = check_missing(xmp_file)
        assert missing == {'iop_order_list', 'auto_presets', 'history_basic_hash', 'history_current_hash'}
        assert rgbcurve_num == '4', f'{rgbcurve_num}'  # anticipating default presets applied
        assert history_end == '5', f'{history_end}'  # anticipating default presets applied

        # Test append_rgbcurve_history_item() using created basic XMP file
        support, curves = get_support_curves()
        append_rgbcurve_history_item(xmp_file, curves, pil_img, icc=None, new_xmp_file=None)

    # Verify content of final XMP
    tree = ET.parse(xmp_file)
    root = tree.getroot()
    description = root.find('.//rdf:Description', namespaces)

    assert description.get('{http://darktable.sf.net/}auto_presets_applied') == '1'
    assert description.get('{http://darktable.sf.net/}history_end') == '5'
    assert description.get('{http://darktable.sf.net/}iop_order_version') == '0'
    assert 'rgbcurve,1,colorin,0' in description.get('{http://darktable.sf.net/}iop_order_list')

    missing, rgbcurve_num, history_end = check_missing(xmp_file)
    assert len(missing) == 0
    assert description.get('{http://darktable.sf.net/}history_basic_hash') == '33e4711b8f6644f5f8c2a164fa3f94cd'
    assert description.get('{http://darktable.sf.net/}history_current_hash') == '6cbed05a9150be22123901a023a7ca8c'  # hash from darktable

    xmp_file.unlink()  # Clean up after test


def test_check_darktable_version():
    with pytest.raises(ValueError):
        check_darktable_version('invalid')
    with pytest.raises(ValueError):
        check_darktable_version('4.7.9')
    assert check_darktable_version('6.0') is None


def test_append_rgbcurve_history_item():
    xmp_file = Path("test.xmp")
    if xmp_file.exists():
        xmp_file.unlink()  # Ensure file does not exist before test
    assert not xmp_file.exists()

    with Image.open("images/adobeRGB.jpg") as pil_img:
        support, curves = get_support_curves()
        append_rgbcurve_history_item(xmp_file, curves, pil_img, icc=None, new_xmp_file=None)

    # Verify content of final XMP
    tree = ET.parse(xmp_file)
    root = tree.getroot()

    description = root.find('.//rdf:Description', namespaces)
    history_data = []
    history_seq = root.find('.//darktable:history/rdf:Seq', namespaces)
    if history_seq is not None:
        for li in history_seq.findall('rdf:li', namespaces):
            entry_data = {}

            for key, value in li.items():
                local_key = local_name(key)
                entry_data[local_key] = value

            history_data.append(entry_data)
            
    assert description.get('{http://darktable.sf.net/}history_basic_hash') == "02d4cdbda625305c5e181669466f51d2"
    for li in history_data:
        if li['operation'] == 'colorin':
            assert li['params'] == "gz48eJzjZBgFowABWAbaAaNgwAEAMNgADg=="

    xmp_file.unlink()  # Clean up after test


def test_process_channel():
    a = np.array([[5, 128, 255]], dtype=np.uint8)
    channel = Image.fromarray(a)
    L = np.array([[0., 0., 0.]])
    pix_black = 0
    pix_white = 0
    bp, wp = process_channel(pix_black, pix_white, channel, L, norm=None)
    assert bp == 5
    assert wp == 255
    pix_white = 0.4
    bp, wp = process_channel(pix_black, pix_white, channel, L, norm=None)
    assert bp == 5
    assert wp == 128