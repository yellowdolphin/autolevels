import subprocess
import pytest
from pathlib import Path
from PIL import ImageCms
import numpy as np
import cv2


# Define path to the test image
TEST_IMAGE = 'images/lübeck.jpg'
DEFAULT_OUTPUT_IMAGE_PATH = Path('images/lübeck_al.jpg')
MODEL = 'models/free_test.pt'

# Create and save an sRGB ICC profile
ICC_PROFILE = "images/sRGB.ICC"
srgb_profile = ImageCms.createProfile("sRGB")
srgb_profile = ImageCms.ImageCmsProfile(srgb_profile)
with open(ICC_PROFILE, "wb") as icc_file:
    icc_file.write(srgb_profile.tobytes())

# Create a minimal 48-bit RGB image (2x2 pixels, 16-bit per channel)
image_data = np.array([
    [[65535, 0, 0], [0, 65535, 0]],
    [[0, 0, 65535], [65535, 65535, 65535]]
], dtype=np.uint16)

# Save 48bit image as PNG and TIFF
PNG_IMAGE = "images/48bit_rgb.png"
TIFF_IMAGE = "images/48bit_rgb.tiff"
cv2.imwrite(PNG_IMAGE, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
cv2.imwrite(TIFF_IMAGE, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))


@pytest.fixture(autouse=True)
def remove_output_image():
    """Fixture to remove output image if it exists before/after each test."""
    DEFAULT_OUTPUT_IMAGE_PATH.unlink(missing_ok=True)
    yield
    DEFAULT_OUTPUT_IMAGE_PATH.unlink(missing_ok=True)


def run_autolevels(args):
    """Helper function to run the script with given args."""
    result = subprocess.run(f'autolevels {args}'.split(), capture_output=True, text=True)
    return result


def test_no_args():
    """Test usage is shown if no args."""
    result = run_autolevels('')
    assert result.returncode == 1
    assert 'No files specified' in result.stderr
    assert result.stdout.startswith('usage: autolevels')


def test_help_option():
    """Test --help option to display help information."""
    result = run_autolevels('--help')
    assert result.returncode == 0
    assert result.stdout.startswith("usage: ")


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_version_option(simulate):
    """Test --version option to print version information."""
    from autolevels.autolevels import __version__
    result = run_autolevels(f'{simulate} --version')
    assert result.returncode == 0
    assert result.stdout == f"AutoLevels version {__version__}\n"


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_default_run(simulate):
    """Test autolevels with default options."""
    result = run_autolevels(f'{simulate} --outdir images -- {TEST_IMAGE}')
    assert result.returncode == 0
    if simulate:
        assert 'black point: [111  97 115] -> [81 67 85]' in result.stdout
        assert 'white point: [254 251 248] -> [254 251 248]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_blackpoint_option(simulate):
    """Test --blackpoint option with single and RGB values."""
    result = run_autolevels(f'{simulate} --outdir images --blackpoint 10 --mode smooth -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'black point: [72 57 58] -> [42 27 28]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --blackpoint 10 --mode smooth --maxblack 75 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'black point: [72 57 58] -> [10 10 10]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --blackpoint 0 14 255 --mode smooth --maxblack 75 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    assert 'black point: [72 57 58] -> [ 0 14 58]' in result.stdout


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_whitepoint_option(simulate):
    """Test --whitepoint option with single and RGB values."""
    result = run_autolevels(f'{simulate} --outdir images --whitepoint 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'white point: [254 251 248] -> [255 255 255]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --whitepoint 200 210 252 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    assert 'white point: [254 251 248] -> [254 251 252]' in result.stdout


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_blackclip_whiteclip_options(simulate):
    """Test --blackclip and --whiteclip options with various percentages."""
    result = run_autolevels(f'{simulate} --outdir images --blackclip 0.007 --whiteclip 0.003 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    if simulate:
        assert 'black point: [127 110 129]' in result.stdout
        assert 'white point: [251 251 247]' in result.stdout
    else:
        assert 'high black point: [127 110 129]' in result.stdout


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_maxblack_minwhite_options(simulate):
    """Test --maxblack and --minwhite options with L and RGB values."""
    result = run_autolevels(f'{simulate} --outdir images --max-blackshift 10 --maxblack 100 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'black point: [111  97 115] -> [101  87 105]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --max-blackshift 10 --maxblack 120 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'black point: [111  97 115] -> [14 14 14]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

    result = run_autolevels(f'{simulate} --outdir images --whitepoint 255 --minwhite 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'white point: [254 251 248] -> [255 252 249]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --whitepoint 255 --minwhite 255 --max-whiteshift 0 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert ('white point: [254 251 248] -> [254 251 248]' in result.stdout) or ('white point:' not in result.stdout)
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --whitepoint 255 --minwhite 200 --max-whiteshift 0 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert ('white point: [254 251 248] -> [254 251 248]' in result.stdout) or ('white point:' not in result.stdout)
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --whitepoint 255 --minwhite 255 --max-whiteshift 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'white point: [254 251 248] -> [255 252 249]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --whitepoint 255 --minwhite 200 --max-whiteshift 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'white point: [254 251 248] -> [255 255 255]' in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --minwhite 200 --max-whiteshift 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert ('white point: [254 251 248] -> [254 251 248]' in result.stdout) or ('white point:' not in result.stdout)
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_mode_option(simulate):
    """Test --mode option with all valid values."""
    for mode in ["smooth", "smoother", "hist", "perceptive"]:
        result = run_autolevels(f'{simulate} --outdir images --mode {mode} -- {TEST_IMAGE}')
        assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_gamma_option(simulate):
    """Test --gamma option with L and RGB values."""
    result = run_autolevels(f'{simulate} --outdir images --gamma 1.2 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir images --gamma 1.0 0.8 1.2 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_saturation_options(simulate):
    """Test saturation-related options."""
    for wensat in ["", "--saturation-first", "--saturation-before-gamma"]:
        result = run_autolevels(f'{simulate} --outdir images {wensat} --saturation 0.0 -- {TEST_IMAGE}')
        assert result.returncode == 0
        assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_output_options(simulate):
    """Test file location options folder, prefix, suffix, etc."""
    output_fn = DEFAULT_OUTPUT_IMAGE_PATH.parent / 'tmp' / 'koblenz.jpg'
    output_fn.unlink(missing_ok=True)
    print("output_fn:", output_fn)
    result = run_autolevels(f'{simulate} --folder images --prefix lü --suffix eck.jpg '
                            '--outdir images/tmp --outprefix ko --outsuffix lenz.jpg -- b')
    assert result.returncode == 0
    assert output_fn.exists() != bool(simulate)
    if simulate:
        assert ' -> images/tmp/koblenz.jpg' in result.stdout
    output_fn.unlink(missing_ok=True)
    if output_fn.parent.exists():
        Path(output_fn.parent).rmdir()


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_fstring_options(simulate):
    """Test --fstring options"""
    output_fn = DEFAULT_OUTPUT_IMAGE_PATH.parent / 'tmp' / 'koblenz.jpg'
    output_fn.unlink(missing_ok=True)
    result = run_autolevels(simulate + ' --folder images --fstring f"lü{x:^.1s}eck.jpg" --outfstring "ko{x:<.1s}lenz.jpg" '
                            '--outdir images/tmp -- b')
    assert result.returncode == 0
    assert output_fn.exists() != bool(simulate)
    if simulate:
        assert ' -> images/tmp/koblenz.jpg' in result.stdout
    output_fn.unlink(missing_ok=True)
    if output_fn.parent.exists():
        Path(output_fn.parent).rmdir()


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_glob_pattern(simulate):
    """Test glob patterns like *.jpg"""
    result = run_autolevels(simulate + ' --outdir images --mode smooth --folder images -- *.jpg')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_reproduce_option(simulate):
    """Test --reproduce option using a previous output image."""
    repro_options = "--blackpoint 42 --whitepoint 242 252 255 --mode smooth --saturation 0.8 --max-whiteshift 3"
    _ = run_autolevels(f'{repro_options} --outdir images --outsuffix _previous.jpg -- {TEST_IMAGE}')
    previous_image = DEFAULT_OUTPUT_IMAGE_PATH.parent / DEFAULT_OUTPUT_IMAGE_PATH.name.replace('_al', '_previous')
    assert previous_image.exists()
    result = run_autolevels(f'{simulate} --outdir images --reproduce {previous_image} -- {TEST_IMAGE}')
    previous_image.unlink()
    assert result.returncode == 0
    assert repro_options in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_model_option(simulate):
    """Test --model option using free curve inference with MODEL."""
    result = run_autolevels(f'{simulate} --outdir images --model {MODEL} -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_48bit_images(simulate):
    """Test --model option with 48bit images."""
    for fn in (PNG_IMAGE, TIFF_IMAGE):
        output_image_path = Path(fn).parent / (Path(fn).stem + '_al' + Path(fn).suffix)
        Path(output_image_path).unlink(missing_ok=True)
        result = run_autolevels(f'{simulate} --outdir images --model {MODEL} -- {fn}')
        assert result.returncode == 0
        assert output_image_path.exists() != bool(simulate)
        Path(output_image_path).unlink(missing_ok=True)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_icc_option(simulate):
    """Test --icc-profile option with 48bit images."""
    for fn in (PNG_IMAGE, TIFF_IMAGE):
        output_image_path = Path(fn).parent / (Path(fn).stem + '_al.jpg')
        Path(output_image_path).unlink(missing_ok=True)
        result = run_autolevels(f'{simulate} --outdir images --outsuffix _al.jpg --icc-profile {ICC_PROFILE} -- {fn}')
        assert result.returncode == 0
        assert output_image_path.exists() != bool(simulate)
        Path(output_image_path).unlink(missing_ok=True)
