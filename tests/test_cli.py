import subprocess
import pytest
from pathlib import Path

# Define path to the test image
TEST_IMAGE = "images/lübeck.jpg"
DEFAULT_OUTPUT_IMAGE_PATH = Path('images/lübeck_al.jpg')

@pytest.fixture(autouse=True)
def remove_output_image():
    """Fixture to remove output image if it exists before/after each test."""
    DEFAULT_OUTPUT_IMAGE_PATH.unlink(missing_ok=True)
    yield
    DEFAULT_OUTPUT_IMAGE_PATH.unlink(missing_ok=True)

def run_autolevels(args):
    """Helper function to run the script with given args."""
    result = subprocess.run(f'./autolevels.py {args}'.split(), capture_output=True, text=True)
    return result

def test_usage_help():
    """Test usage is shown if no args."""
    result = run_autolevels(f'')
    assert result.returncode == 1
    assert 'No files specified' in result.stderr
    assert result.stdout.startswith('usage: autolevels.py')

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_default_run(simulate):
    """Test autolevels with default options."""
    result = run_autolevels(f'{simulate} -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_blackpoint_option(simulate):
    """Test --blackpoint option with single and RGB values."""
    result = run_autolevels(f'{simulate} --blackpoint 10 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --blackpoint 10 20 30 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_whitepoint_option(simulate):
    """Test --whitepoint option with single and RGB values."""
    result = run_autolevels(f'{simulate} --whitepoint 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --whitepoint 200 210 220 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_blackclip_whiteclip_options(simulate):
    """Test --blackclip and --whiteclip options with various percentages."""
    result = run_autolevels(f'{simulate} --blackclip 0.007 --whiteclip 0.003 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_maxblack_minwhite_options(simulate):
    """Test --maxblack and --minwhite options with L and RGB values."""
    result = run_autolevels(f'{simulate} --maxblack 75 --minwhite 240 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --maxblack 50 60 70 --minwhite 230 240 250 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_mode_option(simulate):
    """Test --mode option with all valid values."""
    for mode in ["smooth", "smoother", "hist", "perceptive"]:
        result = run_autolevels(f'{simulate} --mode {mode} -- {TEST_IMAGE}')
        assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_gamma_option(simulate):
    """Test --gamma option with L and RGB values."""
    result = run_autolevels(f'{simulate} --gamma 1.2 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --gamma 1.0 0.8 1.2 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_saturation_options(simulate):
    """Test saturation-related options."""
    for wensat in ["", "--saturation-first", "--saturation-before-gamma"]:
        result = run_autolevels(f'{simulate} {wensat} --saturation 0.0 -- {TEST_IMAGE}')
        assert result.returncode == 0
        assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_output_options(simulate):
    """Test file location options folder, prefix, suffix, etc."""
    output_fn = DEFAULT_OUTPUT_IMAGE_PATH.parent / 'tmp' / 'koblenz.jpg'
    output_fn.unlink(missing_ok=True)
    result = run_autolevels(f'{simulate} --folder images --prefix lü --suffix eck.jpg --outdir images/tmp --outprefix ko --outsuffix lenz.jpg -- b')
    assert result.returncode == 0
    assert output_fn.exists() != bool(simulate)
    if simulate:
        assert ' -> images/tmp/koblenz.jpg' in result.stdout
    output_fn.unlink(missing_ok=True)
    if output_fn.parent.exists():
        Path(output_fn.parent).rmdir()

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_reproduce_option(simulate):
    """Test --reproduce option using a previous output image."""
    repro_options = "--blackpoint 42 --whitepoint 242 252 255 --mode smoother --saturation 0.8 --max-whiteshift 3"
    _ = run_autolevels(f'{repro_options} --outsuffix _previous.jpg -- {TEST_IMAGE}')
    previous_image = DEFAULT_OUTPUT_IMAGE_PATH.parent / DEFAULT_OUTPUT_IMAGE_PATH.name.replace('_al', '_previous')
    assert previous_image.exists()
    result = run_autolevels(f'{simulate} --reproduce {previous_image} -- {TEST_IMAGE}')
    previous_image.unlink()
    assert result.returncode == 0
    assert repro_options in result.stdout
    assert DEFAULT_OUTPUT_IMAGE_PATH.exists() != bool(simulate)

@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_version_option(simulate):
    """Test --version option to print version information."""
    from autolevels import __version__
    result = run_autolevels(f'{simulate} --version')
    assert result.returncode == 0
    assert result.stdout == f"AutoLevels version {__version__}\n"

def test_help_option():
    """Test --help option to display help information."""
    result = run_autolevels(f'--help')
    assert result.returncode == 0
    assert result.stdout.startswith("usage: ")


