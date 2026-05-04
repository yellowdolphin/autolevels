import shutil
import subprocess
from autolevels.export import iop_order_list
import pytest
from pathlib import Path
from typing import Any
from PIL import ImageCms
import numpy as np
import cv2
import exiftool


# Define path to the test image
TEST_IMAGE = 'images/lübeck.jpg'
MODEL = 'models/free_test.pt'
ONNX_MODEL = 'models/free_test.onnx'

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

# Images with ICC and MakerNote
JPEG_WITH_ICC_MAKERNOTE = "images/adobeRGB.jpg"
TIFF_WITH_ICC_MAKERNOTE = "images/adobeRGB.tif"
PNG_WITH_ICC_MAKERNOTE = "images/adobeRGB.png"


def write_metadata_tags(et: exiftool.ExifToolHelper, path: Path, groups: dict[str, tuple[str, str]]) -> None:
    """Inject one tag per metadata group into *path* (overwrites in place)."""
    payload = {tag: value for group, (tag, value) in groups.items()
               if group not in {"MakerNotes", "ICC"}}
    assert path.exists(), f"Path {path} does not exist"
    et.set_tags(str(path), payload, params=["-overwrite_original"])


def read_metadata_tags(et: exiftool.ExifToolHelper, path: Path) -> dict[str, Any]:
    """
    Return the single-file metadata dict for *path*.

    Raise an error if *path* does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"read_metadata_tags: path not found: {path}")
    return et.get_metadata(str(path))[0]


def get_metadata_value(meta: dict[str, Any], group: str, tag: str) -> Any:
    """
    Retrieve a tag from exiftool's metadata dict.

    ExifToolHelper may key tags as "Group:Name" *or* just "Name" depending
    on the -G flag usage, so we try both forms.
    """
    value = meta.get(f"{group}:{tag}")
    return meta.get(tag) if value is None else value


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
def test_default_run(simulate, tmp_path):
    """Test autolevels with default options."""
    output_image_path = tmp_path / (Path(TEST_IMAGE).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {tmp_path} -- {TEST_IMAGE}')
    assert result.returncode == 0
    if simulate:
        assert 'black point: [111  97 115] -> [81 67 85]' in result.stdout
        assert 'white point: [254 251 248] -> [254 251 248]' in result.stdout
    assert output_image_path.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_blackpoint_option(simulate, tmp_path):
    """Test --blackpoint option with single and RGB values."""
    output_image_path = tmp_path / (Path(TEST_IMAGE).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --blackpoint 10 --mode smooth -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'black point: [72 57 58] -> [42 27 28]' in result.stdout
    assert output_image_path.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --blackpoint 10 --mode smooth --maxblack 75 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'black point: [72 57 58] -> [10 10 10]' in result.stdout
    assert output_image_path.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --blackpoint 0 14 255 --mode smooth --maxblack 75 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert output_image_path.exists() != bool(simulate)
    assert 'black point: [72 57 58] -> [ 0 14 58]' in result.stdout


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_whitepoint_option(simulate, tmp_path):
    """Test --whitepoint option with single and RGB values."""
    output_image_path = tmp_path / (Path(TEST_IMAGE).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --whitepoint 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'white point: [254 251 248] -> [255 255 255]' in result.stdout
    assert output_image_path.exists() != bool(simulate)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --whitepoint 200 210 252 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert output_image_path.exists() != bool(simulate)
    assert 'white point: [254 251 248] -> [254 251 252]' in result.stdout


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_blackclip_whiteclip_options(simulate, tmp_path):
    """Test --blackclip and --whiteclip options with various percentages."""
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --blackclip 0.007 --whiteclip 0.003 -- {TEST_IMAGE}')
    output_image_path = tmp_path / (Path(TEST_IMAGE).stem + '_al.jpg')
    assert result.returncode == 0
    assert output_image_path.exists() != bool(simulate)
    if simulate:
        assert 'black point: [127 110 129]' in result.stdout
        assert 'white point: [251 251 247]' in result.stdout
    else:
        assert 'high black point: [127 110 129]' in result.stdout


def test_blackclip_whiteclip_edge_cases(tmp_path):
    """Test high --blackclip and --whiteclip."""
    for mode in ['hist', 'perceptive']:
        result = run_autolevels(f'--simulate --outdir {tmp_path} --blackpoint 0 --whitepoint 255 '
                                '--blackclip 1 --whiteclip 1 '
                                '--maxblack 255 --minwhite 0 '
                                f'--max-blackshift 255 --max-whiteshift 255 --mode {mode} -- {TEST_IMAGE}')
        output_image_path = tmp_path / (Path(TEST_IMAGE).stem + '_al.jpg')
        print("tested mode:", mode)
        print("stdout:", result.stdout)
        assert result.returncode == 0
        assert output_image_path.exists() is False
        assert 'black point: [255 255 255] -> [0 0 0]' in result.stdout
        assert 'white point: [0 0 0] -> [255 255 255]' in result.stdout


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_maxblack_minwhite_options(simulate, tmp_path):
    """Test --maxblack and --minwhite options with L and RGB values."""
    output_image_path = tmp_path / (Path(TEST_IMAGE).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --max-blackshift 10 --maxblack 100 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'black point: [111  97 115] -> [101  87 105]' in result.stdout
    assert output_image_path.exists() != bool(simulate)  # max-blackshift applies to all/no channel
    output_image_path.unlink(missing_ok=True)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --max-blackshift 10 --maxblack 120 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'black point: [111  97 115] -> [14 14 14]' in result.stdout
    assert output_image_path.exists() != bool(simulate)   # max-blackshift applies only beyond maxblack
    output_image_path.unlink(missing_ok=True)

    result = run_autolevels(f'{simulate} --outdir {tmp_path} --whitepoint 255 --minwhite 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'white point: [254 251 248] -> [255 252 249]' in result.stdout
    assert output_image_path.exists() != bool(simulate)
    output_image_path.unlink(missing_ok=True)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --whitepoint 255 --minwhite 255 --max-whiteshift 0 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert ('white point: [254 251 248] -> [254 251 248]' in result.stdout) or ('white point:' not in result.stdout)
    assert output_image_path.exists() != bool(simulate)
    output_image_path.unlink(missing_ok=True)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --whitepoint 255 --minwhite 200 --max-whiteshift 0 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert ('white point: [254 251 248] -> [254 251 248]' in result.stdout) or ('white point:' not in result.stdout)
    assert output_image_path.exists() != bool(simulate)   # max-whiteshift always applies
    output_image_path.unlink(missing_ok=True)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --whitepoint 255 --minwhite 255 --max-whiteshift 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'white point: [254 251 248] -> [255 252 249]' in result.stdout  # preserve hue, saturation
    assert output_image_path.exists() != bool(simulate)
    output_image_path.unlink(missing_ok=True)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --whitepoint 255 --minwhite 200 --max-whiteshift 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert 'white point: [254 251 248] -> [255 255 255]' in result.stdout
    assert output_image_path.exists() != bool(simulate)
    output_image_path.unlink(missing_ok=True)
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --minwhite 200 --max-whiteshift 255 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert ('white point: [254 251 248] -> [254 251 248]' in result.stdout) or ('white point:' not in result.stdout)
    assert output_image_path.exists() != bool(simulate)
    output_image_path.unlink(missing_ok=True)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_mode_option(simulate, tmp_path):
    outdir = tmp_path
    fn = Path(TEST_IMAGE)
    output_image_path = outdir / (fn.stem + '_al.jpg')
    """Test --mode option with all valid values."""
    for mode in ["smooth", "smoother", "hist", "perceptive"]:
        result = run_autolevels(f'{simulate} --outdir {outdir} --mode {mode} -- {fn}')
        assert result.returncode == 0
        assert output_image_path.exists() != bool(simulate)
        output_image_path.unlink(missing_ok=True)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_gamma_option(simulate, tmp_path):
    """Test --gamma option with L and RGB values."""
    outdir = tmp_path
    fn = Path(TEST_IMAGE)
    output_image_path = outdir / (fn.stem + '_al.jpg')
    for gamma in ('1.2', '1.0 0.8 1.2'):
        result = run_autolevels(f'{simulate} --outdir {outdir} --gamma {gamma} -- {fn}')
        assert result.returncode == 0
        assert output_image_path.exists() != bool(simulate)
        output_image_path.unlink(missing_ok=True)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_saturation_options(simulate, tmp_path):
    """Test saturation-related options."""
    outdir = tmp_path
    fn = Path(TEST_IMAGE)
    output_image_path = outdir / (fn.stem + '_al.jpg')
    for wensat in ["", "--saturation-first", "--saturation-before-gamma"]:
        result = run_autolevels(f'{simulate} --outdir {outdir} {wensat} --saturation 0.0 -- {fn}')
        assert result.returncode == 0
        assert output_image_path.exists() != bool(simulate)
        output_image_path.unlink(missing_ok=True)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_output_options(simulate, tmp_path):
    """Test file location options folder, prefix, suffix, etc."""
    outdir = tmp_path
    output_image_path = outdir / 'koblenz.jpg'
    result = run_autolevels(f'{simulate} --folder images --prefix lü --suffix eck.jpg '
                            f'--outdir {outdir} --outprefix ko --outsuffix lenz.jpg -- b')
    assert result.returncode == 0
    assert output_image_path.exists() != bool(simulate)
    if simulate:
        assert f' -> {output_image_path}' in result.stdout


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_fstring_options(simulate, tmp_path):
    """Test --fstring options"""
    outdir = tmp_path
    output_image_path = outdir / 'koblenz.jpg'
    result = run_autolevels(f'{simulate} --outdir {outdir} --folder images '
                            '--fstring    f"lü{x:^.1s}eck.jpg" '
                            '--outfstring "ko{x:<.1s}lenz.jpg" '
                            f'-- b')
    assert result.returncode == 0
    assert output_image_path.exists() != bool(simulate)
    if simulate:
        assert f' -> {output_image_path}' in result.stdout


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_glob_pattern(simulate, tmp_path):
    """Test glob patterns like *.jpg"""
    outdir = tmp_path
    result = run_autolevels(f'{simulate} --outdir {outdir} --mode smooth --folder images -- *.jpg')
    assert result.returncode == 0
    for fn in Path('images').glob('*.jpg'):
        output_image_path = outdir / (fn.stem + '_al.jpg')
        assert output_image_path.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_reproduce_option(simulate, tmp_path):
    """Test --reproduce option using a previous output image."""
    outdir = tmp_path
    outsuffix = '_previous.jpg'
    output_image_path = outdir / (Path(TEST_IMAGE).stem + outsuffix)
    repro_options = "--blackpoint 42 --whitepoint 242 252 255 --mode smooth --saturation 0.8 --max-whiteshift 3"
    _ = run_autolevels(f'{repro_options} --outdir {outdir} --outsuffix {outsuffix} -- {TEST_IMAGE}')
    previous_image = output_image_path
    assert previous_image.exists()
    output_image_path = outdir / (Path(TEST_IMAGE).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {outdir} --reproduce {previous_image} -- {TEST_IMAGE}')
    previous_image.unlink()
    assert result.returncode == 0
    assert repro_options in result.stdout
    assert output_image_path.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_model_option(simulate, tmp_path):
    """Test --model option using free curve inference with MODEL."""
    outdir = tmp_path
    output_image_path = outdir / (Path(TEST_IMAGE).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {outdir} --model {MODEL} -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert output_image_path.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_model_option_with_saturation_first(simulate, tmp_path):
    """Test --model and --saturation-first options."""
    outdir = tmp_path
    output_image_path = outdir / (Path(TEST_IMAGE).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {outdir} --model {MODEL} --saturation-first --saturation 0.8 -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert output_image_path.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_onnx(simulate, tmp_path):
    """Test --model option using onnx instead of torch."""
    outdir = tmp_path
    output_image_path = outdir / (Path(TEST_IMAGE).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {outdir} --model {ONNX_MODEL} -- {TEST_IMAGE}')
    assert result.returncode == 0
    assert output_image_path.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_48bit_images(simulate, tmp_path):
    """Test --model option with 48bit images."""
    for fn in (PNG_IMAGE, TIFF_IMAGE):
        outdir = tmp_path
        output_image_path = outdir / (Path(fn).stem + '_al' + Path(fn).suffix)
        output_image_path.unlink(missing_ok=True)
        result = run_autolevels(f'{simulate} --outdir {outdir} --model {MODEL} -- {fn}')
        assert result.returncode == 0
        assert output_image_path.exists() != bool(simulate)


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_icc_option(simulate, tmp_path):
    """Test --icc-profile option with 48bit images."""
    for fn in (PNG_IMAGE, TIFF_IMAGE):
        outdir = tmp_path
        output_image_path = outdir / (Path(fn).stem + '_al.jpg')
        output_image_path.unlink(missing_ok=True)
        result = run_autolevels(f'{simulate} --outdir {outdir} --outsuffix _al.jpg --icc-profile {ICC_PROFILE} -- {fn}')
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0
        assert output_image_path.exists() != bool(simulate)
        if not simulate:
            with exiftool.ExifToolHelper() as et:
                # first check that there was no ICC profile in the source image
                meta_src = read_metadata_tags(et, fn)
                profile_src = get_metadata_value(meta_src, group="ICC_Profile", tag="ProfileDescription")
                assert profile_src is None, f"Expected no ICC profile in source image {fn}, got '{profile_src}'"

                # check sRGB profile was written to output image
                metadata = read_metadata_tags(et, output_image_path)
                profile = get_metadata_value(metadata, group="ICC_Profile", tag="ProfileDescription")
                assert profile.startswith("sRGB"), f"Expected Profile Description 'sRGB', got '{profile}'"
        output_image_path.unlink(missing_ok=True)


def test_exiftool(tmp_path):
    """Test transferring metadata data between images.

    Verifies that autolevels transfers every metadata group
    (EXIF, EXIF/MakerNote, ICC_Profile, IPTC, XMP) from src → dst
    for JPEG, PNG, and TIFF source files.
    """
    _GROUPS = {
        "IPTC":       ("Keywords",                       ["cat1", "cat2"]),
        "EXIF":       ("YResolution",                    4800),
        "MakerNotes": ("MakerNotes:FocalLength",         17),
        "ICC":        ("ICC_Profile:ProfileDescription", "Adobe RGB (1998)"),
        "XMP":        ("Subject",                        "AutolevelsTestXMP"),
    }
    formats = [
        ("JPEG", ".jpg", JPEG_WITH_ICC_MAKERNOTE),
        ("PNG",  ".png", PNG_WITH_ICC_MAKERNOTE),
        ("TIFF", ".tif", TIFF_WITH_ICC_MAKERNOTE),
    ]
    failures = []

    with exiftool.ExifToolHelper() as et:

        for fmt_label, suffix, fn in formats:

            for outsuffix in ['.jpg', '.tif', '.png']:

                src = tmp_path / f"src{suffix}"
                dst = tmp_path / f"dst{outsuffix}"

                # Copy source file
                shutil.copy(fn, src)

                # Inject metadata into source
                write_metadata_tags(et, src, _GROUPS)

                # Run the function under test
                result = run_autolevels(f'--folder {tmp_path} --outdir {tmp_path} --prefix src --outprefix dst '
                                        f'--outsuffix {outsuffix} -- {suffix}')
                assert result.returncode == 0, result.stderr
                assert dst.exists(), f"[{fmt_label}] failed produce {dst}\n{result.stdout}"

                # Read dst metadata and verify
                dst_meta = read_metadata_tags(et, dst)
                src_meta = read_metadata_tags(et, src)
                print(result.stdout)

                for group, (tag, value) in _GROUPS.items():
                    expected = str(value)  # get_tags, get_metadata return string values
                    actual = get_metadata_value(dst_meta, group, tag)

                    if actual is None:
                        failures.append(
                            f"[{fmt_label}] group={group!r}  tag={tag!r}  "
                            f"→ MISSING from {dst.name}"
                        )
                        continue

                    if group in {"MakerNote"}:
                        # binary blobs may alter encoding on round-trip;
                        # confirming presence is sufficient.
                        continue

                    actual_str = str(actual).strip()
                    if actual_str != expected:
                        failures.append(
                            f"[{fmt_label}] group={group!r}  tag={tag!r}  "
                            f"expected={expected!r}  got={actual_str!r}"
                        )
            dst.unlink(missing_ok=True)

    assert not failures, (
        f"{len(failures)} metadata round-trip failure(s):\n"
        + "\n".join(f"  • {f}" for f in failures)
    )


@pytest.mark.parametrize("simulate", ['--simulate', ''])
def test_darktable_icc(simulate, tmp_path):
    """Test --icc option with darktable export."""
    fn = tmp_path / Path(TEST_IMAGE).name
    from shutil import copyfile
    copyfile(TEST_IMAGE, fn)
    OUTPUT_XMP_PATH = fn.with_suffix(fn.suffix + '.xmp')
    output_image_path = tmp_path / (Path(fn).stem + '_al.jpg')
    result = run_autolevels(f'{simulate} --outdir {tmp_path} --model {MODEL} --icc {ICC_PROFILE} --export darktable -- {fn}')
    assert result.returncode == 0
    print(result.stdout)
    assert output_image_path.exists() != bool(simulate)
    assert OUTPUT_XMP_PATH.exists()

    # Verify content of final XMP
    namespaces = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'darktable': 'http://darktable.sf.net/',
        'xmpMM': 'http://ns.adobe.com/xap/1.0/mm/',
    }
    from autolevels.export import local_name
    import xml.etree.ElementTree as ET

    tree = ET.parse(OUTPUT_XMP_PATH)
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

    assert description.get('{http://darktable.sf.net/}history_basic_hash') == "33e4711b8f6644f5f8c2a164fa3f94cd"
    for li in history_data:
        if li['operation'] == 'colorin':
            assert len(li['params']) > 38  # larger params len due to encoded filename
        elif li['operation'] == 'rgbcurve':
            assert li['num'] == '4'
            assert li['multi_priority'] == '1'
            assert li['multi_name'] == 'AutoLevels'
            assert li['multi_name_hand_edited'] == '1'


def test_darktable_without_export_arg(tmp_path):
    """Test --outsuffix .xmp without --export"""
    fn = tmp_path / Path(TEST_IMAGE).name
    from shutil import copyfile
    copyfile(TEST_IMAGE, fn)
    outsuffix = fn.suffix + '.xmp'
    OUTPUT_XMP_PATH = tmp_path / (fn.stem + outsuffix)
    output_image_path = tmp_path / (Path(fn).stem + '_al.jpg')

    result = run_autolevels(f'--outdir {tmp_path} --model {MODEL} --outsuffix {outsuffix} -- {fn}')
    assert result.returncode == 0
    print(result.stdout)
    assert OUTPUT_XMP_PATH.exists()
    assert not output_image_path.exists()


def test_darktable_versions(tmp_path):
    """Test darktable export for various supported versions of darktable."""
    import xml.etree.ElementTree as ET
    from shutil import copyfile
    fn = tmp_path / Path(TEST_IMAGE).name
    copyfile(TEST_IMAGE, fn)

    for dt_version in ["invalid", "4.8.1", "5.3.0+271~g2a9ae37bcc", "6.0.0"]:
        outsuffix = '_01' + fn.suffix + '.xmp'
        OUTPUT_XMP_PATH = tmp_path / (fn.stem + outsuffix)
        output_image_path = tmp_path / (Path(fn).stem + '_al.jpg')

        cmd = f'--outdir {tmp_path} --model {MODEL} --export darktable {dt_version} --outsuffix {outsuffix} -- {fn}'
        print(cmd)
        result = run_autolevels(cmd)
        assert result.returncode == 0
        print(result.stdout)
        assert 'no darktable version specified' not in result.stdout
        assert not output_image_path.exists(), 'output image produced despite option --outsuffix'
        assert OUTPUT_XMP_PATH.exists() is False if (dt_version == 'invalid') else True

        # Verify content of final XMP
        if not OUTPUT_XMP_PATH.exists(): continue
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'darktable': 'http://darktable.sf.net/',
            'xmpMM': 'http://ns.adobe.com/xap/1.0/mm/',
        }

        tree = ET.parse(OUTPUT_XMP_PATH)
        root = tree.getroot()

        description = root.find('.//rdf:Description', namespaces)
        assert description is not None

        iop_order_list = description.get('{http://darktable.sf.net/}iop_order_list')
        assert iop_order_list is not None
        print(iop_order_list)
        assert ('rasterfile' in iop_order_list) is False if (dt_version == '4.8.1') else True
