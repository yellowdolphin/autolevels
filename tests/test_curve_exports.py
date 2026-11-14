"""Format checks and end-to-end tests without a downloaded production model."""
import importlib
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
from configparser import ConfigParser

import numpy as np
import pytest
from PIL import Image

from autolevels import main
from autolevels.export.common import EXPORT_FORMATS, export_curves, prepare_curves
from autolevels.export.darktable import fit_rgb_curves, compute_monotone_hermite_slopes, hermite_eval
from autolevels.export.rawtherapee import PP3_VERSION


@pytest.fixture
def curves():
    x = np.linspace(0, 1, 256)
    return np.array([0.1 + 0.8 * x, x ** 2, np.sqrt(x)])


@pytest.mark.parametrize('format_name', EXPORT_FORMATS)
def test_invalid_samples_do_not_create_file(tmp_path, format_name, curves):
    exporter = importlib.import_module(f'autolevels.export.{format_name}')
    path = tmp_path / ('bad' + EXPORT_FORMATS[format_name])
    for bad in (np.zeros((256, 3)), np.zeros(255), np.full_like(curves, np.nan), np.full_like(curves, np.inf)):
        with pytest.raises(ValueError):
            exporter.export_curves(path, bad)
        assert not path.exists()


def test_prepare_does_not_mutate():
    values = np.linspace(-0.1, 1.1, 768).reshape(1, 768)
    original = values.copy()
    result = prepare_curves(values)
    assert result.shape == (3, 256)
    assert result.min() == 0 and result.max() == 1
    np.testing.assert_array_equal(values, original)


def test_xmp(tmp_path, curves):
    path = export_curves('lightroom', tmp_path / 'a & b.jpg', curves)
    ns = {'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/',
          'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'}
    root = ET.parse(path)
    desc = root.find('.//rdf:Description', ns)
    assert desc.get('{%s}PresetType' % ns['crs']) == 'Normal'
    assert len(desc.get('{%s}UUID' % ns['crs'])) == 32
    assert 'a & b' in root.find('.//crs:Name/rdf:Alt/rdf:li', ns).text
    assert not any('Exposure' in key or 'WhiteBalance' in key for key in desc.attrib)
    master = root.findall('.//crs:ToneCurvePV2012/rdf:Seq/rdf:li', ns)
    assert [p.text for p in master] == ['0, 0', '255, 255']
    expected_splines = fit_rgb_curves(curves, max_points=16, max_error=0.005)
    for i, channel in enumerate(('Red', 'Green', 'Blue')):
        points = root.findall(f'.//crs:ToneCurvePV2012{channel}/rdf:Seq/rdf:li', ns)
        xy = np.array([[int(n) for n in p.text.split(',')] for p in points])
        np.testing.assert_array_equal(xy, np.rint(expected_splines[i] * 255).astype(int))
        assert 2 <= len(xy) <= 16
        assert xy[0, 0] == 0 and xy[-1, 0] == 255
        assert (np.diff(xy[:, 0]) > 0).all()
        np.testing.assert_array_equal(xy[:, 1], np.rint(curves[i, xy[:, 0]] * 255))


def test_acv(tmp_path, curves):
    path = export_curves('acv', tmp_path / 'scan.jpg', curves)
    data = path.read_bytes()
    assert struct.unpack_from('>HH', data) == (4, 4)
    offset = 4
    expected_splines = fit_rgb_curves(curves, max_points=19, max_error=0.005)
    for channel in range(4):
        count, = struct.unpack_from('>H', data, offset)
        offset += 2
        assert 2 <= count <= 19
        yx = np.array([struct.unpack_from('>HH', data, offset + i * 4) for i in range(count)])
        offset += count * 4
        if channel == 0:
            np.testing.assert_array_equal(yx, [[0, 0], [255, 255]])
        else:
            expected = np.rint(expected_splines[channel - 1] * 255).astype(int)
            np.testing.assert_array_equal(yx[:, ::-1], expected)
            np.testing.assert_array_equal(yx[:, 0], np.rint(curves[channel - 1, yx[:, 1]] * 255))
        assert (np.diff(yx[:, 1]) > 0).all()
    assert offset == len(data)


def read_pp3(path):
    config = ConfigParser()
    config.optionxform = str
    config.read(path)
    return config


def test_pp3(tmp_path, curves):
    path = export_curves('rawtherapee', tmp_path / 'scan.jpg', curves)
    config = read_pp3(path)
    assert config.sections() == ['Version', 'RGB Curves']
    settings = config['RGB Curves']
    assert settings.getboolean('Enabled') and not settings.getboolean('LumaMode')
    expected_splines = fit_rgb_curves(curves, max_error=0.005)
    for i, key in enumerate(('rCurve', 'gCurve', 'bCurve')):
        numbers = [float(v) for v in settings[key].split(';') if v]
        assert numbers[0] == 4
        points = np.array(numbers[1:]).reshape(-1, 2)
        assert 2 <= len(points) <= 20
        assert points[0, 0] == 0 and points[-1, 0] == 1
        assert (np.diff(points[:, 0]) > 0).all()
        np.testing.assert_allclose(points[:, 1], curves[i, np.rint(points[:, 0] * 255).astype(int)], atol=1e-9)
        np.testing.assert_allclose(points, expected_splines[i], atol=1e-9)
        slopes = compute_monotone_hermite_slopes(points[:, 0], points[:, 1])
        reconstructed = hermite_eval(np.linspace(0, 1, 256), points[:, 0], points[:, 1], slopes)
        assert np.max(np.abs(reconstructed - curves[i])) <= 0.005 + 1e-9


def test_pp3_identity_reduces_to_endpoints(tmp_path):
    curves = np.tile(np.linspace(0, 1, 256), (3, 1))
    path = export_curves('rawtherapee', tmp_path / 'identity.jpg', curves)
    config = read_pp3(path)
    for key in ('rCurve', 'gCurve', 'bCurve'):
        values = [float(v) for v in config['RGB Curves'][key].split(';') if v]
        assert values == [4, 0, 0, 1, 1]


EXISTING_PP3 = """\
[Version]
AppVersion=5.9
Version={version}

[Exposure]
Auto=false
Contrast=10

[RGB Curves]
Enabled=false
LumaMode=true
rCurve=0;
gCurve=0;
bCurve=0;

[Sharpening]
Enabled=true
Radius=0.5
"""

@pytest.mark.parametrize(
    'existing_version, expected_version',
    [
        pytest.param(PP3_VERSION - 1, PP3_VERSION, id='older-is-bumped'),
        pytest.param(PP3_VERSION, PP3_VERSION, id='equal-is-kept'),
        pytest.param(PP3_VERSION + 1, PP3_VERSION + 1, id='newer-is-kept'),
    ],
)
def test_pp3_merges_into_existing_file(tmp_path, curves, existing_version, expected_version):
    image = tmp_path / 'scan.jpg'
    # The first export tells us where the sidecar goes (so the test doesn't
    # hard-code the naming scheme) and gives a reference [RGB Curves] section.
    path = export_curves('rawtherapee', image, curves)
    reference = dict(read_pp3(path)['RGB Curves'])

    path.write_text(EXISTING_PP3.format(version=existing_version), encoding='utf-8')
    assert export_curves('rawtherapee', image, curves) == path

    config = read_pp3(path)
    # Order preserved, [RGB Curves] replaced in place (not duplicated or moved).
    assert config.sections() == ['Version', 'Exposure', 'RGB Curves', 'Sharpening']
    assert config['Version']['Version'] == str(expected_version)
    assert config['Version']['AppVersion'] == '5.9'
    assert dict(config['Exposure']) == {'Auto': 'false', 'Contrast': '10'}
    assert dict(config['Sharpening']) == {'Enabled': 'true', 'Radius': '0.5'}
    assert dict(config['RGB Curves']) == reference


def test_cube_roundtrip_with_ffmpeg(tmp_path, curves):
    path = export_curves('cube', tmp_path / 'scan.jpg', curves)
    lines = path.read_text().splitlines()
    assert lines[1] == 'LUT_1D_SIZE 256'
    table = np.loadtxt(path, skiprows=4)
    np.testing.assert_allclose(table.T, curves, atol=1e-9)
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg unavailable for independent LUT reader check')
    # Float planar RGB tests interpolation between samples without codec rounding.
    rng = np.random.default_rng(42)
    rgb = rng.random((3, 8, 8), dtype=np.float32)
    planar_gbr = rgb[[1, 2, 0]].astype('<f4').tobytes()
    proc = subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'gbrpf32le',
        '-video_size', '8x8', '-i', 'pipe:0', '-vf', f'lut1d=file={path}:interp=linear',
        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gbrpf32le', 'pipe:1',
    ], input=planar_gbr, capture_output=True, check=True)
    result = np.frombuffer(proc.stdout, dtype='<f4').reshape(3, 8, 8)[[2, 0, 1]]
    expected = np.stack([np.interp(rgb[i], np.linspace(0, 1, 256), curves[i]) for i in range(3)])
    np.testing.assert_allclose(result, expected, atol=2e-7)


@pytest.mark.parametrize('format_name', EXPORT_FORMATS)
@pytest.mark.parametrize('simulate', [False, True])
def test_pipeline_keeps_image_output(tmp_path, format_name, simulate, curves):
    source = tmp_path / 'source.png'
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(source)
    model_file = tmp_path / 'model.onnx'
    model_file.touch()  # Existing path is required, injected model avoids loading it.
    argv = ['--model', str(model_file), '--model-space', 'none', '--outdir', str(tmp_path),
            '--export', format_name]
    if simulate:
        argv.append('--simulate')
    argv.extend(['--', str(source)])
    assert main(argv=argv, loaded_model=lambda image: curves.copy()) is None
    image_path = tmp_path / 'source_al.png'
    assert image_path.exists() is not simulate
    assert (tmp_path / (image_path.stem + EXPORT_FORMATS[format_name])).exists() is not simulate
    assert not (tmp_path / 'source.png.xmp').exists()
    if not simulate:
        # The source remains untouched; output has the actual model correction.
        np.testing.assert_array_equal(np.asarray(Image.open(source)), 128)
        expected = np.rint(curves[:, 128] * 255)
        np.testing.assert_allclose(np.asarray(Image.open(image_path))[0, 0], expected, atol=1)


def test_export_failure_still_writes_image(tmp_path, monkeypatch, capsys, curves):
    module = importlib.import_module('autolevels.export.cube')
    def fail(*args):
        raise OSError('test write failure')
    monkeypatch.setattr(module, 'export_curves', fail)
    source = tmp_path / 'source.png'
    Image.new('RGB', (16, 16), (128, 128, 128)).save(source)
    model = tmp_path / 'model.onnx'
    model.touch()
    result = main(argv=['--model', str(model), '--model-space', 'none', '--outdir', str(tmp_path),
                        '--export', 'cube', '--', str(source)], loaded_model=lambda image: curves.copy())
    assert result is None
    assert (tmp_path / 'source_al.png').exists()
    assert 'test write failure' in capsys.readouterr().out


def test_grayscale(tmp_path, curves):
    path = export_curves('cube', tmp_path / 'gray.png', curves, grayscale=True)
    table = np.loadtxt(path, skiprows=4)
    np.testing.assert_allclose(table.T, np.repeat(curves.mean(axis=0)[None], 3, axis=0), atol=1e-9)


@pytest.mark.parametrize('format_name', EXPORT_FORMATS)
def test_requires_model(format_name):
    assert main(argv=['--export', format_name, '--', 'missing.png']) == 'Error: --export requires --model'


def test_reject_export_suffix_and_extra_args(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (16, 16)).save(source)
    model = tmp_path / 'model.onnx'
    model.touch()
    for suffix in ('.xmp', '.cube'):
        result = main(argv=['--model', str(model), '--export', 'lightroom', '--outsuffix', suffix,
                            '--outdir', str(tmp_path), '--', str(source)], loaded_model=lambda image: None)
        assert result.startswith('lightroom export expects suffix')
    assert main(argv=['--export', 'cube', 'lightroom', '--', str(source)]).startswith('Error: --export accepts one format')


@pytest.mark.parametrize('format_name', EXPORT_FORMATS)
def test_real_onnx_export(tmp_path, format_name):
    source = tmp_path / 'source.png'
    Image.new('RGB', (16, 16), (90, 130, 170)).save(source)
    result = main(argv=['--model', 'models/free_test.onnx', '--model-space', 'none',
                       '--export', format_name, '--outdir', str(tmp_path), '--', str(source)])
    assert result is None
    assert (tmp_path / 'source_al.png').exists()
    companion = tmp_path / ('source_al' + EXPORT_FORMATS[format_name])
    assert companion.is_file() and companion.stat().st_size > 0


@pytest.mark.parametrize('legacy_name', ['xmp', 'pp3'])
def test_old_export_names_rejected(legacy_name):
    result = main(argv=['--export', legacy_name, '--', 'missing.png'])
    assert result.startswith(f'Error: invalid export {legacy_name},')
