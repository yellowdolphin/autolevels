import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from importlib import resources
import xml.etree.ElementTree as ET
import struct
import zlib
import base64
import binascii
import numpy as np
import hashlib
import io
from PIL import ImageCms


# iop_order_list is only constructed from scratch if it is missing in XMP file or no XMP file exists.
# Note, that it is actually missing if the user did not change it manually.
# Only non-RAW files are supported.
iop_order_list = {}
# first supported version
iop_order_list['4.8'] = (
    'rawprepare,0,invert,0,temperature,0,'
    'highlights,0,cacorrect,0,hotpixels,0,rawdenoise,0,demosaic,0,rgbcurve,1,colorin,0,'
    'denoiseprofile,0,bilateral,0,rotatepixels,0,scalepixels,0,lens,0,cacorrectrgb,0,'
    'hazeremoval,0,ashift,0,flip,0,enlargecanvas,0,overlay,0,clipping,0,liquify,0,'
    'spots,0,retouch,0,exposure,0,mask_manager,0,tonemap,0,toneequal,0,crop,0,'
    'graduatednd,0,profile_gamma,0,equalizer,0,channelmixerrgb,0,diffuse,0,censorize,0'
    ',negadoctor,0,blurs,0,primaries,0,nlmeans,0,colorchecker,0,defringe,0,atrous,0,'
    'lowpass,0,highpass,0,sharpen,0,colortransfer,0,colormapping,0,channelmixer,0,'
    'basicadj,0,colorbalance,0,colorequal,0,colorbalancergb,0,rgbcurve,0,rgblevels,0,'
    'basecurve,0,filmic,0,sigmoid,0,filmicrgb,0,lut3d,0,colisa,0,tonecurve,0,levels,0,'
    'shadhi,0,zonesystem,0,globaltonemap,0,relight,0,bilat,0,colorcorrection,0,'
    'colorcontrast,0,velvia,0,vibrance,0,colorzones,0,bloom,0,colorize,0,lowlight,0,'
    'monochrome,0,grain,0,soften,0,splittoning,0,vignette,0,colorreconstruct,0,'
    'colorout,0,clahe,0,finalscale,0,overexposed,0,rawoverexposed,0,dither,0,borders,0,'
    'watermark,0,gamma,0')
# moved in 5.0: finalscale
iop_order_list['5.0'] = (
    'rawprepare,0,invert,0,temperature,0,'
    'highlights,0,cacorrect,0,hotpixels,0,rawdenoise,0,demosaic,0,rgbcurve,1,colorin,0,'
    'denoiseprofile,0,bilateral,0,rotatepixels,0,scalepixels,0,lens,0,cacorrectrgb,0,'
    'hazeremoval,0,ashift,0,flip,0,enlargecanvas,0,overlay,0,clipping,0,liquify,0,'
    'spots,0,retouch,0,exposure,0,mask_manager,0,tonemap,0,toneequal,0,crop,0,'
    'graduatednd,0,profile_gamma,0,equalizer,0,channelmixerrgb,0,diffuse,0,censorize,0'
    ',negadoctor,0,blurs,0,primaries,0,nlmeans,0,colorchecker,0,defringe,0,atrous,0,'
    'lowpass,0,highpass,0,sharpen,0,colortransfer,0,colormapping,0,channelmixer,0,'
    'basicadj,0,colorbalance,0,colorequal,0,colorbalancergb,0,rgbcurve,0,rgblevels,0,'
    'basecurve,0,filmic,0,sigmoid,0,filmicrgb,0,lut3d,0,colisa,0,tonecurve,0,levels,0,'
    'shadhi,0,zonesystem,0,globaltonemap,0,relight,0,bilat,0,colorcorrection,0,'
    'colorcontrast,0,velvia,0,vibrance,0,colorzones,0,bloom,0,colorize,0,lowlight,0,'
    'monochrome,0,grain,0,soften,0,splittoning,0,vignette,0,colorreconstruct,0,'
    'finalscale,0,colorout,0,clahe,0,overexposed,0,rawoverexposed,0,dither,0,borders,0,'
    'watermark,0,gamma,0')
# new in 5.2: rasterfile
iop_order_list['5.2'] = (
    'rawprepare,0,invert,0,temperature,0,rasterfile,0,'
    'highlights,0,cacorrect,0,hotpixels,0,rawdenoise,0,demosaic,0,rgbcurve,1,colorin,0,'
    'denoiseprofile,0,bilateral,0,rotatepixels,0,scalepixels,0,lens,0,cacorrectrgb,0,'
    'hazeremoval,0,ashift,0,flip,0,enlargecanvas,0,overlay,0,clipping,0,liquify,0,'
    'spots,0,retouch,0,exposure,0,mask_manager,0,tonemap,0,toneequal,0,crop,0,'
    'graduatednd,0,profile_gamma,0,equalizer,0,channelmixerrgb,0,diffuse,0,censorize,0'
    ',negadoctor,0,blurs,0,primaries,0,nlmeans,0,colorchecker,0,defringe,0,atrous,0,'
    'lowpass,0,highpass,0,sharpen,0,colortransfer,0,colormapping,0,channelmixer,0,'
    'basicadj,0,colorbalance,0,colorequal,0,colorbalancergb,0,rgbcurve,0,rgblevels,0,'
    'basecurve,0,filmic,0,sigmoid,0,filmicrgb,0,lut3d,0,colisa,0,tonecurve,0,levels,0,'
    'shadhi,0,zonesystem,0,globaltonemap,0,relight,0,bilat,0,colorcorrection,0,'
    'colorcontrast,0,velvia,0,vibrance,0,colorzones,0,bloom,0,colorize,0,lowlight,0,'
    'monochrome,0,grain,0,soften,0,splittoning,0,vignette,0,colorreconstruct,0,'
    'finalscale,0,colorout,0,clahe,0,overexposed,0,rawoverexposed,0,dither,0,borders,0,'
    'watermark,0,gamma,0')


def get_iop_order_list_version(export_version):
    """Get the iop_order_list version for the given darktable version (export_version)."""
    latest_version = '5.2'
    assert latest_version in iop_order_list, f'ERROR: invalid latest_version {latest_version}'
    if export_version is None:
        print('Warning: no darktable version specified, using latest iop_order_list')
        # No issues found with alien modules in the list, darktable simply ignores them.
        return latest_version

    major_version, minor_version, *patch_version = export_version.split('.')
    try:
        major_version = int(major_version)
        minor_version = int(minor_version)
    except ValueError:
        raise ValueError(f'Invalid darktable version {export_version}, must be in the format "major.minor.patch"')

    # Update if iop_order_list changes in new darktable versions
    if major_version < 5:
        return '4.8'
    if major_version == 5 and minor_version < 2:
        return '5.0'

    return latest_version


def compute_monotone_hermite_slopes(x, y):
    """Compute monotone-preserving slopes (derivatives) for PCHIP-like Hermite spline."""
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=float)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n, dtype=float)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n-1):
        d0, d1 = delta[i-1], delta[i]
        if d0 == 0 or d1 == 0 or d0 * d1 <= 0:
            m[i] = 0.0
        else:
            m[i] = 0.5 * (d0 + d1)
    for i in range(n):
        if i == 0:
            adj = [abs(delta[0])]
        elif i == n-1:
            adj = [abs(delta[-1])]
        else:
            adj = [abs(delta[i-1]), abs(delta[i])]
        if m[i] != 0.0:
            limit = 3.0 * min(adj)
            if abs(m[i]) > limit and limit > 0:
                m[i] = np.sign(m[i]) * limit
    return m


def hermite_eval(xq, xc, yc, mc):
    """Evaluate cubic Hermite spline at query points xq."""
    xq = np.asarray(xq)

    # Locate interval indices
    inds = np.clip(np.searchsorted(xc, xq) - 1, 0, len(xc) - 2)

    # Extract control values
    x0, x1 = xc[inds], xc[inds + 1]
    y0, y1 = yc[inds], yc[inds + 1]
    m0, m1 = mc[inds], mc[inds + 1]

    # Normalized parameter in interval
    h = x1 - x0
    t = (xq - x0) / h

    # Hermite basis polynomials
    t2, t3 = t**2, t**3
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -h00 + 1   # == -2*t3 + 3*t2
    h11 = t3 - t2

    return h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1


def fit_monotone_spline_to_samples(x, y, max_points=20, max_error=1e-6):
    """Greedy insertion: start with endpoints, add the point with max error until max_points reached."""
    n = len(x)
    if n <= 2 or max_points <= 2:
        return np.array([0, n-1], dtype=int)
    selected = {0, n-1}
    while len(selected) < max_points:
        idxs = np.array(sorted(selected))
        xc, yc = x[idxs], y[idxs]
        mc = compute_monotone_hermite_slopes(xc, yc)
        y_fit = hermite_eval(x, xc, yc, mc)
        candidates = list(set(range(n)) - selected)
        err = np.abs(y[candidates] - y_fit[candidates])
        if np.max(err) < max_error:
            break
        worst_idx = candidates[int(np.argmax(err))]
        selected.add(worst_idx)
    return np.array(sorted(selected), dtype=int)


def fit_rgb_curves(curves, max_points=20, max_error=1e-6):
    """
    curves: ndarray shape (3, 256) float32 in [0,1]
    Returns
        list of 3 arrays (one per channel) with variable shapes (N, 2), N <= 20
        or (return_rmse==True) a list of 3 dicts:
            {'indices': idx array, 'x': x coords [0..1], 'y': y coords [0..1], 'rmse': float}
    """
    curves = np.asarray(curves, dtype=float)
    assert curves.shape == (3, 256), f"expected shape (3,256), got {curves.shape}"
    x = np.linspace(0.0, 1.0, 256)
    results = []
    for ch in range(3):
        y = curves[ch]
        idxs = fit_monotone_spline_to_samples(x, y, max_points=max_points, max_error=max_error)
        xc, yc = x[idxs], y[idxs]
        results.append(np.stack((xc, yc), axis=1))

    return results


def local_name(tag):
    """Remove namespace prefix from tag."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def pack_rgbcurve_params(splines,
                         curve_type: str = "MONOTONE_HERMITE",
                         autoscale_mode: str = "RGB, independent channels",
                         compensate_middle_grey: bool = False,
                         preserve_colors: str = "None",
                         compress_mode: str = "only large entries") -> str:
    """
    Pack curve control points and parameters into the binary layout used by darktable's XMP files.

    Args:
        splines: list of 3 arrays with shapes (N, 2) and N <= 20
        curve_type: One of ("CUBIC_SPLINE", "CENTRIPETAL", "MONOTONE_HERMITE").
        autoscale_mode: One of ("RGB, linked channels", "RGB, independent channels").
        compensate_middle_grey: Whether to enable middle grey compensation.
        preserve_colors: One of RGB_NORMS ('None','Luminance','Max','Average','Sum','Norm','Power').
        compress_mode: exif_xmp_encode mode: 'always', 'only large entries', or 'never'.

    Returns:
        Encoded string suitable for XMP storage (hex or gz..base64 depending on size/mode).
    """
    CURVE_TYPES = ("CUBIC_SPLINE", "CENTRIPETAL", "MONOTONE_HERMITE")
    AUTOSCALE_MODES = ("RGB, linked channels", "RGB, independent channels")
    RGB_NORMS = ('None', 'Luminance', 'Max', 'Average', 'Sum', 'Norm', 'Power')

    if len(splines) != 3:
        raise ValueError("results must be a list of 3 channel dicts (R,G,B)")

    try:
        type_idx = CURVE_TYPES.index(curve_type)
    except ValueError:
        raise ValueError(f"curve_type must be one of {CURVE_TYPES}")
    try:
        autoscale_idx = AUTOSCALE_MODES.index(autoscale_mode)
    except ValueError:
        raise ValueError(f"autoscale_mode must be one of {AUTOSCALE_MODES}")
    try:
        preserve_idx = RGB_NORMS.index(preserve_colors)
    except ValueError:
        raise ValueError(f"preserve_colors must be one of {RGB_NORMS}")

    # Pack float32 array: 3 RGB channels, each with 20 pairs (x,y), padded with zeros
    floats = []
    num_nodes = []
    for curve in splines:
        if curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] > 20:
            raise ValueError(f"numpy array with bad shape {curve.shape}, expected (N, 2)")
        num_nodes.append(len(curve))
        floats.extend(np.pad(curve, ((0, 20 - len(curve)), (0, 0))).flatten())

    # Binary pack: 120 float32 then 9 int32
    data = struct.pack(f"{2 * 20 * 3}f", *floats)

    ints = [
        int(num_nodes[0]),
        int(num_nodes[1]),
        int(num_nodes[2]),
        int(type_idx), int(type_idx), int(type_idx),  # per-channel curve type
        int(autoscale_idx),
        int(bool(compensate_middle_grey)),
        int(preserve_idx),
    ]

    data += struct.pack("9i", *ints)

    # Return XMP-encoded string
    return exif_xmp_encode(data, mode=compress_mode)


def get_embedded_profile(pil_img):
    """Return embedded ICC profile and its name or color space name or None"""
    embedded_icc_profile = None
    icc_bytes = pil_img.info.get("icc_profile")
    if icc_bytes:
        try:
            embedded_icc_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            name = getattr(embedded_icc_profile.profile, 'profile_description', '')
            print(f"{pil_img.filename} has embedded ICC profile: {name}")
            return embedded_icc_profile, name
        except ImageCms.PyCMSError as e:
            print(f"Could not read embedded ICC profile from {pil_img.filename}: {e}")

    # If no color profile could be read, try find color space in EXIF
    exif = pil_img.getexif()
    colorspace_tag = 40961
    colorspace_value = exif.get(colorspace_tag)
    exif_to_dt = {
        1: 'sRGB',
        2: 'Adobe RGB',
        65535: 'Uncalibrated',
    }
    name = exif_to_dt.get(colorspace_value, None)

    if name is None:
        # Try find InteroperabilityIndex tag
        INTEROP_TAG = 0xA005
        try:
            interop_ifd = exif.get_ifd(INTEROP_TAG)
        except KeyError:
            return embedded_icc_profile, name
        interop_values = {s for s in interop_ifd.values() if isinstance(s, str)}
        for s in interop_values:
            if 'R03' in s or 'Adobe RGB' in s:
                name = 'Adobe RGB'
                break
            if 'R98' in s or 'sRGB' in s:
                name = 'sRGB'
                break

    return embedded_icc_profile, name


def update_colorin_params(colorin_params, icc):
    """Update colorin_params with user-provided ICC filename."""
    icc = Path(icc).absolute()
    darktable_color_in = Path.home() / ".config/darktable/color/in/"
    if icc.parent != darktable_color_in:
        registered_icc = darktable_color_in / icc.name
        if darktable_color_in.exists() and not registered_icc.exists():
            from shutil import copy2
            try:
                copy2(icc, registered_icc)
            except Exception:
                print(f"Warning: ICC profile not in darktable's {darktable_color_in} directory: {icc}")
        elif not darktable_color_in.exists():
            print(f"Warning: ICC profile not in darktable's {darktable_color_in} directory: {icc}")
    icc = str(icc)

    # Unpacked colorin_params:
    # enum are represented by 4 bytes (little-endian)
    # profile_type (9: EMBEDDED_ICC, 0: FILE, 1: SRGB, 2: ADOBERGB, 3: LIN_REC709, 4: LIN_REC2020,
    # 5: XYZ, 6: LAB, 7: INFRARED, 8: DISPLAY, 10: EMBEDDED_MATRIX, 11: STANDARD_MATRIX,
    # 12: ENHANCED_MATRIX, 13: VENDOR_MATRIX, 14: ALTERNATE_MATRIX, 15: BRG),
    # filename (512 bytes icc absolute path, padded with zero-bytes),
    # intent (0: perceptive), normalize (0: gamut_clipping off), blue_mapping (0),
    # working_space_profile_type,
    # working_space_filename (512 zero-bytes).
    # Total: 4 + 512 + 4 + 4 + 4 + 4 + 512 = 1044 bytes
    data = exif_xmp_decode(colorin_params)
    if len(data) != 1044:
        print(f'Warning: colorin_params is {len(data)} bytes long, expected 1044, this may be an API change.')

    # params to keep
    intent = data[516:520]
    normalize = data[520:524]
    blue_mapping = data[524:528]
    working_space_profile_type = data[528:532]
    working_space_filename = data[532:]
    if False:
        print(f"colorin_params: profile_type={int.from_bytes(data[:4], byteorder='little')}, "
              f"filename={data[4:516].decode('utf-8').strip()}, intent={int.from_bytes(intent, byteorder='little')}, "
              f"normalize={int.from_bytes(normalize, byteorder='little')}, blue_mapping={int.from_bytes(blue_mapping, byteorder='little')}, "
              f"working_space_profile_type={int.from_bytes(working_space_profile_type, byteorder='little')}, "
              f"working_space_filename={working_space_filename.decode('utf-8').strip()}")

    # params to edit
    profile_type = 0
    profile_type = profile_type.to_bytes(4, byteorder='little')
    filename = icc.encode('utf-8')
    pad_size = 512 - len(filename)
    if pad_size < 0:
        raise ValueError(f"ICC profile path {-pad_size} bytes too long, max len: 512 ASCII chars or 512 bytes of UTF-8")

    # Compose new colorin_params
    parts = [
        profile_type,
        filename,
        bytes(pad_size),
        intent,
        normalize,
        blue_mapping,
        working_space_profile_type,
        working_space_filename]

    return exif_xmp_encode(b''.join(parts))


def check_missing(xmp_file):
    """Return missing XMP elements, anticipated rgbcurve_num, history_end"""
    max_num, history_end = 0, 1
    missing = {'change_timestamp', 'iop_order_list', 'auto_presets', 'history_basic_hash', 'history_current_hash'}
    with open(xmp_file, 'r') as f:
        for line in f:
            if 'darktable:num=' in line:
                num = int(line.strip().split('=')[1].replace('"', ''))
                max_num = max(max_num, num)
                history_end += 1
            elif 'darktable:change_timestamp=' in line:
                missing.discard('change_timestamp')
            elif 'darktable:iop_order_list=' in line:
                missing.discard('iop_order_list')
            elif 'darktable:auto_presets_applied' in line and not line.strip().endswith('"0"'):
                missing.discard('auto_presets')
            elif line.strip() == 'darktable:operation="colorin"':
                missing.discard('auto_presets')
            elif line.strip().startswith('darktable:history_basic_hash='):
                missing.discard('history_basic_hash')
            elif line.strip().startswith('darktable:history_current_hash='):
                missing.discard('history_current_hash')

    if 'auto_presets' in missing:
        # Use hardcoded rgbcurve_num and history_end because XMP will be created by create_basic_xmp()
        return missing, '4', '5'

    return missing, str(max_num + 1), str(history_end)


def create_basic_xmp(xmp_file, pil_img):
    if xmp_file.exists():
        return
    derived_from = Path(pil_img.filename).name
    import_timestamp = unix_to_year1_microseconds()

    # Get exif:DateTimeOriginal as naive or UTC datetime from exif or mtime
    exif = pil_img.getexif()
    date_time_original = exif.get(306, None)
    # print("DEBUG datetime from exif:", date_time_original)
    mtime = Path(pil_img.filename).stat().st_mtime  # float, OS-agnostic
    mtime_str = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y:%m:%d %H:%M:%S.%f")[:-3]
    # print("DEBUG datetime from mtime:", mtime_str)
    date_time_original = date_time_original or mtime_str

    # Read and update template XMP file
    if sys.version_info >= (3, 10):
        template = resources.files('autolevels.data') / 'dt_template_v5.xmp'
        xmp = template.read_text()
    else:
        # Python 3.9 fallback
        with resources.open_text('autolevels.data', 'dt_template_v5.xmp') as f:
            xmp = f.read()
    xmp = xmp.replace('DateTimeOriginal=""', f'DateTimeOriginal="{date_time_original}"')
    xmp = xmp.replace('DerivedFrom="-1"', f'DerivedFrom="{derived_from}"')
    xmp = xmp.replace('import_timestamp="-1"', f'import_timestamp="{import_timestamp}"')

    # Write basic XMP (for Windows, UTF-8 must be specified explicitly)
    with open(xmp_file, 'w', encoding='utf-8') as f:
        f.write(xmp)

    return


def check_darktable_version(export_version):
    """Check if the darktable version is supported."""
    if export_version is None:
        return
    try:
        major_version, minor_version, *patch_version = export_version.split('.')
        major_version = int(major_version)
        minor_version = int(minor_version)
    except ValueError:
        raise ValueError(f'Invalid darktable version {export_version}, must be in the format "major.minor.patch"')
    if major_version > 4:
        return
    if major_version == 4 and minor_version > 7:
        return
    raise ValueError(f'darktable version {export_version} not supported, must be 4.8.1 or greater')


def darktable_change_timestamp_fixed(darktable_version):
    """Return True if apply_sidecar updates change_timestamp (not before darktable 5.4).
    
    See issue #18253: https://github.com/darktable-org/darktable/issues/18253
    """
    try:
        major_version, minor_version, *patch = darktable_version.split('.')
        assert int(major_version) >= 5
        assert int(major_version) > 5 or int(minor_version) > 3
    except (AttributeError, ValueError, AssertionError):
        return False
    return True


def append_rgbcurve_history_item(xmp_file, curves, pil_img, icc=None, new_xmp_file=None, export_version=None):
    """
    Append a new RGB curve history item to the XMP file.

    Args:
        xmp_file: Path to the XMP file.
        curves: ndarray shape (3, 256) float32 in [0,1]
    """
    check_darktable_version(export_version)
    new_xmp_file = xmp_file if new_xmp_file is None else new_xmp_file
    if curves.shape[-1] == 768:
        curves = curves.reshape(3, 256)
    splines = fit_rgb_curves(curves, max_error=0.005)
    params = pack_rgbcurve_params(splines)

    if not xmp_file.exists():
        create_basic_xmp(xmp_file, pil_img)

    # Python's xml.ElementTree does not preserve prefixes, namespace declarations, etc.
    # Hence, let's parse the XMP file manually.

    # Define missing history items when auto_presets_applied is "0" (JPEG image, not opened in darkroom)
    embedded_profile, profile_name = get_embedded_profile(pil_img)
    has_embedded_profile = embedded_profile is not None
    # print("Color Space:", profile_name, "Embedded profile:", has_embedded_profile)
    if has_embedded_profile:
        # Embedded color profile
        colorin_params = 'gz48eJzjZBgFowABWAbaAaNgwAEAMNgADg=='
        history_basic_hash = '02d4cdbda625305c5e181669466f51d2'
    elif profile_name == 'Adobe RGB':
        # Adobe RGB
        colorin_params = 'gz48eJxjYhgFowABWAbaAaNgwAEAFEwABw=='
        history_basic_hash = '0f9b5fb92690a1db2a86cc1fe367bf5d'
    else:
        # sRGB (default and fallback)
        colorin_params = 'gz48eJxjZBgFowABWAbaAaNgwAEAEDgABg=='
        history_basic_hash = '33e4711b8f6644f5f8c2a164fa3f94cd'
    blendop_params = "gz11eJxjYIAACQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dcF/IADRAGpyHQU="

    history_item_colorin = dict(num="0", operation="colorin", enabled="1", modversion="7",
                                params=colorin_params,
                                multi_name="", multi_name_hand_edited="0", multi_priority="0", blendop_version="14",
                                blendop_params=blendop_params)
    history_item_colorout = dict(num="1", operation="colorout", enabled="1", modversion="5",
                                 params="gz35eJxjZBgFo4CBAQAEEAAC",
                                 multi_name="", multi_name_hand_edited="0", multi_priority="0", blendop_version="14",
                                 blendop_params=blendop_params)
    history_item_gamma = dict(num="2", operation="gamma", enabled="1", modversion="1",
                              params="0000000000000000",
                              multi_name="", multi_name_hand_edited="0", multi_priority="0", blendop_version="14",
                              blendop_params=blendop_params)
    history_item_flip = dict(num="3", operation="flip", enabled="1", modversion="2",
                             params="ffffffff",
                             multi_name="_builtin_auto", multi_name_hand_edited="0", multi_priority="0", blendop_version="14",
                             blendop_params=blendop_params)

    # Define history item for custom rgbcurve module
    missing, rgbcurve_num, history_end = check_missing(xmp_file)
    now_stamp = unix_to_year1_microseconds()
    app_name = "AutoLevels" if Path(sys.argv[0]).stem == 'autolevels' else "RetroShine"
    history_item_rgbcurve = {
        'num': rgbcurve_num,
        'operation': "rgbcurve",
        'enabled': "1",
        'modversion': "1",
        'params': params,
        'multi_name': app_name,
        'multi_name_hand_edited': "1",
        'multi_priority': "1",
        'blendop_version': "14",
        'blendop_params': "gz08eJxjYGBgYAFiCQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dlAx68oBEMbFxwX+AwGIBgCbGCeh",
    }

    if not darktable_change_timestamp_fixed(export_version):
        # Don't add change_timestamp until issues are fixed.
        missing.discard('change_timestamp')

    xmp_lines = []
    in_history = False
    in_colorin = False
    with open(xmp_file, 'r') as f:
        for line in f:
            if line.strip().startswith('darktable:import_timestamp=') and 'change_timestamp' in missing:
                # Insert missing change_timestamp
                xmp_lines.append(line)
                xmp_lines.append(f'   darktable:change_timestamp="{now_stamp}"\n')
                continue

            elif line.strip().startswith('darktable:change_timestamp='):
                if darktable_change_timestamp_fixed(export_version):
                    xmp_lines.append(f'   darktable:change_timestamp="{now_stamp}"\n')
                else:
                    xmp_lines.append(line)
                continue

            elif line.strip().startswith('darktable:version_name='):
                xmp_lines.append(f'   darktable:version_name="{app_name}"\n')
                continue

            elif line.strip().startswith('darktable:auto_presets_applied='):
                xmp_lines.append('   darktable:auto_presets_applied="1"\n')
                continue

            elif line.strip().startswith('darktable:history_end='):
                xmp_lines.append(f'   darktable:history_end="{history_end}"\n')
                continue

            elif line.strip().startswith('darktable:iop_order_version='):
                xmp_lines.append('   darktable:iop_order_version="0"\n')
                # Add iop_order_list with new "rgbcurve,1" instance before colorin
                if 'iop_order_list' in missing:
                    iop_order_list_version = get_iop_order_list_version(export_version)
                    xmp_lines.append(f'   darktable:iop_order_list="{iop_order_list[iop_order_list_version]}"\n')

                # Add missing tags after iop_order_list
                if 'history_basic_hash' in missing:
                    xmp_lines.append(f'   darktable:history_basic_hash="{history_basic_hash}"\n')
                    missing.discard('history_basic_hash')
                if 'history_current_hash' in missing:
                    xmp_lines.append('   darktable:history_current_hash="replace_with_current_history_hash">\n')
                    missing.discard('history_current_hash')
                continue

            elif line.strip().startswith('darktable:iop_order_list='):
                module_list = line.split('="')[1].split(',')
                colorin_index = module_list.index('colorin')

                if module_list[colorin_index - 2] == 'rgbcurve':
                    # found rgbcurve before colorin, just update multi_priority
                    history_item_rgbcurve['multi_priority'] = module_list[colorin_index - 1]
                    xmp_lines.append(line)
                    continue

                # Pick next unused instance number for new "rgbcurve" instance
                rgbcurve_instances = [int(module_list[i+1]) for i, s in enumerate(module_list) if s == 'rgbcurve']
                rgbcurve_instance = max(rgbcurve_instances) + 1

                # Insert new "rgbcurve" instance before colorin and update multi_priority
                xmp_lines.append(line.replace('colorin,0', f'rgbcurve,{rgbcurve_instance},colorin,0'))
                history_item_rgbcurve['multi_priority'] = rgbcurve_instance
                continue

            elif line.strip().startswith('darktable:history_current_hash='):
                end_tag = ">" if line.strip().endswith(">") else ""
                xmp_lines.append(f'   darktable:history_current_hash="replace_with_current_history_hash"{end_tag}\n')
                continue

            elif line.strip() == "<darktable:history>":
                in_history = True
                xmp_lines.append(line)
                continue

            elif in_history and line.strip() == '<rdf:Seq/>':
                # Replace empty-Seq tag with start-Seq tag
                xmp_lines.append("    <rdf:Seq>\n")
                continue

            elif in_history and line.strip() == '</rdf:Seq>':
                # Delete end-Seq tag (append after new history item)
                continue

            elif in_history and line.strip() == 'darktable:operation="colorin"':
                in_colorin = True
                xmp_lines.append(line)
                continue

            elif icc and in_colorin and line.strip().startswith('darktable:params='):
                assert 'auto_presets' not in missing, 'found colorin despite missing auto_presets'
                colorin_params = line.strip().split('params=')[1].replace('"', '')
                colorin_params = update_colorin_params(colorin_params, icc)
                xmp_lines.append(f'      darktable:params="{colorin_params}"\n')
                in_colorin = False
                continue

            elif line.strip() == "</darktable:history>":
                if 'auto_presets' in missing:
                    # Append new history items (auto presets)
                    xmp_lines.append("     <rdf:li\n")
                    for key, value in history_item_colorin.items():
                        if key == 'params' and icc:
                            colorin_params = update_colorin_params(value, icc)
                            xmp_lines.append(f'      darktable:params="{colorin_params}"\n')
                            continue
                        end_tag = "/>" if key == 'blendop_params' else ""
                        xmp_lines.append(f'      darktable:{key}="{value}"{end_tag}\n')
                    xmp_lines.append("     <rdf:li\n")
                    for key, value in history_item_colorout.items():
                        end_tag = "/>" if key == 'blendop_params' else ""
                        xmp_lines.append(f'      darktable:{key}="{value}"{end_tag}\n')
                    xmp_lines.append("     <rdf:li\n")
                    for key, value in history_item_gamma.items():
                        end_tag = "/>" if key == 'blendop_params' else ""
                        xmp_lines.append(f'      darktable:{key}="{value}"{end_tag}\n')
                    xmp_lines.append("     <rdf:li\n")
                    for key, value in history_item_flip.items():
                        end_tag = "/>" if key == 'blendop_params' else ""
                        xmp_lines.append(f'      darktable:{key}="{value}"{end_tag}\n')

                # Append new history item "rgbcurve"
                xmp_lines.append("     <rdf:li\n")
                for key, value in history_item_rgbcurve.items():
                    end_tag = "/>" if key == 'blendop_params' else ""
                    xmp_lines.append(f'      darktable:{key}="{value}"{end_tag}\n')
                xmp_lines.append("    </rdf:Seq>\n")  # end of Seq tag
                xmp_lines.append(line)
                in_history = False
                continue

            xmp_lines.append(line)

    with open(new_xmp_file, 'w') as f:
        f.writelines(xmp_lines)

    # Update history_current_hash
    new_hash, _ = calculate_history_current_hash(new_xmp_file)
    new_hash = new_hash or "-1"
    xmp_lines = [s.replace('replace_with_current_history_hash', new_hash) for s in xmp_lines]
    with open(new_xmp_file, 'w') as f:
        f.writelines(xmp_lines)


def unix_to_year1_microseconds(unix_timestamp=None):
    """
    Convert Unix timestamp to microseconds since Year 1 CE.

    Args:
        unix_timestamp (float, optional): Unix timestamp in seconds.
                                        If None, uses current time.
        offset_microseconds (int): Additional offset to add/subtract in microseconds

    Returns:
        int: Microseconds since January 1, Year 1 CE (plus offset)
    """
    if unix_timestamp is None:
        unix_timestamp = time.time()
        now = True # DEBUG
    else:
        now = False # DEBUG

    # Calculate microseconds from Year 1 CE to Unix epoch (1970-01-01)
    dt_unix_epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    dt_year1_epoch = datetime(1, 1, 1, tzinfo=timezone.utc)

    # Calculate timedelta from Year 1 CE to Unix epoch
    delta_year1_to_unix_epoch = dt_unix_epoch - dt_year1_epoch

    # Convert to microseconds
    year1_to_unix_microseconds = int(delta_year1_to_unix_epoch.total_seconds() * 1_000_000)

    # Add Unix microseconds to get total microseconds since Year 1 CE
    year1_microseconds = year1_to_unix_microseconds + int(unix_timestamp * 1_000_000)

    # DEBUG (remove when dt issue is fixed and tested)
    if False and now:
        dt = datetime.fromtimestamp((year1_microseconds - 62135596800000000) / 1e+6)
        print(f'DEBUG current timestamp at {dt.strftime("%Y-%m-%d %H:%M:%S")} is {year1_microseconds}')
        # database has change_timestamp 63892452067899160, 2025-09-03 01:21:07 from image.change_timestamp
        # autolevels calculated         63892452067817010  for the same second (82150 µs earlier) -> XMP
        # But this is not the difference, "updated sidecar file found" complaints about!
        # 63892486336110989 - 63892486336041116 = 69873 (70 ms) around 10:52:16
        # but "database timestamp" is reported 10:52:04
        # when I keep the "database edit", change_timestamp is still the same as XMP file
        # database has also "write_timestamp", which is 1756889524 before and 1756889978 after keeping the "database edit".
        # discard history: deletes change_timestamp in XMP and database and updates write_timestamp -> 1756890775 (11:12:55)
        # read_xmp at 11:27:54: updates change_timestamp 70 ms after XMP -> 63892488474690836 (but not write_timestamp)
        # "updated XMP sidecar files found" compares write_timestamp aka "database timestamp" (11:12:55) with change_timestamp or actual mtime?
        # "keep XMP edit" updates write_timestamp in the database with the correct file modification time and change_timestamp to "now"

    return year1_microseconds


def exif_xmp_decode(encoded_data: str) -> bytes:
    """
    Decode darktable's XMP-encoded binary data.

    Handles both compressed (gzipped + base64) and hex-encoded data as used in darktable's XMP files.

    Args:
        encoded_data: The encoded string from XMP (starts with 'gz' for compressed data)

    Returns:
        Decoded binary data

    Raises:
        ValueError: If the input data is malformed or decoding fails
    """
    if not encoded_data:
        return b''

    try:
        # Handle compressed data (starts with 'gz' followed by compression factor)
        if encoded_data.startswith('gz'):
            if len(encoded_data) < 4:
                raise ValueError("Compressed data too short")

            # compression_factor = int(encoded_data[2:4])  # Not actually used for decompression

            # Decode base64 and decompress
            compressed_data = base64.b64decode(encoded_data[4:])

            # Use zlib with appropriate window bits for gzip format
            return zlib.decompress(compressed_data, 15 + 32)

        # Handle hex-encoded data
        else:
            # Remove any whitespace and convert to bytes
            hex_str = ''.join(encoded_data.split()).lower()
            if not all(c in '0123456789abcdef' for c in hex_str):
                raise ValueError("Invalid hex string")

            return bytes.fromhex(hex_str)

    except (binascii.Error, zlib.error, base64.binascii.Error) as e:
        raise ValueError(f"Failed to decode XMP data: {str(e)}") from e


def exif_xmp_encode(data: bytes, mode: str = "only large entries", threshold: int = 100) -> str:
    """
    Encode binary data into darktable's XMP-friendly format.

    Mirrors dt_exif_xmp_encode from darktable:
    - If compression is enabled (mode == 'always' or 'only large entries' with size > threshold),
      compress with zlib, base64-encode, and prefix with 'gz' plus a 2-digit compression factor.
    - Otherwise, return a lowercase hex string.

    Args:
        data: Binary data to encode.
        mode: 'always', 'only large entries', or 'never'. Defaults to 'only large entries'.
        threshold: Size threshold for compression when mode is 'only large entries'. Default 100 bytes.

    Returns:
        Encoded string suitable for XMP storage.

    Raises:
        TypeError: If input is not bytes-like.
        ValueError: If mode is invalid.
    """
    if data is None:
        return ""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("exif_xmp_encode expects bytes-like input")

    do_compress = False
    if mode == "always":
        do_compress = True
    elif mode == "only large entries":
        do_compress = len(data) > threshold
    elif mode == "never":
        do_compress = False
    else:
        raise ValueError("mode must be one of: 'always', 'only large entries', 'never'")

    if do_compress:
        compressed = zlib.compress(data)
        # Compute compression factor like C code: MIN(len / destLen + 1, 99)
        src_len = len(data)
        dest_len = len(compressed) if len(compressed) else 1
        factor = min(src_len // dest_len + 1, 99)
        b64 = base64.b64encode(compressed).decode("ascii")
        return f"gz{factor:02d}{b64}"
    else:
        # Lowercase hex encoding
        return bytes(data).hex()


def calculate_history_current_hash(xmp_file_path):
    """
    Reproduce darktable's history_current_hash.

    Args:
        xmp_file_path: Path to the XMP file

    Returns:
        history_current_hash, original hash
    """
    try:
        # Parse the XMP file
        tree = ET.parse(xmp_file_path)
        root = tree.getroot()

        # Define namespaces (XMP uses namespaces)
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xmp': 'http://ns.adobe.com/xap/1.0/',
            'darktable': 'http://darktable.sf.net/',
            'lr': 'http://ns.adobe.com/lightroom/1.0/',
            'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/',
            'xmpMM': 'http://ns.adobe.com/xap/1.0/mm/',
            'stEvt': 'http://ns.adobe.com/xap/1.0/sType/ResourceEvent#',
            'dc': 'http://purl.org/dc/elements/1.1/'
        }

        # Extract history data
        history_data = []
        history_seq = root.find('.//darktable:history/rdf:Seq', namespaces)
        if history_seq is not None:
            for li in history_seq.findall('rdf:li', namespaces):
                entry_data = {}

                for key, value in li.items():
                    local_key = local_name(key)
                    entry_data[local_key] = value

                history_data.append(entry_data)

        # Extract darktable specific settings from the main description
        darktable_settings = {}
        description = root.find('.//rdf:Description', namespaces)
        if description is not None:
            for key, value in description.attrib.items():
                local_key = local_name(key)
                darktable_settings[local_key] = value

        # Test binary blobs used in darktable's database
        if False:
            for i, op in enumerate(history_data):
                params_blob = exif_xmp_decode(op["params"])
                blendop_params_blob = exif_xmp_decode(op["blendop_params"])
                params = exif_xmp_encode(params_blob)
                blendop_params = exif_xmp_encode(blendop_params_blob)
                if op['params'] != params:
                    print(f"{i} params not reversibly decodable")
                if op['blendop_params'] != blendop_params:
                    print(f"{i} blendop_params not reversibly decodable")

        # Build active history similar to _history_hash_compute_from_db:
        # - consider items with num <= history_end
        # - group by (operation, multi_priority) keeping the latest (max num)
        # - iterate in ascending num and only include enabled items
        # Use history_end from XMP description if present; otherwise allow all entries
        try:
            history_end = int(darktable_settings.get('history_end')) if 'history_end' in darktable_settings else (1 << 30)
        except Exception:
            history_end = 1 << 30

        latest_by_key = {}
        for entry in history_data:
            try:
                num = int(entry.get('num', '0'))
            except Exception:
                num = 0
            if num > history_end:
                continue
            op_name = entry.get('operation', '')
            multi_priority = int(entry.get('multi_priority', '0'))
            key = (op_name, multi_priority)
            # Keep only the highest num per key
            prev = latest_by_key.get(key)
            if prev is None or int(prev.get('num', '0')) < num:
                latest_by_key[key] = entry

        # Sort selected entries by num (ORDER BY num)
        selected = sorted(latest_by_key.values(), key=lambda e: int(e.get('num', '0')))

        # Compute current hash: concat operation, params blob, blendop_params blob
        md5 = hashlib.md5()
        history_on = False
        for entry in selected:
            enabled = entry.get('enabled', '0') in ('1', 'true', 'True')
            if not enabled:
                continue
            op_name = entry.get('operation', '')
            if op_name:
                md5.update(op_name.encode('utf-8'))
            # params
            params_blob = exif_xmp_decode(entry.get('params', ''))
            # round-trip check (optional verbose)
            try:
                if entry.get('params', '') and exif_xmp_encode(params_blob) != entry.get('params', ''):
                    print("   params not reversibly decodable")
            except Exception:
                pass
            if params_blob:
                md5.update(params_blob)
            # blendop_params
            blendop_blob = exif_xmp_decode(entry.get('blendop_params', ''))
            try:
                if entry.get('blendop_params', '') and exif_xmp_encode(blendop_blob) != entry.get('blendop_params', ''):
                    print("   blendop_params not reversibly decodable")
            except Exception:
                pass
            if blendop_blob:
                md5.update(blendop_blob)
            history_on = True

        # Append module order if there was at least one enabled entry
        current_hash = None
        if history_on:
            try:
                iop_version = (
                    int(darktable_settings.get('iop_order_version'))
                    if 'iop_order_version' in darktable_settings
                    else None)
            except Exception:
                iop_version = None
            if iop_version is None:
                # Fallback to legacy if not present
                iop_version = 1
            # C code updates with bytes of native int; use little-endian 32-bit for stability
            md5.update(struct.pack('<i', iop_version))
            if iop_version == 0:  # DT_IOP_ORDER_CUSTOM
                iop_list_str = darktable_settings.get('iop_order_list')
                if iop_list_str:
                    md5.update(iop_list_str.encode('utf-8'))
            current_hash = md5.digest().hex()

        return current_hash, darktable_settings['history_current_hash']

    except Exception as e:
        print(f"Error processing {xmp_file_path}: {e}")
        return None, None
