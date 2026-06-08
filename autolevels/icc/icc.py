import struct
from pathlib import Path
from importlib import resources
from PIL import Image
import numpy as np
import lcms2


TRC_TAGS = 'RedTRC', 'GreenTRC', 'BlueTRC', 'GrayTRC'
CM_TAGS = 'RedMatrixColumn', 'GreenMatrixColumn', 'BlueMatrixColumn'
A2B_TAGS = 'AToB0', 'AToB1', 'AToB2', 'AToB3'
B2A_TAGS = 'BToA0', 'BToA1', 'BToA2', 'BToA3'
sRGB_CM_V2 = [[0.43607, 0.22249, 0.01392],
              [0.38515, 0.71687, 0.09708],
              [0.14307, 0.06061, 0.7141]]
D50 = np.array([0.9642, 1.0, 0.8249])  # ICC D50 PCS illuminant (XYZ)


def inspect_icc_profile(data: bytes) -> dict:
    """
    Quickly parse an ICC profile's header and tag table.

    Returns lists of tags grouped in a dict.
    """
    data = data.to_bytes() if isinstance(data, lcms2.Profile) else data
    if len(data) < 132:
        raise ValueError("Data too short to be a valid ICC profile")

    # --- Profile Header Fields ---
    dev_class   = data[12:16].decode("ascii", errors="replace").strip()
    color_space = data[16:20].decode("ascii", errors="replace").strip()
    pcs         = data[20:24].decode("ascii", errors="replace").strip()

    # --- Tag table ---
    (tag_count,) = struct.unpack_from(">I", data, 128)

    tags = set()
    for i in range(tag_count):
        base = 132 + i * 12
        if base + 4 > len(data):
            break
        sig = data[base:base+4].decode("ascii", errors="replace")
        tags.add(sig)

    # --- Check for groups of interest ---
    trc_tags  = {t for t in tags if t in {'rTRC', 'gTRC', 'bTRC', 'kTRC'}}
    cm_tags   = {t for t in tags if t in {'rXYZ', 'gXYZ', 'bXYZ'}}
    atob_tags = {t for t in tags if t.startswith("A2B")}
    btoa_tags = {t for t in tags if t.startswith("B2A")}

    return {
        "class":       dev_class,
        "color_space": color_space,
        "pcs":         pcs,
        "trc_tags":    sorted(trc_tags),
        "cm_tags":     sorted(cm_tags),
        "AToB_tags":   sorted(atob_tags),
        "BToA_tags":   sorted(btoa_tags),
        "all_tags":    sorted(tags),
    }


def is_invertible(profile, pil_img):
    """Check if ICC profile is invertible"""
    tags = inspect_icc_profile(profile)
    is_invertible = True
    if tags['pcs'] != 'XYZ' and not tags['BToA_tags']:
        is_invertible = False
    if not tags['BToA_tags']:
        has_trcs = all(f'{x}TRC' in tags['trc_tags'] for x in 'rgb')
        has_cm = all(f'{x}XYZ' in tags['cm_tags'] for x in 'rgb')
        grayscale = pil_img.mode == 'L' and 'kTRC' in tags.trc_tags
        if not (grayscale or (has_trcs and has_cm)):
            is_invertible = False
    return is_invertible


def get_icc_version(data: bytes, return_dict: bool = False) -> str | dict:
    """
    Return the decoded ICC version.

    Parameters:
        return_dict (bool): return a dict with major, minor, patch, version_str, reserved_ok
    """
    if len(data) < 12:
        raise ValueError("Data too short to contain an ICC version field")

    major       =  data[8]
    minor       = (data[9] & 0xF0) >> 4
    patch       =  data[9] & 0x0F
    reserved_ok =  data[10] == 0 and data[11] == 0

    return {
        "major": major,
        "minor": minor,
        "patch": patch,
        "version_str": f"{major}.{minor}.{patch}",
        "reserved_ok": reserved_ok,  # False would indicate a malformed header
    } if return_dict else f"{major}.{minor}.{patch}"


def get_icc_profile(path):
    """Read ICC_Profile parameters from image or ICC files

    Returns:
        lcms2.Profile
    """
    if Path(path).suffix.lower() in {'.icc', '.icm'}:
        try:
            profile = lcms2.Profile(filename=str(path))
            assert profile is not None
            profile.path = Path(path)
            return profile
        except Exception as e:
            print(f"could not read valid ICC profile from {path} ({e})")
            return None
    elif Path(path).is_file():
        # Extract ICC bytes embedded in image files, lcms can't read those
        with Image.open(path) as img:
            icc_bytes = img.info.get("icc_profile")
        if not icc_bytes:
            return None
        profile = lcms2.Profile(buffer=icc_bytes)
        profile.path = Path(path)
        return profile
    else:
        # Try if this is a builtin profile name
        resource_path = resources.files('autolevels.data') / (
            path.translate(str.maketrans({
                ' ': '_',
                '/': '-',
                '(': None,
                ')': None}))
            + '.icc')

        try:
            profile = lcms2.Profile(builtin=path)
            profile.path = resource_path
            # An ICC file is needed for metadata transfer with ExifTool
            if not profile.path.exists():
                #print(f'wrote {profile.path}')
                #profile.save(profile.path)
                raise ValueError(f'{profile.path} not found')

            # Normalize ProfileDescription
            if profile.name == 'RGB built-in':
                profile.name = path

            return profile
        except Exception as e:
            # Not lcms2-builtin, but autolevels-builtin
            if resource_path.is_file():
                profile = lcms2.Profile(filename=str(resource_path))
                profile.path = resource_path
                return profile
            print(f"lcms2 error: {e}")
            return None


def decode_trc(raw):
    """
    Decode ICC TRC (Transfer Function) data from bytes

    Args:
        raw: bytes containing ICC TRC data

    Returns:
        Decoding function that maps input values to linear RGB,
        Encoding function that maps linear RGB to target space
    """
    if not raw or len(raw) < 4:
        return None, None
    sig = raw[:4].decode()

    if sig == 'curv':
        # Decode curve parameters
        # Number of parameters is encoded because payload may end with 4-byte-alignment padding
        n_params = struct.unpack('>I', raw[8:12])[0]  # parameter count (uInt32Number)

        if n_params == 0:
            #print("Identity (linear) mapping")
            return (lambda x: x), (lambda x: x)

        if n_params == 1:
            gamma = struct.unpack(f'>{n_params}H', raw[12:16])[0] / 256.0
            #print(f"Gamma curve with exponent {gamma}")

            def fn(x):
                with np.errstate(invalid="ignore"):
                    return np.where(x > 0, np.pow(x, gamma), 0)

            def inv_fn(x):
                with np.errstate(invalid="ignore"):
                    return np.where(x > 0, np.pow(x, 1/gamma), 0)
            return fn, inv_fn

        params = np.frombuffer(raw, dtype='>u2', count=n_params, offset=12) / 65535.0
        #print(f"{n_params} curve points for linear interpolation")

        def fn(x):
            xs = np.linspace(0, 1, n_params)
            ys = params
            return np.interp(x, xs, ys)

        def inv_fn(x):
            ys = np.linspace(0, 1, n_params)
            xs = params
            return np.interp(x, xs, ys)

        return fn, inv_fn

    if sig == 'para':
        # Decode parametric curves
        n_params = len(raw) // 4 - 3
        function_type, *params = struct.unpack(f'>H2x{n_params}i', raw[8:])
        #print(f"Parametric curve of type {function_type} with {n_params} parameters: {params}")

        # Type 0, plain gamma
        if function_type == 0:
            if n_params != 1:
                raise ValueError(f"bad ICC profile: type 0 parametric curve must have exactly 1 parameter, got {n_params}")
            g = params[0] / 65536
            if g <= 0:
                raise ValueError(f"bad ICC profile: type 0 parametric curve must have positive parameter, got {g}")
            if g == 1:
                return (lambda x: x, lambda x: x)
            return (lambda x: x ** g, lambda x: x ** (1 / g))

        # Type 1, clipped gamma
        if function_type == 1:
            if n_params != 3:
                raise ValueError(f"bad ICC profile: type 1 parametric curve must have exactly 3 parameters, got {n_params}")
            g, a, b = (p / 65536 for p in params)
            if g <= 0:
                raise ValueError(f"bad ICC profile: type 1 parametric curve must have positive parameter, got {g}")
            if a <= 0:
                raise ValueError(f"bad ICC profile: type 1 parametric curve must have positive a parameter, got a={a}")
            if g == 1:
                return (lambda x: a * x + b, lambda x: (x - b) / a)
            return get_clipped_gamma_trc(g, a, b)

        # Type 2, clipped gamma with manual offset
        if function_type == 2:
            if n_params != 4:
                raise ValueError(f"bad ICC profile: type 2 parametric curve must have exactly 4 parameters, got {n_params}")
            g, a, b, c = (p / 65536 for p in params)
            if g <= 0:
                raise ValueError(f"bad ICC profile: type 2 parametric curve must have positive parameter, got {g}")
            if a <= 0:
                raise ValueError(f"bad ICC profile: type 2 parametric curve must have positive a parameter, got a={a}")
            return get_clipped_gamma_trc(g, a, b, c)

        # Type 3, classic sRGB-like Gamma function with linear segment
        if function_type == 3 and n_params == 5:
            g, a, b, c, d = (p / 65536 for p in params)
            discontinuity = abs(c * d - (a * d + b) ** g)
            if discontinuity > 1e-4:
                print(f"Warning: ICC profile of type 3 has discontinuity of {discontinuity}")
            return get_gamma_trc(g, a, b, c, d)

        # Type 4
        if function_type == 4 and n_params == 7:
            g, a, b, c, d, e, f = (p / 65536 for p in params)
            discontinuity = abs((a * d + b) ** g + e - c * d - f)
            if discontinuity > 1e-4:
                print(f"Warning: ICC profile of type 4 has discontinuity of {discontinuity}")
            if d < -b / a:
                raise ValueError(f"bad ICC profile: type 4 parametric curve must have d >= -b / a, got d={d}, -b/a={-b/a}")
            return get_7p_trc(g, a, b, c, d, e, f)

    print("DEBUG para:", len(sig), type(sig), sig == 'para')
    raise NotImplementedError(f"cannot decode TRC of type {sig}")


def get_clipped_gamma_trc(g, a, b, c=0):
    """
    Create a type-1 or type-2 parametric TRC function
    """
    def fn(x):
        """device -> linear"""
        with np.errstate(invalid="ignore"):
            return np.where(
                x >= -b / a,
                (a * x + b) ** g + c,  # raises Warning for ax + b < 0
                c,
            )

    def inv_fn(x):
        """linear -> device"""
        with np.errstate(invalid="ignore"):
            return np.where(
                x >= c,
                ((x - c) ** (1 / g) - b) / a,  # raises Warning for x < 0
                0,
            )

    return fn, inv_fn


def get_gamma_trc(g=2.4, a=1/1.055, b=0.055/1.055, c=1/12.92, d=0.04045):
    """
    Create a gamma TRC function
    """
    def fn(x):
        """sRGB -> linear"""
        with np.errstate(invalid="ignore"):
            return np.where(
                x >= d,
                (a * x + b) ** g,  # raises Warning for x < 0
                c * x,
            )

    def inv_fn(x):
        """linear -> sRGB"""
        threshold = c * d
        linear_slope = 1 / c
        with np.errstate(invalid="ignore"):
            return np.where(
                x >= threshold,
                (x ** (1 / g) - b) / a,  # raises Warning for y < 0
                linear_slope * x,
            )

    return fn, inv_fn


def get_7p_trc(g, a, b, c, d, e, f):
    def fn(x):
        """device -> linear"""
        return np.where(
            x >= d,
            np.power(np.maximum(a * x + b, 0.0), g) + e,
            c * x + f,
        )

    def inv_fn(x):
        """linear -> device"""
        return np.where(
            x >= c * d + f,
            (np.maximum(x - e, 0.0) ** (1 / g) - b) / a,
            (x - f) / c,
        )

    return fn, inv_fn


def get_line_color_chart():
    # Image dimensions
    width = 256
    height = 7

    # Create empty float32 RGB image
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Shared gradient
    g = np.linspace(0.0, 1.0, width, dtype=np.float32)

    # 0. black -> white
    img[0, :, 0] = g
    img[0, :, 1] = g
    img[0, :, 2] = g

    # 1. black -> red
    img[1, :, 0] = g

    # 2. black -> green
    img[2, :, 1] = g

    # 3. black -> blue
    img[3, :, 2] = g

    # 4. cyan -> white
    img[4, :, 0] = g
    img[4, :, 1] = 1.0
    img[4, :, 2] = 1.0

    # 5. magenta -> white
    img[5, :, 0] = 1.0
    img[5, :, 1] = g
    img[5, :, 2] = 1.0

    # 6. yellow -> white
    img[6, :, 0] = 1.0
    img[6, :, 1] = 1.0
    img[6, :, 2] = g

    return img


def infer_gamma(icc_profile):
    """Infer a gamma value for a profile with linear or gamma-like TRC"""

    GAMMA_22_BUILTIN = {'Adobe RGB (1998)', 'Best RGB', 'Beta RGB', 'Bruce RGB', 'CIE RGB', 'Don RGB 4',
                        'Ekta Space PS5', 'NTSC RGB', 'PAL/SECAM RGB', 'SMPTE-C RGB', 'Wide Gamut RGB'}

    GAMMA_24_BUILTIN = {'ITU-R BT.709 Reference Display', 'ITU-R BT.2020 Reference Display'}

    LINEAR_BUILTIN = {'linear RIMM RGB profile v4', 'Linear Rec709 RGB', 'Linear Rec2020 RGB'}

    if 'srgb' in icc_profile.name.lower():
        return 2.225

    if icc_profile.name in GAMMA_22_BUILTIN:
        return 2.2

    if icc_profile.name in {'Apple RGB', 'ColorMatch RGB', 'ProPhoto RGB'}:
        return 1.8

    if icc_profile.name in LINEAR_BUILTIN:
        return 1.0

    if icc_profile.name in GAMMA_24_BUILTIN:
        return 2.4

    # Infer gamma by converting middle gray from icc_profile to sRGB
    ref_profile = get_icc_profile('Adobe RGB (1998)')
    x = np.array([[[0.5, 0.5, 0.5]]])
    trc = profile_to_profile(x, icc_profile, ref_profile)
    trc = trc.mean()
    gamma = 2.2 * np.log(trc) / np.log(0.5)
    print(f"infered gamma for {icc_profile.name}: {gamma}")

    return gamma


def convert_curve_gamma(curve, gamma):
    """
    Convert curve from sRGB to input ICC profile with gamma
    """
    curve = curve.reshape(1, 3, 256).transpose(2, 0, 1).astype(np.float64)
    grid_points = np.tile(np.linspace(0, 1, 256, dtype=np.float64)[:, None, None], (1, 1, 3))

    print(f"Converting curve back to input space with gamma {gamma:.2f}")
    #print(f"DEBUG: curve stats before: {curve.dtype} {curve.min()} {curve.max()} {curve.mean()}")
    #np.savez("trcs1.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)

    curve_x_rgb = np.power(grid_points, 2.2 / gamma)
    curve_y_rgb = np.power(curve, 2.2 / gamma)

    # Resample to grid points
    curve = np.stack([np.interp(grid_points[:, :, c], curve_x_rgb[:, 0, c], curve_y_rgb[:, 0, c]) for c in range(3)], axis=-1)

    #print(f"DEBUG: curve stats after inv_trc: {curve.min()} {curve.max()} {curve.mean()}")
    #np.savez("trcs2.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)

    return curve.transpose(1, 2, 0).reshape(1, 1, 768).astype(np.float32)


def convert_curve_profile(curve, input_icc_profile, working_profile):
    """
    Convert curve from working profile back to input ICC profile
    """
    curve = curve.reshape(1, 3, 256).transpose(2, 0, 1).astype(np.float64)
    grid_points = np.tile(np.linspace(0, 1, 256, dtype=np.float64)[:, None, None], (1, 1, 3))
    print(f"Converting curve back from working to input profile ({input_icc_profile.name})...")
    #print(f"DEBUG: curve stats before: {curve.dtype} {curve.min()} {curve.max()} {curve.mean()}")
    #np.savez("trcs1.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)

    curve_x_rgb = profile_to_profile(grid_points, working_profile, input_icc_profile)
    #print()
    #print("curve_y transform...")
    curve_y_rgb = profile_to_profile(curve, working_profile, input_icc_profile)
    #print("curve_y after transform:")
    #print(curve_y_rgb[::64, 0, :])
    #print()

    # Resample to grid points
    curve = np.stack([np.interp(grid_points[:, :, c], curve_x_rgb[:, 0, c], curve_y_rgb[:, 0, c]) for c in range(3)], axis=-1)
    #print("interpolated curve_y:")
    #print(curve[::64, 0, :])
    #print()

    #print(f"DEBUG: curve stats after inv_trc: {curve.min()} {curve.max()} {curve.mean()}")
    #np.savez("trcs2.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)

    return curve.transpose(1, 2, 0).reshape(1, 1, 768).astype(np.float32).clip(0, 1)


def profile_to_profile(array, source_profile, target_profile, rendering_intent='perceptual'):
    def prefix(profile):
        return (
            'Lab' if profile.name == 'Lab identity built-in' else
            'XYZ' if profile.name == 'XYZ identity built-in' else
            'RGB')

    dtype_suffix = 'FLT' if array.dtype == np.float32 else 'DBL'
    try:
        transform = lcms2.Transform(source_profile, f"{prefix(source_profile)}_{dtype_suffix}",
                                    target_profile, f"{prefix(target_profile)}_{dtype_suffix}",
                                    intent=rendering_intent.upper(),
                                    #flags="GAMUTCHECK,SOFTPROOFING",
                                    )
    except Exception as e:
        print(f"conversion from {source_profile.name} to {target_profile.name} ({rendering_intent}) failed: {e}")

    try:
        array = transform.apply(np.ascontiguousarray(array))
        assert array is not None
        return array

    except Exception as e:
        print(f'lcms failed to transform from {source_profile.name} to {target_profile.name} '
              f'({rendering_intent}, {dtype_suffix}): {e}')
