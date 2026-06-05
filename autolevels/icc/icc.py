import struct
from importlib import resources
import numpy as np
import exiftool
from autolevels.icc.lut import decode_lut, apply_a2b
import lcms2
from PIL import Image


TRC_TAGS = 'RedTRC', 'GreenTRC', 'BlueTRC', 'GrayTRC'
CM_TAGS = 'RedMatrixColumn', 'GreenMatrixColumn', 'BlueMatrixColumn'
A2B_TAGS = 'AToB0', 'AToB1', 'AToB2', 'AToB3'
B2A_TAGS = 'BToA0', 'BToA1', 'BToA2', 'BToA3'
sRGB_CM_V2 = [[0.43607, 0.22249, 0.01392],
              [0.38515, 0.71687, 0.09708],
              [0.14307, 0.06061, 0.7141]]
D50 = np.array([0.9642, 1.0, 0.8249])  # ICC D50 PCS illuminant (XYZ)


def get_icc_profile(path, exittool_path):
    """Read ICC_Profile parameters from image or ICC files

    Returns:
        ICC_Profile (dict)
    """
    profile = {}
    trcs = {}
    with exiftool.ExifTool(executable=exittool_path) as et:
        description = et.execute('-ICC_Profile:ProfileDescription', str(path)).split(':')[-1].strip()
        if not description:
            print(f"DEBUG: No ICC profile found in {path}")
            return None

        # Get ProfileVersion
        icc_version = et.execute('-ICC_Profile:ProfileVersion', str(path)).split(':')[-1].strip()
        if icc_version.isdigit():
            # decode '512' -> '2.0.0'
            icc_version = f"{(v := int(icc_version)) >> 8}.{(v >> 4) & 0xF}.{v & 0xF}"

        # Get PCS (Profile Connection Space)
        pcs = et.execute('-ICC_Profile:ProfileConnectionSpace', str(path)).split(':')[-1].strip()
        if not pcs:
            print(f"Error: profile {path} has no ProfileConnectionSpace tag")
            return None
        profile['pcs'] = pcs

        if pcs == 'XYZ':  # ICC.1:2022 requires this for CM/TRC profiles
            # Decode TRC (Transfer Function) data
            for tag in TRC_TAGS:
                raw = et.execute(f'-ICC_Profile:{tag}', '-b', str(path), raw_bytes=True)  # bytes directly
                trc_function = decode_trc(raw)  # (TRC, inverse TRC)
                if all(trc_function):
                    trcs[tag] = trc_function
            if trcs:
                profile['trcs'] = trcs

            # Get Color Matrix data
            cm = [et.execute('-n', '-s3', f'-ICC_Profile:{tag}', str(path)).split()[-3:] for tag in CM_TAGS]
            if all(cm):
                profile['cm'] = np.array(cm, dtype=np.float32)

        # Get A2B (AToB) data
        a2b_data = {}
        for tag in A2B_TAGS:
            raw = et.execute(f'-ICC_Profile:{tag}', '-b', str(path), raw_bytes=True)
            decoded = decode_lut(raw)
            if decoded:
                a2b_data[tag] = decoded

        # Get B2A (BToA) data
        b2a_data = {}
        for tag in B2A_TAGS:
            raw = et.execute(f'-ICC_Profile:{tag}', '-b', str(path), raw_bytes=True)
            decoded = decode_lut(raw)
            if decoded:
                b2a_data[tag] = decoded

    profile['path'] = path
    profile['description'] = description
    profile['version'] = icc_version or '2.0.0'
    if a2b_data:
        profile['a2b'] = a2b_data
    if b2a_data:
        profile['b2a'] = b2a_data

    try:
        if path.suffix.lower() in {'.icc', '.icm'}:
            profile['lcms'] = lcms2.Profile(filename=str(path))
        else:
            # Extract ICC bytes embedded in image files, lcms can't read those
            with Image.open(path) as img:
                icc_bytes = img.info.get("icc_profile")
                profile['lcms'] = lcms2.Profile(buffer=icc_bytes)
    except Exception as e:
        print(f"lcms error on {path}:", e)
        raise

    # DEBUG
    if False:
        for key, value in profile.items():
            if key in {'a2b', 'b2a'} and value:
                print(f"    {key} data:")
                for k, v in value.items():
                    print(f"        {k}: {type(v)} {v.shape if hasattr(v, 'shape') else ''}")
                    if isinstance(v, dict):
                        for l, w in v.items():
                            print(f"            {l}: {type(w)} {
                                w.shape if hasattr(w, 'shape') else
                                w if isinstance(w, (str, int)) else
                                len(w) if isinstance(w, list) else ''}")
                continue
            print(f"    {key}: {value}")

    return profile


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


def get_srgb_profile(version, exiftool_path):
    major_version = 2 if version.startswith("2.") else 4
    if major_version == 4:
        return get_icc_profile(resources.files('autolevels.data') / 'sRGB_v4_ICC_preference.icc',
                               exiftool_path)

    gamma_trc = get_gamma_trc()
    profile = {
        'path': resources.files('autolevels.data') / 'sRGB2014.icc',
        'description': 'sRGB',
        'version': '2.0.0',
        'pcs': 'XYZ',
        'trcs': {tag: gamma_trc for tag in TRC_TAGS[0:3]},
        'cm': np.array(sRGB_CM_V2),
        'lcms': lcms2.Profile('sRGB'),
    }
    return profile


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


def convert_to_srgb(array, input_icc_profile, exiftool_path, trcs=None):
    """
    Convert pixel array from input ICC profile to sRGB

    Only apply TRCs, no color matrices!
    """
    print(f"\nConverting model input to sRGB: {array.dtype} {array.min()} {round(array.mean())} {array.max()}")  # uint8
    dtype = array.dtype
    maxvalue = 255 if dtype == np.uint8 else 65535
    array = array.astype(np.float32) / maxvalue
    # Input space to linear RGB
    if 'trcs' in input_icc_profile:
        if 'GrayTRC' in input_icc_profile['trcs']:
            trc = input_icc_profile['trcs']['GrayTRC'][0]
            array = trc(array)

        elif 'RedTRC' in input_icc_profile['trcs']:
            if not ('GreenTRC' in input_icc_profile['trcs'] and 'BlueTRC' in input_icc_profile['trcs']):
                print("TRC tags are incomplete, cannot convert ICC profile")
                return array
            for i, tag in enumerate(TRC_TAGS[0:3]):
                trc = input_icc_profile['trcs'][tag][0]
                array[:, :, i] = trc(array[:, :, i])

            # DEBUG: save TRCs for input->sRGB conversion
            version = input_icc_profile.get('version', '2.0.0')
            srgb_profile = get_srgb_profile(version, exiftool_path)
            xs = np.linspace(0, 1, 256)
            xs_rgb = np.tile(xs[:, None, None], (1, 1, 3))  # (256, 1, 3)
            ys_rgb = profile_to_profile(xs_rgb, input_icc_profile, srgb_profile).reshape(256, 3)
            #np.savez("trcs.npz", xs=xs_rgb, ys=ys_rgb)
            #print("DEBUG: saved input_profile -> sRGB TRCs to trcs.npz")

    else:
        if trcs is None:
            # No TRCs found, construct one from converting a linear curve from input profile to sRGB
            num_gridpoints = 256
            version = input_icc_profile.get('version', '2.0.0')
            srgb_profile = get_srgb_profile(version, exiftool_path)
            xs = np.linspace(0, 1, num_gridpoints)
            xs_rgb = np.tile(xs[:, None, None], (1, 1, 3))  # (256, 1, 3)
            ys_rgb = profile_to_profile(xs_rgb, input_icc_profile, srgb_profile, lut_interpolation='tetrahedral').reshape(num_gridpoints, 3)

            np.savez("trcs.npz", xs=xs_rgb, ys=ys_rgb)
            print("DEBUG: saved input_profile -> sRGB TRCs to trcs.npz")

            def fn(x):
                for c in range(3):
                    x[..., c] = np.interp(x[..., c], xs, ys_rgb[:, c])
                return x

            def inv_fn(x):
                for c in range(3):
                    x[..., c] = np.interp(x[..., c], ys_rgb[:, c], xs)
                return x

            trcs = (fn, inv_fn)

        array = trcs[0](array)

        # Convert back to uint8
        array = (array.clip(0, 1) * maxvalue).astype(dtype)
        return array, trcs  # save end-to-end TRCs for next image

    # Apply encoding sRGB TRC
    trc, inv_trc = get_gamma_trc()  # check
    array = inv_trc(array)

    # Convert back to uint8
    array = (array.clip(0, 1) * maxvalue).astype(dtype)
    print(f"DEBUG: model input in sRGB:     {array.dtype} {array.min()} {round(array.mean())} {array.max()}")

    return array, None


def infer_gamma(icc_profile, exiftool_path):
    if icc_profile['description'].lower().startswith('srgb'):
        return 2.2

    if icc_profile['description'].lower().startswith('adobe'):
        return 2.2

    trcs = icc_profile.get('trcs')
    if trcs is not None:
        trc = trcs.get('GrayTRC') or trcs.get('GreenTRC')
        gamma = np.log(trc[0](0.5)) / np.log(0.5)
        print(f"DEBUG: TRC-fitted gamma: {gamma}")
        return gamma

    if icc_profile.get('a2b'):
        # get pseudo-TRC by converting linear TRC
        version = icc_profile.get('version') or '2.0'
        srgb_profile = get_srgb_profile(version, exiftool_path)
        x = np.array([[[0.5, 0.5, 0.5]]])
        trc = profile_to_profile(x, icc_profile, srgb_profile)
        trc = trc.mean()
        gamma = 2.2 * np.log(trc) / np.log(0.5)
        print(f"DEBUG: a2b-fitted gamma: {gamma}")
        return gamma

    print(f"Warning: gamma could not be derived from {icc_profile['description']} profile ({icc_profile.get('path')})")
    return 2.2


def convert_curve_gamma(curve, gamma):
    """
    Convert curve from sRGB to input ICC profile with gamma
    """
    curve = curve.reshape(1, 3, 256).transpose(2, 0, 1).astype(np.float64)
    grid_points = np.tile(np.linspace(0, 1, 256, dtype=np.float64)[:, None, None], (1, 1, 3))

    print(f"Converting curve back from to input space with gamma {gamma}...")
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
    print(f"Converting curve back from working to input profile ({input_icc_profile['description']})...")
    print(f"DEBUG: curve stats before: {curve.dtype} {curve.min()} {curve.max()} {curve.mean()}")
    np.savez("trcs1.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)

    curve_x_rgb = profile_to_profile(grid_points, working_profile, input_icc_profile)
    print()
    print("curve_y transform...")
    curve_y_rgb = profile_to_profile(curve, working_profile, input_icc_profile)
    print("curve_y after transform:")
    print(curve_y_rgb[::64, 0, :])
    print()

    # Resample to grid points
    curve = np.stack([np.interp(grid_points[:, :, c], curve_x_rgb[:, 0, c], curve_y_rgb[:, 0, c]) for c in range(3)], axis=-1)
    print("interpolated curve_y:")
    print(curve[::64, 0, :])
    print()

    print(f"DEBUG: curve stats after inv_trc: {curve.min()} {curve.max()} {curve.mean()}")
    np.savez("trcs2.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)

    return curve.transpose(1, 2, 0).reshape(1, 1, 768).astype(np.float32).clip(0, 1)


def convert_curve(curve, input_icc_profile, trcs=None):
    """
    Convert curve back from sRGB to input ICC profile

    Parameters:
        curve: curve to convert, shape (1, 1, 768)
        input_icc_profile: input ICC profile
        trcs: end-to-end TRCs from convert_to_srgb

    Only apply TRCs, no color matrices!
    """
    # Reshape curves into pseudo-image, use float64 (np.interp returns float64 anyways)
    curve = curve.reshape(1, 3, 256).transpose(2, 0, 1).astype(np.float64)
    grid_points = np.tile(np.linspace(0, 1, 256, dtype=np.float64)[:, None, None], (1, 1, 3))
    curve_x_rgb = grid_points.copy()
    print("Converting curve back to input space...")
    #print(f"DEBUG: curve stats before: {curve.dtype} {curve.min()} {curve.max()} {curve.mean()}")
    #np.savez("trcs1.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)

    if trcs is not None:
        inv_trc = trcs[1]
        curve_x_rgb = inv_trc(curve_x_rgb)
        curve_y_rgb = inv_trc(curve)

        # Resample to grid points
        curve = np.stack([
            np.interp(grid_points[:, :, c], curve_x_rgb[:, 0, c], curve_y_rgb[:, 0, c]) for c in range(3)
        ], axis=-1)
        #print(f"DEBUG: curve stats after inv_trc: {curve.min()} {curve.max()} {curve.mean()}")
        return curve.transpose(1, 2, 0).reshape(1, 1, 768).astype(np.float32)

    # sRGB -> linear
    trc, inv_trc = get_gamma_trc()  # check
    curve_x_rgb = trc(curve_x_rgb)
    curve_y_rgb = trc(curve)
    #np.savez("trcs.npz", curve_x_rgb=curve_x_rgb, curve_y_rgb=curve_y_rgb)  # OK, bit noisy
    #print(f"DEBUG: linear curve stats after trc: {curve_y_rgb.dtype} {curve_y_rgb.min()} {curve_y_rgb.max()} {curve_y_rgb.mean()}")

    # linear -> input space
    assert 'trcs' in input_icc_profile, "convert_to_srgb did not return trcs?"
    if 'GrayTRC' in input_icc_profile['trcs']:
        inv_trc = input_icc_profile['trcs']['GrayTRC'][1]
        curve_x_rgb = inv_trc(curve_x_rgb)
        curve_y_rgb = inv_trc(curve_y_rgb)

        # Resample to grid points
        curve = np.stack([
            np.interp(grid_points[:, :, c], curve_x_rgb[:, 0, c], curve_y_rgb[:, 0, c]) for c in range(3)
        ], axis=-1)

        #print(f"DEBUG: curve stats after GrayTRC: {curve.dtype} {curve.min()} {curve.max()} {curve.mean()}")
        return curve.transpose(1, 2, 0).reshape(1, 1, 768).astype(np.float32)

    elif 'RedTRC' in input_icc_profile['trcs']:
        if not ('GreenTRC' in input_icc_profile['trcs'] and 'BlueTRC' in input_icc_profile['trcs']):
            print("TRC tags are incomplete, cannot convert ICC profile")
            return curve.transpose(1, 2, 0).reshape(1, 1, 768).astype(np.float32)
        for i, tag in enumerate(TRC_TAGS[0:3]):
            inv_trc = input_icc_profile['trcs'][tag][1]
            curve_x_rgb[:, :, i] = inv_trc(curve_x_rgb[:, :, i])
            curve_y_rgb[:, :, i] = inv_trc(curve_y_rgb[:, :, i])

        # Resample to grid points
        curve = np.stack([
            np.interp(grid_points[:, :, c], curve_x_rgb[:, 0, c], curve_y_rgb[:, 0, c]) for c in range(3)
        ], axis=-1)
        #print(f"DEBUG: curve stats after rgbTRCs: {curve.dtype} {curve.min()} {curve.max()} {curve.mean()}")
        #np.savez("trcs2.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)

    #np.savez("trcs.npz", curve_x_rgb=grid_points, curve_y_rgb=curve)
    return curve.transpose(1, 2, 0).reshape(1, 1, 768).astype(np.float32)


def profile_to_profile(array, input_icc_profile, output_icc_profile, rendering_intent='perceptual',
                       lut_interpolation='linear', lcms=True):
    """
    Convert pixel array from input to output color space

    Parameters:
        array: np.array (float32), pixel array, normalized to 1.0
        input_icc_profile: dict, source ICC profile
        output_icc_profile: dict, target ICC profile
        rendering_intent: str, rendering intent, default: "perceptual"

    Returns:
        array: Converted pixel array
    """
    # disable for debugging
    #if output_icc_profile['description'] == input_icc_profile['description']:
    #    return array

    if lcms:
        assert input_icc_profile.get('lcms') is not None
        assert output_icc_profile.get('lcms') is not None
        source_profile = input_icc_profile['lcms']
        target_profile = output_icc_profile['lcms']
        dtype_suffix = 'FLT' if array.dtype == np.float32 else 'DBL'
        print(f"lcms transform from {source_profile.name} to {target_profile.name}...")
        try:
            transform = lcms2.Transform(source_profile, f"RGB_{dtype_suffix}", target_profile, f"RGB_{dtype_suffix}",
                                        intent=rendering_intent.upper(),
                                        #flags="GAMUTCHECK,SOFTPROOFING",
                                        )
        except Exception as e:
            print(f"lcms cannot transform from {source_profile.name} to {target_profile.name}: {e}")

        print(f"apply to {array.dtype} array...")
        try:
            array = transform.apply(array)
            assert array is not None
            return array

        except Exception as e:
            print(f'lcms failed to transform from {source_profile.name} to {target_profile.name}: {e}\n')
            lcms = False

        if False:
            # DEBUG (needs lcms to succeed)
            lcms_rgb = transform.apply(array)
            xyz_profile = lcms2.Profile('XYZ')
            lab_profile = lcms2.Profile('Lab')
            transform_xyz = lcms2.Transform(source_profile, "RGB_FLT", xyz_profile, "XYZ_FLT",
                                            intent=rendering_intent.upper())
            transform_lab = lcms2.Transform(source_profile, "RGB_FLT", lab_profile, "Lab_FLT",
                                            intent=rendering_intent.upper())
            lcms_xyz = transform_xyz.apply(array)
            lcms_lab = transform_lab.apply(array)
            lcms_lab[..., 0] /= 100
            lcms_lab[..., 1] = (lcms_lab[..., 1] + 128.0) / 255.0
            lcms_lab[..., 2] = (lcms_lab[..., 2] + 128.0) / 255.0

    # Check if conversion is possible with the two profiles, establish ICC.1:2022 precedence.
    # LUT-based ICC profile must support at least 'perceptual' rendering intent.
    # CM/TRC profiles of the input/display class only support 'relative_colorimetric' rendering intent,
    # but this implementation allows to overrule that by user input and by available LUT-based intents.
    if 'a2b' not in input_icc_profile and 'trcs' not in input_icc_profile:
        raise ValueError("source color profile has neither valid TRC nor A2B data, conversion impossible")
    if 'b2a' not in output_icc_profile and 'trcs' not in output_icc_profile:
        raise ValueError("target color profile has neither valid TRC nor B2A data, conversion impossible")
    intent_from_key = {
        'AToB0': 'perceptual',
        'AToB1': 'relative_colorimetric',
        'AToB2': 'saturation',
        'AToB3': 'absolute_colorimetric',
        'BToA0': 'perceptual',
        'BToA1': 'relative_colorimetric',
        'BToA2': 'saturation',
        'BToA3': 'absolute_colorimetric'
    }
    if 'a2b' in input_icc_profile:
        # check available A2B rendering intents
        available_input_intents = set(intent_from_key[key] for key in input_icc_profile['a2b'].keys())
        #print("DEBUG: Available A2B rendering intents:", available_input_intents)
    if 'b2a' in output_icc_profile:
        # check available B2A rendering intents
        available_output_intents = set(intent_from_key[key] for key in output_icc_profile['b2a'].keys())
        #print("DEBUG: Available B2A rendering intents:", available_output_intents)
    if 'a2b' in input_icc_profile and 'b2a' in output_icc_profile:
        compatible_rendering_intents = available_input_intents.intersection(available_output_intents)
        #print("DEBUG: Compatible rendering intents:", compatible_rendering_intents)
        if rendering_intent in compatible_rendering_intents:
            print(f"DEBUG: Using rendering intent: {rendering_intent}")
        else:
            print(f"Warning: rendering intent '{rendering_intent}' not supported by both profiles.")
            if compatible_rendering_intents:
                print(f"Available intents: {compatible_rendering_intents}")
                rendering_intent = 'perceptual' if 'perceptual' in compatible_rendering_intents else compatible_rendering_intents.pop()
                print(f"Using rendering intent: {rendering_intent}")
    elif 'a2b' in input_icc_profile:
        if rendering_intent not in available_input_intents:
            print(f"Warning: rendering intent '{rendering_intent}' not supported by input profile.")
            rendering_intent = (
                # prefer intent compatible with input/display CM/TRC output_icc_profile
                'relative_colorimetric' if 'relative_colorimetric' in available_input_intents else
                'perceptual' if 'perceptual' in available_input_intents else
                available_input_intents.pop())
            print(f"Using rendering intent: {rendering_intent}")
    elif 'b2a' in output_icc_profile:
        if rendering_intent not in available_output_intents:
            print(f"Warning: rendering intent '{rendering_intent}' not supported by output profile.")
            rendering_intent = (
                # prefer intent compatible with input/display CM/TRC input_icc_profile
                'relative_colorimetric' if 'relative_colorimetric' in available_output_intents else
                'perceptual' if 'perceptual' in available_output_intents else
                available_output_intents.pop())
            print(f"Using rendering intent: {rendering_intent}")
    #print(f"DEBUG: Using rendering intent: {rendering_intent}\n")

    # select A2B and B2A data based on rendering intent
    a2b_data = b2a_data = None
    if 'a2b' in input_icc_profile:
        a2b_data = (
            input_icc_profile['a2b']['AToB0'] if rendering_intent == 'perceptual' else
            input_icc_profile['a2b']['AToB1'] if rendering_intent == 'relative_colorimetric' else
            input_icc_profile['a2b']['AToB2'] if rendering_intent == 'saturation' else
            input_icc_profile['a2b']['AToB3'] if rendering_intent == 'absolute_colorimetric' else
            None
        )
    if 'b2a' in output_icc_profile:
        b2a_data = (
            output_icc_profile['b2a']['BToA0'] if rendering_intent == 'perceptual' else
            output_icc_profile['b2a']['BToA1'] if rendering_intent == 'relative_colorimetric' else
            output_icc_profile['b2a']['BToA2'] if rendering_intent == 'saturation' else
            output_icc_profile['b2a']['BToA3'] if rendering_intent == 'absolute_colorimetric' else
            None
        )

    if False:
        # For debugging only: prefer CM/TRC over LUT
        a2b_data = None if (input_icc_profile.get('cm') is not None and input_icc_profile.get('trcs') is not None) else a2b_data
        b2a_data = None if (output_icc_profile.get('cm') is not None and output_icc_profile.get('trcs') is not None) else b2a_data
        input_icc_profile['pcs'] = 'XYZ'

    # Prefer LUT over CM/TRC if both exist (ICC.1:2022, chapter 8.10)
    source_cm = input_icc_profile.get('cm') if a2b_data is None else None
    source_trcs = input_icc_profile.get('trcs') if a2b_data is None else None
    target_cm = output_icc_profile.get('cm') if b2a_data is None else None
    target_trcs = output_icc_profile.get('trcs') if b2a_data is None else None

    print("DEBUG: image before conversion:   ", array.dtype, array.min(), array.mean(), array.max())

    # input -> linear RGB
    if source_trcs is not None:
        if 'GrayTRC' in source_trcs:
            trc = source_trcs['GrayTRC'][0]
            array = trc(array)
        elif 'RedTRC' in source_trcs:
            if not ('GreenTRC' in source_trcs and 'BlueTRC' in source_trcs):
                print("TRC tags are incomplete, cannot leave input color space")
                return array
            array = array.copy()  # don't modify array in caller space
            for i, tag in enumerate(TRC_TAGS[0:3]):
                trc = input_icc_profile['trcs'][tag][0]
                array[:, :, i] = trc(array[:, :, i])
            print("DEBUG: linear RGB after input TRC:", array.dtype, array.min(), array.mean(), array.max())

    # Input color matrix
    if source_cm is not None:
        array = array @ source_cm
        assert input_icc_profile['pcs'] == 'XYZ', f'UNEXPECTED: {input_icc_profile['description']} has TRC/CM and PCS {input_icc_profile['pcs']}'
        print("DEBUG: XYZ after input CM:")
        for c in range(3):
            print("  Channel {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(c, array[:, :, c].min(), array[:, :, c].max(), array[:, :, c].mean()))

        # Ensure LUT and CM/TRC profile use the same XYZ normalization
        if b2a_data is not None:
            # Normalise to the LUT XYZ scale so connect_pcs / apply_a2b see
            # the same [0, 1] encoding that a LUT-based profile would produce.
            if output_icc_profile['pcs'] == 'XYZ':
                array = array / _xyz_norm_scale(b2a_data['type'])
            elif output_icc_profile['pcs'] == 'Lab':
                array = array / _xyz_norm_scale(b2a_data['type'])
                print(f"Divided array by {_xyz_norm_scale(b2a_data['type'])}")
                print(f"array has now a range of {array.min()} to {array.max()}")
            else:
                raise ValueError("Unsupported PCS: {}".format(output_icc_profile['pcs']))
            print("DEBUG: XYZ after LUT-like normalization:")
            for c in range(3):
                print("  Channel {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(c, array[:, :, c].min(), array[:, :, c].max(), array[:, :, c].mean()))
            if lcms:
                print("\nlcms XYZ stats:")
                for c in range(3):
                    print("  Channel {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(c, lcms_xyz[:, :, c].min(), lcms_xyz[:, :, c].max(), lcms_xyz[:, :, c].mean()))


    # A2B
    if a2b_data is not None:
        print("DEBUG: array stats before A2B: min={:.4f}, max={:.4f}, mean={:.4f}".format(array.min(), array.max(), array.mean()))
        array = apply_a2b(array, a2b_data, lut_interpolation)
        if input_icc_profile['pcs'] == 'Lab':
            print("DEBUG: Lab stats after A2B:")
            for c in range(3):
                print(f"  {'Lab'[c]}: min={array[:, :, c].min():.4f}, max={array[:, :, c].max():.4f}, mean={array[:, :, c].mean():.4f}")
            if lcms:
                print("\nlcms Lab stats:")
                for c in range(3):
                    print(f"  {'Lab'[c]}: min={lcms_lab[:, :, c].min():.4f}, max={lcms_lab[:, :, c].max():.4f}, mean={lcms_lab[:, :, c].mean():.4f}")
        else:
            print("DEBUG: XYZ stats after A2B:")
            for c in range(3):
                print("  Channel {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(c, array[:, :, c].min(), array[:, :, c].max(), array[:, :, c].mean()))
            if lcms:
                print("\nlcms XYZ stats:")
                for c in range(3):
                    print("  Channel {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(c, lcms_xyz[:, :, c].min(), lcms_xyz[:, :, c].max(), lcms_xyz[:, :, c].mean()))

    # Compare PCS with profile type
    if not b2a_data and output_icc_profile['pcs'] != 'XYZ':
        print(f"WARNING: {output_icc_profile['description']} has CM/TRC (no B2A), but PCS is 'Lab' - probably this is wrong!")
        if input_icc_profile['pcs'] == 'XYZ':
            skip_pcs_conversion = True
            force_pcs_conversion = False
        else:
            skip_pcs_conversion = False
            force_pcs_conversion = True
    else:
        skip_pcs_conversion = False
        force_pcs_conversion = False

    # Connection space conversion
    if force_pcs_conversion or (input_icc_profile['pcs'] != output_icc_profile['pcs'] and not skip_pcs_conversion):
        print(f"DEBUG: converting from {input_icc_profile['pcs']} to {output_icc_profile['pcs']}")
        H, W, C = array.shape
        array = connect_pcs(
            array.reshape(-1, C),
            src_pcs = input_icc_profile['pcs'],
            dst_pcs = output_icc_profile['pcs'],
            src_lut_type = '' if a2b_data is None else a2b_data['type'],
            dst_lut_type = '' if b2a_data is None else b2a_data['type'],
        ).reshape(H, W, C)
        if output_icc_profile['pcs'] == 'Lab':
            print("DEBUG: Lab stats after PCS conversion:")
            for c in range(3):
                print(f"  {'Lab'[c]}: min={array[:, :, c].min():.4f}, max={array[:, :, c].max():.4f}, mean={array[:, :, c].mean():.4f}")
            if lcms:
                print("\nlcms Lab stats:")
                for c in range(3):
                    print(f"  {'Lab'[c]}: min={lcms_lab[:, :, c].min():.4f}, max={lcms_lab[:, :, c].max():.4f}, mean={lcms_lab[:, :, c].mean():.4f}")
        else:
            print("DEBUG: XYZ stats after PCS conversion:")
            for c in range(3):
                print("  Channel {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(c, array[:, :, c].min(), array[:, :, c].max(), array[:, :, c].mean()))
            if lcms:
                print("\nlcms XYZ stats:")
                for c in range(3):
                    print("  Channel {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(c, lcms_xyz[:, :, c].min(), lcms_xyz[:, :, c].max(), lcms_xyz[:, :, c].mean()))

    # Output color matrix
    if target_cm is not None:
        # Ensure LUT and CM/TRC-profile use the same XYZ normalization
        if a2b_data is not None:
            # Undo LUT normalisation → actual XYZ
            print(f"DEBUG: multiplying with _xyz_norm_scale for type {a2b_data['type']}: {_xyz_norm_scale(a2b_data['type'])}")
            array = array * _xyz_norm_scale(a2b_data['type'])

        try:
            inverse_output_cm = np.linalg.inv(target_cm)
        except np.linalg.LinAlgError as e:
            from PIL import Image, ImageDraw
            print(f'Error: {output_icc_profile['description']} has an invalid color matrix ({e})')
            error_img = Image.new("RGB", array.shape[:2], "white")
            draw = ImageDraw.Draw(error_img)
            draw.text((10, 10), f"Error: ICC profile {output_icc_profile['description']} has an invalid color matrix", fill="black")
            return np.array(error_img)
        array = array @ inverse_output_cm
        print("DEBUG: linear RGB after output CM:", array.dtype, array.min(), array.mean(), array.max())

    # B2A
    if b2a_data is not None:
        array = apply_a2b(array, b2a_data, lut_interpolation)
        print("DEBUG: RGB after output B2A: min={:.4f}, max={:.4f}, mean={:.4f}".format(array.min(), array.max(), array.mean()))
        if lcms:
            print("lcms:  RGB:                  min={:.4f}, max={:.4f}, mean={:.4f}".format(lcms_rgb.min(), lcms_rgb.max(), lcms_rgb.mean()))
        return array

    # linear RGB -> output
    if target_trcs is not None:
        if 'GrayTRC' in target_trcs:
            inv_trc = target_trcs['GrayTRC'][1]
            array = inv_trc(array)
        elif 'RedTRC' in target_trcs:
            if not ('GreenTRC' in target_trcs and 'BlueTRC' in target_trcs):
                print("TRC tags are incomplete, cannot convert ICC profile")
                return array
            array = array.copy()  # don't modify array in caller space
            for i, tag in enumerate(TRC_TAGS[0:3]):
                inv_trc = target_trcs[tag][1]
                array[:, :, i] = inv_trc(array[:, :, i])
        print("DEBUG: RGB after output TRC:", array.dtype, array.min(), array.mean(), array.max())

    return array


# ---- PCS conversion functions ----

# ── normalization scale factors ───────────────────────────────────────────────

def _xyz_norm_scale(lut_type):
    """Actual XYZ = normalized × scale."""
    # negligible, could be skipped: ≈ 1.99997
    return 65535.0 / 32768.0 if lut_type in ('mft1', 'mft2') else 2.0


def _lab_l_norm_scale(lut_type):
    """Actual L* = normalized × scale."""
    # 100.4 for mft2, 100.0 for others
    return 100.0 * 65535.0 / 65280.0 if lut_type == 'mft2' else 100.0
    #return 100 - 1.43  # L-bias gone but aritfacts at gray and CMY gradients


# ── CIE f / f⁻¹ ──────────────────────────────────────────────────────────────

_delta = 6.0 / 29.0
_delta3 = _delta ** 3          # ≈ 0.008856
_delta2_3 = 3.0 * _delta ** 2  # = 108/841


def _f(t):
    return np.where(t > _delta3,
                    np.cbrt(t),
                    t / _delta2_3 + 4.0 / 29.0)


def _f_inv(t):
    return np.where(t > _delta,
                    t ** 3,
                    _delta2_3 * (t - 4.0 / 29.0))


# ── public API ────────────────────────────────────────────────────────────────

def connect_pcs(p, src_pcs, dst_pcs, src_lut_type, dst_lut_type):
    """
    Convert between ICC PCS colour spaces in their normalised [0, 1] form.

    Handles XYZ ↔ Lab using the ICC PCS D50 white point, and also
    renormalises when the same PCS is shared by two profiles whose LUT
    types encode values with different scale factors (e.g. mft2 → mAB).

    Parameters
    ----------
    p            : np.ndarray (N, 3), float64, normalised [0, 1]
    src_pcs      : Source PCS, either 'XYZ' or 'Lab'
    dst_pcs      : Destination PCS, either 'XYZ' or 'Lab'
    src_lut_type : Source LUT type, one of 'mft1' | 'mft2' | 'mAB' | 'mBA'
    dst_lut_type : Destination LUT type, one of 'mft1' | 'mft2' | 'mAB' | 'mBA'

    Returns
    -------
    np.ndarray (N, 3), float64, normalised [0, 1] in dst_pcs encoding
    """
    if src_pcs == 'XYZ' and dst_pcs == 'Lab':
        return _normalized_xyz_to_lab(p, src_lut_type, dst_lut_type)
    elif src_pcs == 'Lab' and dst_pcs == 'XYZ':
        return _normalized_lab_to_xyz(p, src_lut_type, dst_lut_type)
    elif src_pcs == dst_pcs:
        return _renormalize_pcs(p, src_pcs, src_lut_type, dst_lut_type)
    else:
        raise ValueError(f"Unsupported PCS pair: {src_pcs!r} → {dst_pcs!r}")


# ── internal converters ───────────────────────────────────────────────────────

def _normalized_xyz_to_lab(p, src_lut_type, dst_lut_type):
    # 1. Normalised → actual XYZ
    xyz = p * _xyz_norm_scale(src_lut_type)
    #print(f"multiplied XYZ by factor {_xyz_norm_scale(src_lut_type)}")
    #print(f"XYZ range: {xyz.min()} to {xyz.max()}")

    # 2. Adapt to D50, apply f()
    f = _f(xyz / D50)                          # (N, 3)

    # 3. XYZ → Lab
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    #print("Lab channels before normalization/clip:")
    #print(f"  L: {L.min()} to {L.max()}")
    #print(f"  a: {a.min()} to {a.max()}")
    #print(f"  b: {b.min()} to {b.max()}")

    # 4. Lab → normalised
    L_scale = _lab_l_norm_scale(dst_lut_type)
    #print(f"dividing L* by {L_scale}")
    result = np.stack([
        L / L_scale,
        (a + 128.0) / 255.0,
        (b + 128.0) / 255.0,
    ], axis=1)

    return np.clip(result, 0.0, 1.0)


def _normalized_lab_to_xyz(p, src_lut_type, dst_lut_type):
    # 1. Normalised → actual Lab
    L_scale = _lab_l_norm_scale(src_lut_type)
    #print(f"multiplying L* by {L_scale}")
    L = p[:, 0] * L_scale
    a = p[:, 1] * 255.0 - 128.0
    b = p[:, 2] * 255.0 - 128.0

    # 2. Lab → XYZ (apply f⁻¹)
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    xyz = D50 * _f_inv(np.stack([fx, fy, fz], axis=1))    # (N, 3)

    # 3. Actual XYZ → normalised
    result = xyz / _xyz_norm_scale(dst_lut_type)

    return np.clip(result, 0.0, 1.0)


def _renormalize_pcs(p, pcs, src_lut_type, dst_lut_type):
    """Rescale when both profiles share a PCS but use different LUT encodings."""
    if pcs == 'XYZ':
        scale = _xyz_norm_scale(src_lut_type) / _xyz_norm_scale(dst_lut_type)
        return np.clip(p * scale, 0.0, 1.0)
    else:  # Lab
        result = np.empty_like(p)
        l_scale = _lab_l_norm_scale(src_lut_type) / _lab_l_norm_scale(dst_lut_type)
        #print(f"multiplying L* by {l_scale}")
        result[:, 0] = p[:, 0] * l_scale
        result[:, 1] = p[:, 1]     # a* encoding is identical across all LUT types
        result[:, 2] = p[:, 2]     # b* encoding is identical across all LUT types
        return np.clip(result, 0.0, 1.0)
