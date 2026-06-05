import struct
import numpy as np


def decode_lut(raw):
    """
    Decode ICC A2B and B2A data from bytes.

    Supports:
      - 'mft1' (LUT8Type)      ICC spec §10.8
      - 'mft2' (LUT16Type)     ICC spec §10.9
      - 'mAB ' (lutAToBType)   ICC spec §10.10
      - 'mBA ' (lutBToAType)   ICC spec §10.11

    Every table/curve is returned in normalised float form (domain [0, 1])
    so that callers can apply them without knowing the original bit-depth.

    Returns
    -------
    dict with keys (present only when the tag actually contains the element):

      type                : str  [ 'mft1' | 'mft2' | 'mAB' | 'mBA' ]
      num_input_channels  : int
      num_output_channels : int

      -- LUT8 / LUT16 only --
      matrix          : np.ndarray shape (3, 3) or None
      clut_grid_size  : int
      input_tables    : list[np.ndarray]    one 1-D array per input channel
      clut            : np.ndarray          shape (grid^in, out_channels)
      output_tables   : list[np.ndarray]    one 1-D array per output channel

      -- mAB / mBA only --
      b_curves        : list[np.ndarray] | None
      matrix          : np.ndarray shape (3, 3) | None
      m_curves        : list[np.ndarray] | None
      clut            : np.ndarray | None
      a_curves        : list[np.ndarray] | None
    """
    if not raw or len(raw) < 4:
        return None

    sig = raw[:4].decode("latin-1")

    # ------------------------------------------------------------------ #
    #  helpers                                                           #
    # ------------------------------------------------------------------ #

    def s15f16(data, offset):
        """Read one s15Fixed16Number → float."""
        val = struct.unpack_from(">i", data, offset)[0]          # signed 32-bit
        return val / 65536.0

    def decode_curve_tag(data, offset):
        """
        Parse a curveType or parametricCurveType embedded inside mAB/mBA.

        Returns a 1-D np.ndarray in [0,1] representing the curve as a LUT,
        OR a callable (for parametric curves) that is already pre-sampled
        to 256 points so the caller always gets an ndarray.
        """
        tag_sig = data[offset:offset + 4].decode("latin-1")

        if tag_sig == "curv":
            count = struct.unpack_from(">I", data, offset + 8)[0]
            if count == 0:
                # identity
                return np.linspace(0.0, 1.0, 256, dtype=np.float64)
            elif count == 1:
                gamma_raw = struct.unpack_from(">H", data, offset + 12)[0]
                gamma = gamma_raw / 256.0
                x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
                return np.power(np.maximum(x, 0.0), gamma)
            else:
                raw_vals = struct.unpack_from(f">{count}H", data, offset + 12)
                return np.array(raw_vals, dtype=np.float64) / 65535.0

        elif tag_sig == "para":
            fn_type = struct.unpack_from(">H", data, offset + 8)[0]
            params_raw = data[offset + 12:]

            def read_params(n):
                return [s15f16(params_raw, i * 4) for i in range(n)]

            x = np.linspace(0.0, 1.0, 256, dtype=np.float64)

            if fn_type == 0:          # Y = X^g
                g, = read_params(1)
                y = np.power(np.maximum(x, 0.0), g)
            elif fn_type == 1:        # CIE 122-1966
                g, a, b = read_params(3)
                y = np.where(x >= -b / a,
                             np.power(np.maximum(a * x + b, 0.0), g),
                             0.0)
            elif fn_type == 2:        # IEC 61966-3
                g, a, b, c = read_params(4)
                y = np.where(x >= -b / a,
                             np.power(np.maximum(a * x + b, 0.0), g) + c,
                             c)
            elif fn_type == 3:        # IEC 61966-2.1 (sRGB)
                g, a, b, c, d = read_params(5)
                y = np.where(x >= d,
                             np.power(np.maximum(a * x + b, 0.0), g),
                             c * x)
            elif fn_type == 4:
                g, a, b, c, d, e, f = read_params(7)
                y = np.where(x >= d,
                             np.power(np.maximum(a * x + b, 0.0), g) + e,
                             c * x + f)
            else:
                raise NotImplementedError(f"parametricCurveType function {fn_type}")

            #return np.clip(y, 0.0, 1.0)
            return y  # lcms does not clip here

        else:
            raise ValueError(f"Unexpected curve tag '{tag_sig}' at offset {offset}")

    def curve_byte_size(data, offset):
        """Return the padded byte size of a curveType/parametricCurveType."""
        tag_sig = data[offset:offset + 4].decode("latin-1")
        if tag_sig == "curv":
            count = struct.unpack_from(">I", data, offset + 8)[0]
            size = 12 + count * 2
        elif tag_sig == "para":
            fn_type = struct.unpack_from(">H", data, offset + 8)[0]
            n_params = [1, 3, 4, 5, 7][fn_type]
            size = 12 + n_params * 4
        else:
            raise ValueError(f"Unknown curve tag '{tag_sig}'")
        # 4-byte alignment padding
        return size + (4 - size % 4) % 4

    # ------------------------------------------------------------------ #
    #  LUT8 – 'mft1'                                                     #
    # ------------------------------------------------------------------ #
    if sig == "mft1":
        components = {"type": "mft1"}

        # Header: sig(4) + reserved(4) + in(1) + out(1) + grid(1) + pad(1) + matrix(9×s15f16)
        num_in = raw[8]
        num_out = raw[9]
        grid = raw[10]

        components["num_input_channels"] = num_in
        components["num_output_channels"] = num_out
        components["clut_grid_size"] = grid

        # 3×3 matrix (only meaningful when num_in == 3)
        matrix = None
        if num_in == 3:
            vals = [s15f16(raw, 12 + i * 4) for i in range(9)]
            matrix = np.array(vals, dtype=np.float64).reshape(3, 3)
        components["matrix"] = matrix

        offset = 12 + 9 * 4

        # Input tables: num_in channels × 256 entries × 1 byte
        input_tables = []
        for _ in range(num_in):
            table = np.frombuffer(raw, dtype=np.uint8, count=256, offset=offset).astype(np.float64) / 255.0
            input_tables.append(table)
            offset += 256
        components["input_tables"] = input_tables

        # CLUT: grid^num_in × num_out entries × 1 byte
        clut_entries = (grid ** num_in) * num_out
        clut_raw = np.frombuffer(raw, dtype=np.uint8, count=clut_entries, offset=offset).astype(np.float64) / 255.0
        components["clut"] = clut_raw.reshape(-1, num_out)
        offset += clut_entries

        # Output tables: num_out channels × 256 entries × 1 byte
        output_tables = []
        for _ in range(num_out):
            table = np.frombuffer(raw, dtype=np.uint8, count=256, offset=offset).astype(np.float64) / 255.0
            output_tables.append(table)
            offset += 256
        components["output_tables"] = output_tables

        return components

    # ------------------------------------------------------------------ #
    #  LUT16 – 'mft2'                                                    #
    # ------------------------------------------------------------------ #
    elif sig == "mft2":
        components = {"type": "mft2"}

        num_in = raw[8]
        num_out = raw[9]
        grid = raw[10]

        components["num_input_channels"] = num_in
        components["num_output_channels"] = num_out
        components["clut_grid_size"] = grid

        matrix = None
        if num_in == 3:
            vals = [s15f16(raw, 12 + i * 4) for i in range(9)]
            matrix = np.array(vals, dtype=np.float64).reshape(3, 3)
        components["matrix"] = matrix

        offset = 12 + 9 * 4

        # Number of input / output table entries (uint16 each)
        num_in_entries = struct.unpack_from(">H", raw, offset)[0]
        num_out_entries = struct.unpack_from(">H", raw, offset + 2)[0]
        offset += 4

        # Input tables
        input_tables = []
        for _ in range(num_in):
            vals = struct.unpack_from(f">{num_in_entries}H", raw, offset)
            input_tables.append(np.array(vals, dtype=np.float64) / 65535.0)
            offset += num_in_entries * 2
        components["input_tables"] = input_tables

        # CLUT
        clut_entries = (grid ** num_in) * num_out
        vals = struct.unpack_from(f">{clut_entries}H", raw, offset)
        components["clut"] = np.array(vals, dtype=np.float64).reshape(-1, num_out) / 65535.0
        offset += clut_entries * 2

        # Output tables
        output_tables = []
        for _ in range(num_out):
            vals = struct.unpack_from(f">{num_out_entries}H", raw, offset)
            output_tables.append(np.array(vals, dtype=np.float64) / 65535.0)
            offset += num_out_entries * 2
        components["output_tables"] = output_tables

        return components

    # ------------------------------------------------------------------ #
    #  lutAToBType – 'mAB '                                              #
    # ------------------------------------------------------------------ #
    elif sig == "mAB ":
        components = {"type": "mAB"}

        num_in = raw[8]
        num_out = raw[9]
        components["num_input_channels"] = num_in
        components["num_output_channels"] = num_out

        # Offsets to each element (0 means element absent)
        off_b      = struct.unpack_from(">I", raw, 12)[0]
        off_matrix = struct.unpack_from(">I", raw, 16)[0]
        off_m      = struct.unpack_from(">I", raw, 20)[0]
        off_clut   = struct.unpack_from(">I", raw, 24)[0]
        off_a      = struct.unpack_from(">I", raw, 28)[0]

        def read_curves(base_offset, n_channels):
            curves = []
            cur = base_offset
            for _ in range(n_channels):
                curves.append(decode_curve_tag(raw, cur))
                cur += curve_byte_size(raw, cur)
            return curves

        # B curves (output side – always present)
        components["b_curves"] = read_curves(off_b, num_out) if off_b else None

        # Matrix (3×4: 3×3 + 3 offsets), only when present
        if off_matrix:
            vals = [s15f16(raw, off_matrix + i * 4) for i in range(12)]
            mat  = np.array(vals[:9],  dtype=np.float64).reshape(3, 3)
            bias = np.array(vals[9:12], dtype=np.float64)
            components["matrix"] = mat
            components["matrix_bias"] = bias
        else:
            components["matrix"]      = None
            components["matrix_bias"] = None

        # M curves
        components["m_curves"] = read_curves(off_m, num_out) if off_m else None

        # CLUT
        if off_clut:
            grid_points = list(raw[off_clut:off_clut + 16])[:num_in]   # one per input channel
            precision   = raw[off_clut + 16]                           # 1 = uint8, 2 = uint16
            clut_offset = off_clut + 20                                # 16 grid + 1 prec + 3 pad

            total_entries = 1
            for g in grid_points:
                total_entries *= g
            total_entries *= num_out

            if precision == 1:
                vals = np.frombuffer(raw, dtype=np.uint8, count=total_entries, offset=clut_offset).astype(np.float64) / 255.0
            else:
                vals = np.array(struct.unpack_from(f">{total_entries}H", raw, clut_offset), dtype=np.float64) / 65535.0

            components["clut"]            = vals.reshape(-1, num_out)
            components["clut_grid_points"] = grid_points
        else:
            components["clut"]            = None
            components["clut_grid_points"] = None

        # A curves
        components["a_curves"] = read_curves(off_a, num_in) if off_a else None

        return components

    # ------------------------------------------------------------------ #
    #  lutBToAType – 'mBA '                                              #
    # ------------------------------------------------------------------ #
    elif sig == "mBA ":
        # Same binary layout as mAB but pipeline runs in reverse:
        # B → Matrix → M → CLUT → A
        components = {"type": "mBA"}

        num_in = raw[8]
        num_out = raw[9]
        components["num_input_channels"] = num_in
        components["num_output_channels"] = num_out

        off_b = struct.unpack_from(">I", raw, 12)[0]
        off_matrix = struct.unpack_from(">I", raw, 16)[0]
        off_m = struct.unpack_from(">I", raw, 20)[0]
        off_clut = struct.unpack_from(">I", raw, 24)[0]
        off_a = struct.unpack_from(">I", raw, 28)[0]

        def read_curves(base_offset, n_channels):
            curves = []
            cur = base_offset
            for _ in range(n_channels):
                curves.append(decode_curve_tag(raw, cur))
                cur += curve_byte_size(raw, cur)
            return curves

        components["b_curves"] = read_curves(off_b, num_in) if off_b else None

        if off_matrix:
            vals = [s15f16(raw, off_matrix + i * 4) for i in range(12)]
            components["matrix"] = np.array(vals[:9], dtype=np.float64).reshape(3, 3)
            components["matrix_bias"] = np.array(vals[9:12], dtype=np.float64)
        else:
            components["matrix"] = None
            components["matrix_bias"] = None

        components["m_curves"] = read_curves(off_m, num_in) if off_m else None

        if off_clut:
            grid_points = list(raw[off_clut:off_clut + 16])[:num_out]
            precision   = raw[off_clut + 16]
            clut_offset = off_clut + 20

            total_entries = 1
            for g in grid_points:
                total_entries *= g
            total_entries *= num_out

            if precision == 1:
                vals = np.frombuffer(raw, dtype=np.uint8, count=total_entries, offset=clut_offset).astype(np.float64) / 255.0
            else:
                vals = np.array(struct.unpack_from(f">{total_entries}H", raw, clut_offset), dtype=np.float64) / 65535.0

            components["clut"]             = vals.reshape(-1, num_out)
            components["clut_grid_points"] = grid_points
        else:
            components["clut"]             = None
            components["clut_grid_points"] = None

        components["a_curves"] = read_curves(off_a, num_out) if off_a else None

        return components

    raise NotImplementedError(f"cannot decode A2B of type {sig!r}")


def apply_a2b(pixels, a2b_data, lut_interpolation='linear'):
    """
    Convert a pixel array through an ICC A2B (or B2A) LUT pipeline.

    Parameters
    ----------
    pixels   : np.ndarray, shape (H, W, C), float64, values in [0, 1]
    a2b_data : dict returned by decode_lut()
    lut_interpolation : str, interpolation method for CLUT ('linear' or 'tetrahedral')

    Returns
    -------
    np.ndarray, shape (H, W, C_out), float64, values in [0, 1]
    """
    H, W, C = pixels.shape
    lut_type = a2b_data['type']

    # Flatten to (N, C) for all pipeline steps, reshape at the end
    p = pixels.reshape(-1, C)

    if lut_type in ('mft1', 'mft2'):
        # Fixed pipeline: input tables → matrix → CLUT → output tables
        p = apply_input_tables(p, a2b_data['input_tables'])
        if a2b_data['matrix'] is not None:
            p = apply_matrix(p, a2b_data['matrix'], bias=None)
        p = apply_clut(p, a2b_data['clut'], a2b_data['clut_grid_size'], lut_interpolation)
        p = apply_output_tables(p, a2b_data['output_tables'])
        #print("DEBUG: value ranges after output tables: min={:.4f}, max={:.4f}, mean={:.4f}".format(p.min(), p.max(), p.mean()))

    elif lut_type == 'mAB':
        # A → CLUT → M → Matrix → B
        if a2b_data['a_curves'] is not None:
            p = apply_curves(p, a2b_data['a_curves'])
        if a2b_data['clut'] is not None:
            p = apply_clut(p, a2b_data['clut'], a2b_data['clut_grid_points'], lut_interpolation)
        if a2b_data['m_curves'] is not None:
            p = apply_curves(p, a2b_data['m_curves'])
        if a2b_data['matrix'] is not None:
            p = apply_matrix(p, a2b_data['matrix'], a2b_data['matrix_bias'])
        if a2b_data['b_curves'] is not None:
            p = apply_curves(p, a2b_data['b_curves'])

    elif lut_type == 'mBA':
        # B → Matrix → M → CLUT → A
        if a2b_data['b_curves'] is not None:
            p = apply_curves(p, a2b_data['b_curves'])
            #print(f"after B curve: {p.min():.4f} {p.max():.4f}")
        if a2b_data['matrix'] is not None:
            p = apply_matrix(p, a2b_data['matrix'], a2b_data['matrix_bias'])
        if a2b_data['m_curves'] is not None:
            p = apply_curves(p, a2b_data['m_curves'])
            #print(f"after M curve: {p.min():.4f} {p.max():.4f}")
        if a2b_data['clut'] is not None:
            p = apply_clut(p, a2b_data['clut'], a2b_data['clut_grid_points'], lut_interpolation)
        if a2b_data['a_curves'] is not None:
            p = apply_curves(p, a2b_data['a_curves'])
            #print(f"after A curve: {p.min():.4f} {p.max():.4f}")

    else:
        raise NotImplementedError(f"Unknown LUT type: {lut_type!r}")

    C_out = p.shape[1]
    return p.reshape(H, W, C_out)


def _apply_1d_tables(p, tables):
    """Shared implementation for input and output 1-D LUT lookup."""
    result = np.empty_like(p)
    for i, table in enumerate(tables):
        xp = np.linspace(0.0, 1.0, len(table))
        result[:, i] = np.interp(p[:, i], xp, table)
    return result


def apply_input_tables(p, tables):
    """
    Apply per-channel 1-D input LUTs via linear interpolation.

    Parameters
    ----------
    p      : np.ndarray (N, C_in),  float64, [0, 1]
    tables : list of C_in np.ndarrays, each of arbitrary length, [0, 1]

    Returns
    -------
    np.ndarray (N, C_in), float64
    """
    #print("\napply_input_tables")
    return _apply_1d_tables(p, tables)


def apply_output_tables(p, tables):
    """
    Apply per-channel 1-D output LUTs via linear interpolation.

    Parameters
    ----------
    p      : np.ndarray (N, C_out), float64, [0, 1]
    tables : list of C_out np.ndarrays, each of arbitrary length, [0, 1]

    Returns
    -------
    np.ndarray (N, C_out), float64
    """
    #print("\napply_output_tables")
    return _apply_1d_tables(p, tables)


def apply_curves(p, curves):
    """
    Apply per-channel tone curves (a-, m-, or b-curves from mAB/mBA tags).

    Each curve was decoded by decode_lut() into a 1-D numpy array sampled
    uniformly over [0, 1], representing one of:
      - Identity (256-point linear ramp)
      - Gamma curve (256-point power function)
      - Arbitrary LUT (the raw curve entries, any length)
      - Parametric curve (pre-sampled to 256 points)

    All of these are already in the same normalised ndarray form, so the
    application is identical in every case: linear interpolation over [0, 1].

    Parameters
    ----------
    p      : np.ndarray (N, C), float64, [0, 1]
    curves : list of C np.ndarrays, each of arbitrary length, [0, 1]

    Returns
    -------
    np.ndarray (N, C), float64
    """
    #print("\napply_curves")
    return _apply_1d_tables(p, curves)


def apply_matrix(p, matrix, bias=None):
    """
    Apply a linear matrix transform and optional bias (translation).

    Computes:  p_out = p @ matrix.T + bias

    Parameters
    ----------
    p      : np.ndarray (N, C), float64
    matrix : np.ndarray (C_out, C_in), float64  — e.g. shape (3, 3)
    bias   : np.ndarray (C_out,) or None        — the matrix_bias offsets
                                                   from mAB/mBA tags

    Returns
    -------
    np.ndarray (N, C_out), float64
    """
    result = p @ matrix.T
    if bias is not None:
        result = result + bias          # broadcast over N; avoids in-place alloc issue
    # DEBUG
    #print("\napply_matrix:")
    #print(matrix)
    #print("\nbias:")
    #print(bias)
    #print(f"DEBUG: p after matrix: {p.min():.4f} {p.max():.4f}")
    return result


def apply_clut_multilinear(p, clut, grid):
    """
    Apply an n-D colour lookup table using multilinear (n-linear) interpolation.

    Parameters
    ----------
    p    : np.ndarray (N, C_in),  float64, [0, 1]
    clut : np.ndarray (total_nodes, C_out), float64, [0, 1]
             Nodes are in the ICC-standard layout: last input channel varies fastest.
    grid : int or list[int]
             Uniform grid size (mft1/mft2) or per-channel grid points (mAB/mBA).

    Returns
    -------
    np.ndarray (N, C_out), float64
    """
    N, n_in = p.shape
    n_out   = clut.shape[1]

    #print("\napply_clut - p shape:", p.shape, "clut shape:", clut.shape, "grid:", grid)

    # Normalise grid description to a per-channel list
    if isinstance(grid, int):
        grid_points = [grid] * n_in
    else:
        grid_points = list(grid)

    gp = np.array(grid_points, dtype=np.float64)

    # Pre-compute flat-index strides: last channel varies fastest (ICC layout)
    strides = np.ones(n_in, dtype=np.int64)
    for i in range(n_in - 2, -1, -1):
        strides[i] = strides[i + 1] * grid_points[i + 1]

    # Map [0, 1] → [0, g-1] and separate into floor index + fraction
    coords    = np.clip(p, 0.0, 1.0) * (gp - 1.0)          # (N, n_in)
    floor_idx = np.floor(coords).astype(np.int64)            # (N, n_in)

    # Clamp so that floor+1 is always a valid index (handles p == 1.0 exactly)
    max_floor = (np.array(grid_points, dtype=np.int64) - 2).clip(min=0)
    floor_idx = np.minimum(floor_idx, max_floor)             # (N, n_in)
    frac      = coords - floor_idx                           # (N, n_in), in [0, 1]

    # Multilinear interpolation: iterate over all 2^n_in corners
    result = np.zeros((N, n_out), dtype=np.float64)

    for corner in range(1 << n_in):
        # Build per-pixel flat index and interpolation weight for this corner
        weight    = np.ones(N, dtype=np.float64)
        flat_idx  = np.zeros(N, dtype=np.int64)

        for dim in range(n_in):
            if corner & (1 << dim):               # use ceil node on this dimension
                flat_idx += (floor_idx[:, dim] + 1) * strides[dim]
                weight   *= frac[:, dim]
            else:                                  # use floor node on this dimension
                flat_idx += floor_idx[:, dim] * strides[dim]
                weight   *= (1.0 - frac[:, dim])

        result += weight[:, np.newaxis] * clut[flat_idx]     # (N, n_out)

    # DEBUG: print result stats
    #print("DEBUG: LUT result stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(result.min(), result.max(), result.mean()))
    return result


def apply_clut_scipy(p, clut, grid, order=1):
    """
    CLUT interpolation via scipy.ndimage.map_coordinates.

    Advantages over the pure-NumPy multilinear implementation:
      - order=1 : same linear result, but faster for large N (C inner loop)
      - order=3 : tricubic spline — smoother gradients, better for smooth CLUTs
      - Handles arbitrary n_in and non-uniform grid sizes naturally.

    Parameters
    ----------
    p     : np.ndarray (N, n_in),  float64, [0, 1]
    clut  : np.ndarray (total_nodes, n_out), float64, [0, 1]
    grid  : int or list[int]
    order : int, 1 (linear) or 3 (cubic spline) — passed to map_coordinates

    Returns
    -------
    np.ndarray (N, n_out), float64
    """
    from scipy.ndimage import map_coordinates

    N, n_in = p.shape
    n_out = clut.shape[1]

    if isinstance(grid, int):
        grid_points = [grid] * n_in
    else:
        grid_points = list(grid)

    gp = np.array(grid_points, dtype=np.float64)

    # Reshape flat CLUT to (g0, g1, …, g_{n-1}, n_out) — C order matches ICC layout
    clut_grid = clut.reshape(grid_points + [n_out])

    # Pixel coordinates in grid-index space, shape (n_in, N) as map_coordinates expects
    #coords = np.clip(p, 0.0, 1.0).T * (gp - 1.0)[:, np.newaxis]   # (n_in, N), clipping apparently not needed
    #print(f"DEBUG: p before CLUT: {p.min():.4f} {p.max():.4f}")
    coords = p.T * (gp - 1.0)[:, np.newaxis]   # (n_in, N)

    # One map_coordinates call per output channel
    result = np.empty((N, n_out), dtype=np.float64)
    for c in range(n_out):
        result[:, c] = map_coordinates(
            clut_grid[..., c],
            coords,
            order=order,
            mode='nearest',   # clamp — matches ICC out-of-range behaviour
            prefilter=False,  # prefilter=True only needed when order > 1 and
                              # you want pure spline; False avoids the ringing
                              # artefact at hard-clipped CLUT boundaries
        )

    return result


def apply_clut_tetrahedral(p, clut, grid):
    """
    CLUT interpolation via colour.algebra.table_interpolation_tetrahedral.

    Tetrahedral interpolation is the industry-standard method for ICC colour
    management (used by LittleCMS, Adobe ACE, etc.). Each unit cube of the
    CLUT is split into 6 tetrahedra; the output is a weighted sum of the 4
    enclosing vertices rather than all 8 corners.

    Advantages over multilinear:
      - More accurate at saturated colours (multilinear introduces a slight
        bias toward the cube's body diagonal).
      - Faster: 4 vertex lookups instead of 2^n_in = 8.
      - Matches what reference CMMs produce, making round-trip comparison easier.

    Limitation: only valid for n_in == 3. For other channel counts, fall back
    to apply_clut_multilinear() or apply_clut_scipy().

    Parameters
    ----------
    p    : np.ndarray (N, 3),  float64, [0, 1]
    clut : np.ndarray (total_nodes, n_out), float64, [0, 1]
    grid : int or list[int] — must be uniform (single grid size) or a list of
           3 equal values; colour's implementation requires a cubic grid.

    Returns
    -------
    np.ndarray (N, n_out), float64
    """
    import colour

    N, n_in = p.shape
    if n_in != 3:
        raise ValueError(
            f"Tetrahedral interpolation requires exactly 3 input channels, got {n_in}. "
            "Use apply_clut_multilinear() or apply_clut_scipy() for other channel counts."
        )

    if isinstance(grid, int):
        grid_points = [grid] * 3
    else:
        grid_points = list(grid)

    if len(set(grid_points)) != 1:
        raise ValueError(
            f"Tetrahedral interpolation requires a uniform grid; got {grid_points}. "
            "Use apply_clut_multilinear() or apply_clut_scipy() for non-uniform grids."
        )

    n_out = clut.shape[1]

    # colour expects shape (g, g, g, n_out) in C order — identical to ICC layout
    clut_grid = clut.reshape(grid_points + [n_out])

    # colour's implementation accepts (..., 3) and returns (..., n_out)
    result = colour.algebra.table_interpolation_tetrahedral(
        np.clip(p, 0.0, 1.0),
        clut_grid,
    )

    return np.asarray(result, dtype=np.float64)


def apply_clut(p, clut, grid, interpolation='linear'):
    """
    Apply a CLUT with selectable interpolation method.

    interpolation : 'linear'       — scipy linear (order=1) or cubic (order=3)
                    'tetrahedral'  — colour-science, n_in==3 uniform grid only (default)
    """
    method = 'scipy' if interpolation == 'linear' else 'tetrahedral'
    #print("\napply_clut with method:", method)
    if method == 'tetrahedral':
        _, n_in = p.shape
        if n_in == 3 and (isinstance(grid, int) or len(set(grid)) == 1):
            return apply_clut_tetrahedral(p, clut, grid)
        # silent fallback for non-3-channel or non-uniform CLUTs
        return apply_clut_scipy(p, clut, grid)
    elif method == 'scipy':
        return apply_clut_scipy(p, clut, grid)
    else:
        return apply_clut_multilinear(p, clut, grid)


# --- for debugging only: denormalize XYZ, Lab


def denormalize_pcs_lab(p, lut_type):
    """
    Convert ICC-normalized [0, 1] pipeline output to actual Lab values.

    ICC Lab PCS encoding by LUT type:

      mft1 (LUT8):
        L*  : encoded as L * 255/100         → divide by 255 → ÷100 scaling
        a*,b*: encoded as (a* + 128)         → divide by 255 → ×255 − 128

      mft2 (LUT16):
        L*  : encoded as L * 65280/100       → divide by 65535 → ×(100 × 65535/65280)
        a*,b*: encoded as (a* + 128)*65535/255 → divide by 65535 → ×255 − 128

      mAB / mBA:
        L*  : B-curve output directly in [0, 1] → ×100
        a*,b*: B-curve output directly in [0, 1] → ×255 − 128

    Parameters
    ----------
    p        : np.ndarray (N, 3), float64, ICC-normalized [0, 1]
    lut_type : str – 'mft1' | 'mft2' | 'mAB' | 'mBA'

    Returns
    -------
    np.ndarray (N, 3), float64 — (L*, a*, b*) in colorimetric ranges
    """
    result = np.empty_like(p)

    if lut_type == 'mft2':
        result[:, 0] = p[:, 0] * (100.0 * 65535.0 / 65280.0)  # ≈ 100.39
    else:
        # mft1: normalized by 255, L encoded as L/100 exactly
        # mAB/mBA: B-curves output L/100 directly
        result[:, 0] = p[:, 0] * 100.0

    # a* and b*: identical formula for all LUT types
    result[:, 1] = p[:, 1] * 255.0 - 128.0
    result[:, 2] = p[:, 2] * 255.0 - 128.0

    return result


def denormalize_pcs_xyz(p, lut_type):
    """
    Convert ICC-normalized [0, 1] pipeline output to actual XYZ values.

    ICC XYZ PCS encoding:
      mft1/mft2: encoded as X * 32768/65535 (= X / 2 approx)
                 → divide by 65535 → ×(65535/32768) ≈ ×1.9999
      mAB/mBA  : B-curve output normalized as X/2.0 directly → ×2.0

    Parameters
    ----------
    p        : np.ndarray (N, 3), float64, ICC-normalized [0, 1]
    lut_type : str – 'mft1' | 'mft2' | 'mAB' | 'mBA'

    Returns
    -------
    np.ndarray (N, 3), float64 — (X, Y, Z) where 1.0 = D50 white
    """
    if lut_type in ('mft1', 'mft2'):
        return p * (65535.0 / 32768.0)
    else:
        return p * 2.0


def denormalize_pcs(p, pcs, lut_type):
    """Route to the correct PCS denormalization."""
    if pcs == 'Lab':
        return denormalize_pcs_lab(p, lut_type)
    elif pcs == 'XYZ':
        return denormalize_pcs_xyz(p, lut_type)
    else:
        raise NotImplementedError(f"Unknown PCS: {pcs!r}")
