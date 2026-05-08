import struct
from PIL import Image
import shutil
import tempfile
from pathlib import Path
import exiftool
import numpy as np


def extract_jpeg_info_from_tiff(im: Image.Image):
    """
    Extracts JPEG Q-tables and subsampling from a TIFF file with JPEG compression.
    Bypasses PIL's missing tag limitations by parsing the DQT and SOF markers 
    directly from the raw JPEG stream embedded in the file.

    Args:
        im: PIL.Image object of an opened TIFF file (must still have an active `fp`).

    Returns:
        q_tables: dict of {table_id: list_of_64_ints} (in zigzag order, as stored)
        subsampling: string representing subsampling (e.g., '4:2:0', '4:4:4', '4:2:2')
    """
    if im.format not in ('TIFF', None):
        raise ValueError("Image does not appear to be a TIFF.")

    # 1. Determine the byte offset for the raw JPEG data
    first_offset = None
    if hasattr(im, "tag_v2"):
        # Tag 273: StripOffsets, Tag 324: TileOffsets
        if 273 in im.tag_v2:
            offsets = im.tag_v2[273]
            first_offset = offsets[0] if isinstance(offsets, (tuple, list)) else offsets
        elif 324 in im.tag_v2:
            offsets = im.tag_v2[324]
            first_offset = offsets[0] if isinstance(offsets, (tuple, list)) else offsets

    # Fallback to PIL's internal tile tuple if tags aren't populated
    if first_offset is None and hasattr(im, "tile"):
        for tile in im.tile:
            if isinstance(tile[0], str) and 'jpeg' in tile[0]:
                first_offset = tile[2]
                break

    if first_offset is None:
        raise ValueError("Could not determine offset for JPEG data in TIFF.")

    # 2. Check for global JPEGTables tag (347) which may store Q-Tables globally
    jpeg_tables_data = b""
    if hasattr(im, "tag_v2") and 347 in im.tag_v2:
        jpeg_tables_data = im.tag_v2[347]

    # 3. Define a helper to parse the JPEG stream markers
    def parse_jpeg_stream(data):
        q_tables = {}
        subsampling_factors = {}

        i = 0
        while i < len(data) - 1:
            if data[i] == 0xFF:
                marker = data[i+1]
                # Skip padding and SOI
                if marker in (0xFF, 0x00, 0xD8):
                    i += 1 if marker == 0xFF else 2
                    continue
                # SOS (Start of Scan) - Header parsing is complete
                if marker == 0xDA: 
                    break

                if i + 3 >= len(data):
                    break

                # Read marker length (includes the 2 length bytes themselves)
                length = struct.unpack(">H", data[i+2:i+4])[0]

                if i + 2 + length > len(data):
                    break # Incomplete marker in our chunk

                if marker == 0xDB: # DQT (Define Quantization Table)
                    payload_idx = i + 4
                    end_idx = i + 2 + length
                    while payload_idx < end_idx:
                        info = data[payload_idx]
                        prec = info >> 4
                        tbl_id = info & 0x0F
                        payload_idx += 1

                        tbl_len = 64 if prec == 0 else 128
                        if payload_idx + tbl_len <= end_idx:
                            # Extract raw Q-table. JPEG standard stores DQT in zigzag.
                            # PIL's save(qtables=...) natively expects zigzag order.
                            q_tables[tbl_id] = list(data[payload_idx:payload_idx+tbl_len])
                        payload_idx += tbl_len

                elif marker in (0xC0, 0xC1, 0xC2): # SOF0, SOF1, SOF2 (Start of Frame)
                    num_components = data[i + 9]
                    offset = i + 10
                    for _ in range(num_components):
                        if offset + 2 >= i + 2 + length:
                            break
                        comp_id = data[offset]
                        factors = data[offset + 1]
                        h_samp = factors >> 4
                        v_samp = factors & 0x0F
                        subsampling_factors[comp_id] = (h_samp, v_samp)
                        offset += 3
                        
                i += 2 + length # Jump to next marker
            else:
                i += 1

        return q_tables, subsampling_factors

    q_tables = {}

    # Extract from JPEGTables tag first, if present
    if jpeg_tables_data:
        qt, _ = parse_jpeg_stream(jpeg_tables_data)
        q_tables.update(qt)

    # Jump to offset and read a chunk (64KB is overkill, but guarantees we hit SOS)
    fp = im.fp
    if fp is None:
        raise ValueError("Image file pointer is closed. Call this immediately after Image.open().")

    original_pos = fp.tell()
    fp.seek(first_offset)
    data = fp.read(65536)
    fp.seek(original_pos) # Restore pointer just in case PIL needs it

    qt, subsamp = parse_jpeg_stream(data)
    q_tables.update(qt)

    # 4. Map the component sampling factors to standard PIL subsampling strings
    subsampling_str = None
    if len(subsamp) == 3:
        comp_ids = list(subsamp.keys())
        y_factors = subsamp[comp_ids[0]]
        cb_factors = subsamp[comp_ids[1]]
        cr_factors = subsamp[comp_ids[2]]

        if cb_factors == (1, 1) and cr_factors == (1, 1):
            if y_factors == (2, 2):
                subsampling_str = "4:2:0"
            elif y_factors == (2, 1):
                subsampling_str = "4:2:2"
            elif y_factors == (1, 1):
                subsampling_str = "4:4:4"
            elif y_factors == (1, 2):
                subsampling_str = "4:4:0"
    elif len(subsamp) == 1:
        subsampling_str = "4:4:4" # Effective subsampling for Grayscale

    return q_tables, subsampling_str


# --- TIFF with JPEG-compression
"""
Transfer metadata (EXIF/MakerNotes, XMP, ICC Profile) from a source TIFF
to a target TIFF with JPEG compression (multi-strip), without corrupting
the target's image data.

Root cause: exiftool -tagsfromfile with -all:all or -EXIF:all rewrites
the entire IFD, collapsing multi-strip JPEG into a single strip and
corrupting StripOffsets/StripByteCounts.

This script uses a binary append strategy:
  1. The entire target file is kept unchanged (image strips stay at original offsets)
  2. Metadata blobs from source are appended at the end
  3. A new IFD0 is appended, pointing to both original structural data AND new metadata
  4. Only the 4-byte IFD0 pointer in the TIFF header is updated
"""
 # Tags that describe image structure - always taken from TARGET (never overwritten)
TIFF_STRUCTURAL_TAGS = {
    0x0100,  # ImageWidth
    0x0101,  # ImageLength
    0x0102,  # BitsPerSample
    0x0103,  # Compression
    0x0106,  # PhotometricInterpretation
    0x0111,  # StripOffsets
    0x0115,  # SamplesPerPixel
    0x0116,  # RowsPerStrip
    0x0117,  # StripByteCounts
    0x011c,  # PlanarConfiguration
    0x015b,  # JPEGTables
    0x0142,  # TileWidth
    0x0143,  # TileLength
    0x0144,  # TileOffsets
    0x0145,  # TileByteCounts
}

# TIFF data type sizes in bytes
TIFF_TYPE_SIZE = {1:1, 2:1, 3:2, 4:4, 5:8, 6:1, 7:1, 8:2, 9:4, 10:8, 11:4, 12:8}

def get_fmt(data):
    bo = data[:2]
    if bo == b'II':
        return '<'
    elif bo == b'MM':
        return '>'
    raise ValueError(f"Unknown byte order: {bo!r}")


def read_ifd_entries(data, ifd_offset, fmt):
    """Return list of (tag, type_, count, raw_4bytes) tuples."""
    num = struct.unpack_from(fmt+'H', data, ifd_offset)[0]
    entries = []
    for i in range(num):
        pos = ifd_offset + 2 + i * 12
        tag, type_, count = struct.unpack_from(fmt+'HHI', data, pos)
        raw_val = data[pos+8:pos+12]
        entries.append((tag, type_, count, raw_val))
    next_ifd = struct.unpack_from(fmt+'I', data, ifd_offset + 2 + num * 12)[0]
    return entries, next_ifd


def get_tag_bytes(data, type_, count, raw_val, fmt):
    """Return the raw data bytes for a tag value."""
    ts = TIFF_TYPE_SIZE.get(type_, 1)
    total = count * ts
    if total <= 4:
        return bytes(raw_val[:total])
    else:
        off = struct.unpack_from(fmt+'I', raw_val)[0]
        return bytes(data[off:off + total])


def copy_subifd(src_data, subifd_offset, src_fmt, dst_buf, dst_fmt):
    """
    Copy a sub-IFD (e.g. ExifIFD, GPS IFD) from src to the end of dst_buf.
    Returns the new offset of the sub-IFD in dst_buf.
    Recursively handles nested IFDs (e.g. MakerNotes if they contain an IFD).
    """
    entries, _ = read_ifd_entries(src_data, subifd_offset, src_fmt)

    # Nested IFD pointer tags within ExifIFD
    NESTED_IFD_TAGS = {0x8769, 0x8825, 0xa005}  # ExifIFD, GPS, Interop

    # First pass: collect all data blobs and their new offsets
    new_offsets = {}   # tag -> new offset in dst_buf for data
    blobs = []         # (tag, bytes_data) to append

    for tag, type_, count, raw_val in entries:
        ts = TIFF_TYPE_SIZE.get(type_, 1)
        total = count * ts
        if total <= 4:
            continue  # inline, no separate data needed

        if tag in NESTED_IFD_TAGS:
            # This is a pointer to a nested IFD — handle recursively
            sub_offset = struct.unpack_from(src_fmt+'I', raw_val)[0]
            new_sub_offset = copy_subifd(src_data, sub_offset, src_fmt, dst_buf, dst_fmt)
            new_offsets[tag] = new_sub_offset  # store the IFD offset, not data offset
        else:
            src_off = struct.unpack_from(src_fmt+'I', raw_val)[0]
            blob = bytes(src_data[src_off:src_off + total])
            blobs.append((tag, blob))

    # Append all data blobs
    for tag, blob in blobs:
        # Align to 2 bytes
        if len(dst_buf) % 2:
            dst_buf += b'\x00'
        new_offsets[tag] = len(dst_buf)
        dst_buf += blob

    # Align before IFD
    if len(dst_buf) % 2:
        dst_buf += b'\x00'

    # Write IFD
    ifd_offset = len(dst_buf)
    dst_buf += struct.pack(dst_fmt+'H', len(entries))

    for tag, type_, count, raw_val in sorted(entries, key=lambda x: x[0]):
        ts = TIFF_TYPE_SIZE.get(type_, 1)
        total = count * ts

        if total <= 4:
            # Inline value — but may need byte-order conversion if src != dst
            # For simplicity, just use the raw bytes (both are little-endian typically)
            dst_buf += struct.pack(dst_fmt+'HHI', tag, type_, count)
            dst_buf += bytes(raw_val)
        elif tag in new_offsets and tag in NESTED_IFD_TAGS:
            # nested IFD pointer
            dst_buf += struct.pack(dst_fmt+'HHI', tag, type_, count)
            dst_buf += struct.pack(dst_fmt+'I', new_offsets[tag])
        else:
            dst_buf += struct.pack(dst_fmt+'HHI', tag, type_, count)
            dst_buf += struct.pack(dst_fmt+'I', new_offsets[tag])

    dst_buf += struct.pack(dst_fmt+'I', 0)  # no next IFD
    return ifd_offset, dst_buf


def transfer_metadata_tiff2tiff(source_path, target_path, output_path):
    src = bytes(Path(source_path).read_bytes())
    tgt = bytes(Path(target_path).read_bytes())

    src_fmt = get_fmt(src)
    tgt_fmt = get_fmt(tgt)

    src_ifd0_off = struct.unpack_from(src_fmt+'I', src, 4)[0]
    tgt_ifd0_off = struct.unpack_from(tgt_fmt+'I', tgt, 4)[0]

    src_entries, _ = read_ifd_entries(src, src_ifd0_off, src_fmt)
    tgt_entries, _ = read_ifd_entries(tgt, tgt_ifd0_off, tgt_fmt)

    src_tag_map = {tag: (type_, count, raw_val) for tag, type_, count, raw_val in src_entries}
    tgt_tag_map = {tag: (type_, count, raw_val) for tag, type_, count, raw_val in tgt_entries}

    # Tags that need a sub-IFD copy (ExifIFD=0x8769, GPS=0x8825)
    SUBIFD_POINTER_TAGS = {0x8769, 0x8825, 0xa005}

    # ---- Build the new file ----
    # Start with the complete target binary (image strips at original positions)
    out = bytearray(tgt)

    # Collect new metadata from source
    new_tag_data = {}   # tag -> (type_, count, inline_or_offset, new_value_or_offset)

    # Decide which tags come from source vs target
    all_tags = set(src_tag_map) | set(tgt_tag_map)

    final_entries = {}  # tag -> (type_, count, value_bytes_4)

    for tag in all_tags:
        if tag in TIFF_STRUCTURAL_TAGS:
            # Always use target's structural tags
            if tag in tgt_tag_map:
                final_entries[tag] = tgt_tag_map[tag]
        elif tag in SUBIFD_POINTER_TAGS:
            # Sub-IFDs (ExifIFD, GPS): copy from source if present
            if tag in src_tag_map:
                type_, count, raw_val = src_tag_map[tag]
                sub_offset = struct.unpack_from(src_fmt+'I', raw_val)[0]
                new_sub_off, out = copy_subifd(src, sub_offset, src_fmt, out, tgt_fmt)
                final_entries[tag] = (type_, count, struct.pack(tgt_fmt+'I', new_sub_off))
            # If not in source, keep target's (if any)
            elif tag in tgt_tag_map:
                final_entries[tag] = tgt_tag_map[tag]
        else:
            # Regular metadata tag: prefer source, else keep target
            if tag in src_tag_map:
                type_, count, raw_val = src_tag_map[tag]
                ts = TIFF_TYPE_SIZE.get(type_, 1)
                total = count * ts
                if total <= 4:
                    final_entries[tag] = (type_, count, raw_val)
                else:
                    # Copy data blob to end of output
                    if len(out) % 2:
                        out += b'\x00'
                    src_off = struct.unpack_from(src_fmt+'I', raw_val)[0]
                    blob = src[src_off:src_off + total]
                    new_off = len(out)
                    out += blob
                    final_entries[tag] = (type_, count, struct.pack(tgt_fmt+'I', new_off))
            elif tag in tgt_tag_map:
                final_entries[tag] = tgt_tag_map[tag]

    # Align before writing IFD
    if len(out) % 2:
        out += b'\x00'

    # Write new IFD0
    new_ifd0_off = len(out)
    sorted_tags = sorted(final_entries.keys())
    out += struct.pack(tgt_fmt+'H', len(sorted_tags))

    for tag in sorted_tags:
        type_, count, value_4 = final_entries[tag]
        out += struct.pack(tgt_fmt+'HHI', tag, type_, count)
        out += bytes(value_4)

    out += struct.pack(tgt_fmt+'I', 0)  # no next IFD

    # Update TIFF header to point to new IFD0
    struct.pack_into(tgt_fmt+'I', out, 4, new_ifd0_off)

    Path(output_path).write_bytes(out)
# --- end TIFF with JPEG-compression


def verify_tiff_image(fn, reference=None, strict=False):
    """
    Return True if the TIFF image is valid, False otherwise.

    Args:
        fn: Path to the TIFF file to verify
        reference: Optional path to reference file for size comparison
        strict: If True, test also pixel data shape

    Returns:
        bool: True if verification passes, False otherwise
    """
    # Fastest test: file size should increase with metadata transferred
    if reference is not None:
        tolerance = 10000  # allow some invalid metadata be removed
        if Path(fn).stat().st_size + tolerance < Path(reference).stat().st_size:
            #print(f"exiftool abort: temp file size ({Path(fn).stat().st_size/1024/1024:.3f} MB) smaller than "
            #      f"original size ({Path(reference).stat().st_size/1024/1024:.3f} MB)")
            return False

    # Check header and magic byte
    with open(fn, 'rb') as f:
        magic = f.read(4)
    if not (magic.startswith(b'II*\x00') or magic.startswith(b'MM\x00*')):
        #print(f"exiftool abort: temp file lacks a valid TIFF header: {magic!r}")
        return False

    try:
        with Image.open(fn) as img:
            if strict:
                # Test if pixel data has correct shape
                pixel_data_size = np.array(img).shape[1::-1]
                if pixel_data_size == img.size:
                    return True
                else:
                    #print(f"exiftool abort: bad pixel data shape {pixel_data_size}, expected {img.size}")
                    return False

            # Only test if pixel data can be accessed (valid IDF0 link)
            _ = img.load()
    except Exception as e:
        #print(f"exiftool abort: failed to load image: {e}")
        return False

    return True


def exiftool_safe_transfer(src, dst, exiftool_path=None):
    """
    Safely copy metadata from src to dst TIFF without altering pixel data.

    Operates on a temporary copy of dst. The original dst is replaced only
    after the copy passes verification.
    Return True on success, False otherwise. Always cleans up the temp file.
    """
    src, dst = Path(src), Path(dst)

    with tempfile.NamedTemporaryFile(delete=False, suffix=dst.suffix) as tmp_f:
        tmp = Path(tmp_f.name)

    try:
        # If src is already TIFF, try with custom TIFF-to-TIFF writer first (faster, more reliable)
        if src.suffix.lower() in {'.tif', '.tiff'}:
            shutil.copy2(dst, tmp)
            transfer_metadata_tiff2tiff(str(src), str(dst), str(tmp))

        if verify_tiff_image(tmp, reference=src, strict=True):
            shutil.move(tmp, dst)
            return True

        # Try with exiftools
        args = [
            '-all=',  # erase any metadata (incl. ICC) from dst before transfer
            '-tagsFromFile', str(src),  # w/o further options, transfers all writable tags
                                        # but some values and tags are wrong
            '-exif:all', '-iptc:all', '-xmp:all',
            '-icc_profile<icc_profile',  # adds profile if missing/deleted
        ] + [str(tmp)]

        with exiftool.ExifTool(executable=exiftool_path) as et:
            shutil.copy2(dst, tmp)
            res = et.execute(*[a.encode() for a in args])  # slow

            if verify_tiff_image(tmp, reference=src, strict=True):
                shutil.move(tmp, dst)
                return True

            # Try again with --IFD0
            # prevents file corruption in some cases
            # but also excludes all its subgroups: ExifIFD, GlobParamIFD,
            # GPS, IFD1, InteropIFD, MakerNotes, PrintIM and SubIFD.
            shutil.copy2(dst, tmp)
            args.insert(-1, '--IFD0:all')
            res = et.execute(*[a.encode() for a in args])

            if verify_tiff_image(tmp, reference=src, strict=True):
                shutil.move(tmp, dst)
                return True

    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"exiftool failed for {dst}: {e}")

    return False