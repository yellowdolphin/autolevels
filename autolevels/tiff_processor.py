import struct
from PIL import Image


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
