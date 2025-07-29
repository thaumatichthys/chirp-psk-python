import numpy as np

def score_ascii(byte_array):
    # Score how many bytes are regular printable ASCII characters (space to ~)
    return np.sum(((byte_array >= ord('a')) & (byte_array <= ord('z'))) | ((byte_array >= ord('a')) & (byte_array <= ord('z'))))

def bits_to_bytes(bit_array):
    # Ensure length is multiple of 8
    bit_array = bit_array[:len(bit_array) // 8 * 8]
    if len(bit_array) == 0:
        return np.array([], dtype=np.uint8)
    bits_reshaped = bit_array.reshape(-1, 8)
    return np.packbits(bits_reshaped, axis=1).flatten()

def try_alignments(bit_array, max_pad=8):
    best_score = -1
    best_bytes = None
    best_info = {}

    for pad in range(max_pad):
        for invert in [False, True]:
            padded = np.pad(bit_array, (pad, 0), constant_values=0)
            if invert:
                padded = 1 - padded
            byte_array = bits_to_bytes(padded)
            score = score_ascii(byte_array)

            if score > best_score:
                best_score = score
                best_bytes = byte_array
                best_info = {
                    "pad": pad,
                    "inverted": invert,
                    "score": score,
                    "decoded": ''.join(chr(b) if 0x20 <= b <= 0x7E else '.' for b in byte_array)
                }

    return best_info, best_bytes