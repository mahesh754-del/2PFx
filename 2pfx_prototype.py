# =============================================================================
# 2PF× — Two-Point Flow Lossless Video Compression (Pixel-Level Prototype)
# Fixed version with proper int16 addition + clipping
# =============================================================================

import numpy as np
import struct
from pathlib import Path
import os

class TwoPointFlowCompressor:
    def __init__(self):
        self.version = b"2PFX"

    def compress(self, frames: list[np.ndarray], output_path: str):
        if not frames:
            raise ValueError("No frames provided")
        H, W, C = frames[0].shape
        if C != 3:
            raise ValueError("Frames must be RGB (3 channels)")

        ref_frame = np.zeros((H, W, 3), dtype=np.int16)

        with open(output_path, "wb") as f:
            f.write(self.version)
            f.write(struct.pack("<III", len(frames), H, W))

            for frame in frames:
                frame_int = frame.astype(np.int16)
                delta = frame_int - ref_frame

                binary_bits = bytearray()
                decimal_vals = bytearray()
                bit_count = 0
                current_byte = 0

                for y in range(H):
                    for x in range(W):
                        for comp in delta[y, x]:  # dr, dg, db
                            abs_c = abs(comp)
                            if abs_c <= 7:
                                val = abs_c if comp >= 0 else (8 | abs_c)
                            else:
                                val = 8  # escape code
                            current_byte = (current_byte << 4) | val
                            bit_count += 4
                            if bit_count >= 8:
                                binary_bits.append(current_byte)
                                current_byte = 0
                                bit_count = 0
                            if abs_c > 7:
                                decimal_vals.extend(struct.pack("<h", comp))

                if bit_count > 0:
                    current_byte <<= (8 - bit_count)
                    binary_bits.append(current_byte)

                f.write(struct.pack("<II", len(binary_bits), len(decimal_vals)))
                f.write(binary_bits)
                f.write(decimal_vals)

                ref_frame = frame_int.copy()  # important: keep full int16 copy

        print(f"Compressed to {output_path} — Size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")

    def decompress(self, input_path: str) -> list[np.ndarray]:
        with open(input_path, "rb") as f:
            magic = f.read(4)
            if magic != self.version:
                raise ValueError("Not a 2PF× file")
            n_frames, H, W = struct.unpack("<III", f.read(12))

            ref_frame = np.zeros((H, W, 3), dtype=np.int16)
            frames_out = []

            for _ in range(n_frames):
                bin_len, dec_len = struct.unpack("<II", f.read(8))
                binary_data = f.read(bin_len)
                decimal_data = f.read(dec_len)

                delta = np.zeros((H, W, 3), dtype=np.int16)
                pixel_idx = 0
                dec_offset = 0
                bitstream = int.from_bytes(binary_data, "big")
                bits_left = len(binary_data) * 8

                while pixel_idx < H * W and bits_left >= 4:
                    y, x = divmod(pixel_idx, W)
                    for c in range(3):
                        if bits_left < 4:
                            break
                        val = (bitstream >> (bits_left - 4)) & 0xF
                        bits_left -= 4
                        if val == 8:
                            if dec_offset + 2 > dec_len:
                                raise ValueError("Missing large delta bytes")
                            comp = struct.unpack_from("<h", decimal_data, dec_offset)[0]
                            dec_offset += 2
                        else:
                            comp = val if val <= 7 else -(val & 7)
                        delta[y, x, c] = comp
                    pixel_idx += 1

                if dec_offset != dec_len:
                    raise ValueError(f"Decimal data mismatch: consumed {dec_offset}, expected {dec_len}")

                # Critical fix: compute in int16, then clip and cast
                frame_int = ref_frame + delta
                frame = np.clip(frame_int, 0, 255).astype(np.uint8)
                frames_out.append(frame)
                ref_frame = frame_int.copy()  # preserve full int16 for next delta

        return frames_out


# ─── Test with synthetic low-motion video ────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # Create 4 frames of 32×32 RGB, mostly static with small changes
    H, W = 32, 32
    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[8:24, 8:24] = [80, 140, 220]  # big block

    frames = []
    for i in range(4):
        frame = base.copy()
        if i > 0:
            # Add small noise only in the block
            noise = np.random.randint(-6, 7, size=(16,16,3), dtype=np.int8)
            frame[8:24, 8:24] = np.clip(frame[8:24, 8:24].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)

    original_total_bytes = sum(f.nbytes for f in frames)
    print(f"Original total size: {original_total_bytes} bytes ({original_total_bytes / (1024):.1f} KiB)")

    compressor = TwoPointFlowCompressor()
    test_file = "test_low_motion.2pfx"
    compressor.compress(frames, test_file)

    compressed_size = Path(test_file).stat().st_size
    print(f"Compressed size: {compressed_size} bytes ({compressed_size / (1024):.1f} KiB)")
    print(f"Compression ratio: {original_total_bytes / compressed_size:.2f}x")

    try:
        recovered = compressor.decompress(test_file)
        print("Decompression successful!")

        # Verify lossless
        lossless = all(np.array_equal(a, b) for a, b in zip(frames, recovered))
        print(f"Lossless reconstruction: {'YES' if lossless else 'NO'}")

        if not lossless:
            print("First differing pixel:")
            for i, (orig, rec) in enumerate(zip(frames, recovered)):
                if not np.array_equal(orig, rec):
                    diff = np.where(orig != rec)
                    if len(diff[0]) > 0:
                        y, x = diff[0][0], diff[1][0]
                        print(f"Frame {i}, pixel ({y},{x}): orig={orig[y,x].tolist()} vs rec={rec[y,x].tolist()}")
                    break
    except Exception as e:
        print("Error during test:", e)

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
