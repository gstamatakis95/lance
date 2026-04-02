// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bit-packing utilities for TurboQuant codes.
//!
//! Packs b-bit quantization indices into compact byte arrays. Each code is a
//! Lloyd-Max bin index in [0, 2^b - 1]. Packing reduces storage by 8/b times
//! compared to storing each code as a full byte.
//!
//! # Layout (little-endian within each byte)
//!
//! ```text
//! b=4: two codes per byte (SIMD-friendly nibble extraction)
//!   byte[0] = code[0] | (code[1] << 4)
//!   byte[1] = code[2] | (code[3] << 4)
//!
//! b=2: four codes per byte
//!   byte[0] = code[0] | (code[1] << 2) | (code[2] << 4) | (code[3] << 6)
//!
//! b=1: eight codes per byte (same as RabitQ sign-bit packing)
//!   byte[0] = code[0] | (code[1] << 1) | ... | (code[7] << 7)
//!
//! b=8: one code per byte (trivial, no packing needed)
//!
//! b=3: codes span byte boundaries (general bit-stream packing)
//! ```
//!
//! # Packed sizes for d=768
//!
//! | Bits | Packed bytes | Formula |
//! |------|-------------|---------|
//! | 1    | 96          | 768/8   |
//! | 2    | 192         | 768/4   |
//! | 3    | 288         | ceil(768*3/8) |
//! | 4    | 384         | 768/2   |
//! | 8    | 768         | 768     |

use lance_core::{Error, Result};

/// Pack an array of b-bit codes (each stored as u8) into a compact byte array.
///
/// # Arguments
/// * `codes` - Quantization indices, each in [0, 2^num_bits - 1]
/// * `num_bits` - Bit-width per code (1-8)
///
/// # Returns
/// Packed byte array of length ceil(codes.len() * num_bits / 8)
pub fn pack_codes(codes: &[u8], num_bits: u32) -> Result<Vec<u8>> {
    match num_bits {
        1 => Ok(pack_1bit(codes)),
        2 => Ok(pack_2bit(codes)),
        3 => Ok(pack_generic(codes, 3)),
        4 => Ok(pack_4bit(codes)),
        8 => Ok(codes.to_vec()),
        b if b <= 8 => Ok(pack_generic(codes, b)),
        _ => Err(Error::invalid_input(format!(
            "num_bits must be 1-8, got {}",
            num_bits
        ))),
    }
}

/// Unpack a compact byte array back into individual b-bit codes.
///
/// # Arguments
/// * `packed` - Packed byte array
/// * `dim` - Number of codes to unpack
/// * `num_bits` - Bit-width per code (1-8)
///
/// # Returns
/// Array of `dim` u8 values, each in [0, 2^num_bits - 1]
pub fn unpack_codes(packed: &[u8], dim: usize, num_bits: u32) -> Result<Vec<u8>> {
    match num_bits {
        1 => Ok(unpack_1bit(packed, dim)),
        2 => Ok(unpack_2bit(packed, dim)),
        3 => Ok(unpack_generic(packed, dim, 3)),
        4 => Ok(unpack_4bit(packed, dim)),
        8 => Ok(packed[..dim].to_vec()),
        b if b <= 8 => Ok(unpack_generic(packed, dim, b)),
        _ => Err(Error::invalid_input(format!(
            "num_bits must be 1-8, got {}",
            num_bits
        ))),
    }
}

/// Returns the packed byte length for `dim` codes at `num_bits` bits each.
pub fn packed_len(dim: usize, num_bits: u32) -> usize {
    (dim * num_bits as usize + 7) / 8
}

// --- 1-bit packing: 8 codes per byte ---

fn pack_1bit(codes: &[u8]) -> Vec<u8> {
    let n_bytes = (codes.len() + 7) / 8;
    let mut packed = vec![0u8; n_bytes];
    for (bit_idx, &code) in codes.iter().enumerate() {
        if code & 1 != 0 {
            packed[bit_idx / 8] |= 1u8 << (bit_idx % 8);
        }
    }
    packed
}

fn unpack_1bit(packed: &[u8], dim: usize) -> Vec<u8> {
    let mut codes = vec![0u8; dim];
    for bit_idx in 0..dim {
        codes[bit_idx] = (packed[bit_idx / 8] >> (bit_idx % 8)) & 1;
    }
    codes
}

// --- 2-bit packing: 4 codes per byte ---

fn pack_2bit(codes: &[u8]) -> Vec<u8> {
    let n_bytes = (codes.len() + 3) / 4;
    let mut packed = vec![0u8; n_bytes];
    for (i, &code) in codes.iter().enumerate() {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        packed[byte_idx] |= (code & 0x03) << shift;
    }
    packed
}

fn unpack_2bit(packed: &[u8], dim: usize) -> Vec<u8> {
    let mut codes = vec![0u8; dim];
    for i in 0..dim {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        codes[i] = (packed[byte_idx] >> shift) & 0x03;
    }
    codes
}

// --- 4-bit packing: 2 codes per byte (SIMD-friendly nibble extraction) ---

fn pack_4bit(codes: &[u8]) -> Vec<u8> {
    let n_bytes = (codes.len() + 1) / 2;
    let mut packed = vec![0u8; n_bytes];
    for (i, &code) in codes.iter().enumerate() {
        let byte_idx = i / 2;
        if i % 2 == 0 {
            packed[byte_idx] |= code & 0x0F;
        } else {
            packed[byte_idx] |= (code & 0x0F) << 4;
        }
    }
    packed
}

fn unpack_4bit(packed: &[u8], dim: usize) -> Vec<u8> {
    let mut codes = vec![0u8; dim];
    for i in 0..dim {
        let byte_idx = i / 2;
        if i % 2 == 0 {
            codes[i] = packed[byte_idx] & 0x0F;
        } else {
            codes[i] = (packed[byte_idx] >> 4) & 0x0F;
        }
    }
    codes
}

// --- Generic packing for arbitrary bit-widths (e.g., 3-bit) ---

fn pack_generic(codes: &[u8], num_bits: u32) -> Vec<u8> {
    let total_bits = codes.len() * num_bits as usize;
    let n_bytes = (total_bits + 7) / 8;
    let mut packed = vec![0u8; n_bytes];
    let mask = (1u8 << num_bits) - 1;

    let mut bit_offset = 0usize;
    for &code in codes {
        let code = code & mask;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;

        // Write bits, possibly spanning two bytes
        packed[byte_idx] |= code << bit_idx;
        if bit_idx + num_bits as usize > 8 {
            packed[byte_idx + 1] |= code >> (8 - bit_idx);
        }

        bit_offset += num_bits as usize;
    }
    packed
}

fn unpack_generic(packed: &[u8], dim: usize, num_bits: u32) -> Vec<u8> {
    let mask = (1u8 << num_bits) - 1;
    let mut codes = vec![0u8; dim];

    let mut bit_offset = 0usize;
    for code in codes.iter_mut() {
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;

        let mut val = packed[byte_idx] >> bit_idx;
        if bit_idx + num_bits as usize > 8 && byte_idx + 1 < packed.len() {
            val |= packed[byte_idx + 1] << (8 - bit_idx);
        }
        *code = val & mask;

        bit_offset += num_bits as usize;
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip_1bit() {
        let codes: Vec<u8> = (0..128).map(|i| (i % 2) as u8).collect();
        let packed = pack_codes(&codes, 1).unwrap();
        assert_eq!(packed.len(), packed_len(128, 1));
        let unpacked = unpack_codes(&packed, 128, 1).unwrap();
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_2bit() {
        let codes: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
        let packed = pack_codes(&codes, 2).unwrap();
        assert_eq!(packed.len(), packed_len(256, 2));
        let unpacked = unpack_codes(&packed, 256, 2).unwrap();
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_3bit() {
        let codes: Vec<u8> = (0..200).map(|i| (i % 8) as u8).collect();
        let packed = pack_codes(&codes, 3).unwrap();
        assert_eq!(packed.len(), packed_len(200, 3));
        let unpacked = unpack_codes(&packed, 200, 3).unwrap();
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit() {
        let codes: Vec<u8> = (0..768).map(|i| (i % 16) as u8).collect();
        let packed = pack_codes(&codes, 4).unwrap();
        assert_eq!(packed.len(), packed_len(768, 4));
        let unpacked = unpack_codes(&packed, 768, 4).unwrap();
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_8bit() {
        let codes: Vec<u8> = (0..128).map(|i| i as u8).collect();
        let packed = pack_codes(&codes, 8).unwrap();
        assert_eq!(packed.len(), 128);
        let unpacked = unpack_codes(&packed, 128, 8).unwrap();
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_packed_len() {
        // d=768, b=4: 768*4/8 = 384
        assert_eq!(packed_len(768, 4), 384);
        // d=768, b=2: 768*2/8 = 192
        assert_eq!(packed_len(768, 2), 192);
        // d=768, b=1: 768/8 = 96
        assert_eq!(packed_len(768, 1), 96);
        // d=768, b=3: ceil(768*3/8) = 288
        assert_eq!(packed_len(768, 3), 288);
        // d=768, b=8: 768
        assert_eq!(packed_len(768, 8), 768);
    }

    #[test]
    fn test_4bit_nibble_layout() {
        // Verify that 4-bit packing stores low nibble first
        let codes = vec![0x0A, 0x05]; // two codes: 10, 5
        let packed = pack_4bit(&codes);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x5A); // 0x0A in low nibble, 0x05 in high nibble
    }

    #[test]
    fn test_odd_length_codes() {
        // Test with non-multiple-of-group-size code counts
        for b in [1, 2, 3, 4] {
            let dim = 13; // intentionally odd
            let max_val = (1u16 << b) - 1;
            let codes: Vec<u8> = (0..dim).map(|i| (i as u16 % (max_val + 1)) as u8).collect();
            let packed = pack_codes(&codes, b).unwrap();
            let unpacked = unpack_codes(&packed, dim, b).unwrap();
            assert_eq!(codes, unpacked, "roundtrip failed for b={}, dim={}", b, dim);
        }
    }

    #[test]
    fn test_all_zeros() {
        let codes = vec![0u8; 100];
        for b in [1, 2, 3, 4, 8] {
            let packed = pack_codes(&codes, b).unwrap();
            let unpacked = unpack_codes(&packed, 100, b).unwrap();
            assert_eq!(codes, unpacked, "all-zeros roundtrip failed for b={}", b);
        }
    }

    #[test]
    fn test_all_max_values() {
        for b in [1u32, 2, 3, 4, 8] {
            let max_val = ((1u16 << b) - 1) as u8;
            let codes = vec![max_val; 100];
            let packed = pack_codes(&codes, b).unwrap();
            let unpacked = unpack_codes(&packed, 100, b).unwrap();
            assert_eq!(
                codes, unpacked,
                "all-max roundtrip failed for b={}",
                b
            );
        }
    }
}
