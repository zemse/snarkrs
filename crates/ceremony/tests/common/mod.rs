/// `01 02 .. 20`, the beacon the ceremony suites use.
pub const BEACON_HEX: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";

pub fn beacon_bytes() -> Vec<u8> {
    (0..BEACON_HEX.len() / 2)
        .map(|i| u8::from_str_radix(&BEACON_HEX[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::beacon_bytes;

    #[test]
    fn beacon_bytes_matches_pinned_fixture() {
        let bytes = beacon_bytes();
        assert_eq!(bytes.len(), 32);
        assert_eq!(
            bytes,
            [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
                0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
                0x1d, 0x1e, 0x1f, 0x20,
            ]
        );
    }
}
