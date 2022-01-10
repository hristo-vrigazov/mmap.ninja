use byteorder::LittleEndian;
use std::str::{Utf8Error};
use std::primitive::str;
use byteorder::ReadBytesExt;
use byteorder::WriteBytesExt;

// TODO: support all parameters as in the Python library!
pub fn bytes_to_int(mut inp: &[u8]) -> std::io::Result<i32> {
    inp.read_i32::<LittleEndian>()
}

pub fn int_to_bytes(inp: i32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.write_i32::<LittleEndian>(inp).expect("Failed to convert int to bytes!");
    bytes
}

pub fn bytes_to_string(inp: &[u8]) -> Result<&str, Utf8Error> {
    std::str::from_utf8(inp)
}