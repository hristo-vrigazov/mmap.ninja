use byteorder::LittleEndian;
use std::primitive::str;
use byteorder::ReadBytesExt;
use byteorder::WriteBytesExt;
use std::fs;

// TODO: support all parameters as in the Python library!
pub fn bytes_to_i32(mut inp: &[u8]) -> i32 {
    inp.read_i32::<LittleEndian>().expect("Could not convert bytes to int32, corruption?")
}

pub fn i32_to_bytes(inp: i32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.write_i32::<LittleEndian>(inp).expect("Could not convert in32 to bytes!");
    bytes
}

pub fn bytes_to_str(inp: &[u8]) -> &str {
    std::str::from_utf8(inp).expect("Could not convert bytes into string!")
}

pub fn str_to_bytes(inp: &str) -> &[u8] {
    inp.as_bytes()
}

pub fn i32_to_file(inp: i32, filename: &str) {
    let bytes = i32_to_bytes(inp);
    fs::write(filename, bytes).expect("Unable to write file!");
}

pub fn file_to_i32(filename: &str) -> i32 {
    let bytes = fs::read(filename).expect("Unable to read file!");
    bytes_to_i32(&bytes)
}