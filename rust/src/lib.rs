pub mod base;


#[cfg(test)]
mod tests {
    use crate::base::{bytes_to_int32, bytes_to_string, int32_to_bytes, string_to_bytes};

    #[test]
    fn test_int_conversions() {
        let expected: i32 = 17;
        let actual = bytes_to_int32(&int32_to_bytes(expected));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_string_conversions() {
        let expected = "ugabuga";
        let actual = bytes_to_string(&string_to_bytes(expected));
        assert_eq!(actual, expected);
    }

    // #[test]
    // fn it_works() {
    //     use memmap::MmapOptions;
    //     use std::io::Write;
    //     use std::fs::File;
    //     use std::str::from_utf8;
    //     use byteorder::{BigEndian, LittleEndian, ReadBytesExt}; // 1.2.7
    //     let file = File::open("/home/hvrigazov/experiments/list_of_strings/data.ninja").unwrap();
    //     let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    //     let bytes = &mmap[5..11];
    //     let num = bytes_to_string(bytes).unwrap();
    //     // let mut buf = &mmap[12..16];
    //     // let num = buf.read_i32::<LittleEndian>().unwrap();
    //     println!("{}", num);
    //
    //     let result = 2 + 2;
    //     assert_eq!(result, 4);
    // }
}
