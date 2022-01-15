pub mod base;
pub mod numpy;


#[cfg(test)]
mod tests {
    use crate::base::{bytes_to_i32, bytes_to_shape, bytes_to_str, file_to_i32, file_to_string, i32_to_bytes, i32_to_file, shape_to_bytes, str_to_bytes, str_to_file};

    #[test]
    fn test_int_conversions() {
        let expected: i32 = 17;
        let actual = bytes_to_i32(&i32_to_bytes(expected));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_string_conversions() {
        let expected = "ugabuga";
        let actual = bytes_to_str(&str_to_bytes(expected));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_int_file_conversions() {
        let expected: i32 = 17;
        let filename = "./test_rust";
        i32_to_file(expected, filename);
        let actual = file_to_i32(filename);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_string_file_conversions() {
        let expected = "ugabuga";
        let filename = "./test_rust2";
        str_to_file(expected, filename);
        let actual = file_to_string(filename);
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_shape_conversions() {
        let shape = vec![1, 9, 10, 1];
        let actual = bytes_to_shape(&shape_to_bytes(&shape));
        assert_eq!(actual, shape);
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
