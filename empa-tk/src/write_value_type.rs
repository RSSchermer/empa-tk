use std::fmt::Write;
use std::mem;
use std::ops::Rem;

pub fn write_value_type<V>(s: &mut String) {
    let size = mem::size_of::<V>();

    if size.rem(4) != 0 {
        panic!("Expected an `abi::Sized` type's size to be a multiple of 4")
    }

    write!(s, "struct VALUE_TYPE {{").unwrap();

    let field_count = size / 4;

    for i in 0..field_count {
        write!(s, "    field_{}: u32,\n", i).unwrap();
    }

    write!(s, "}}\n\n").unwrap();
}
