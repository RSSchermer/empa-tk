struct DispatchWorkgroups {
    x: u32,
    y: u32,
    z: u32
}

@group(0) @binding(0)
var<uniform> group_size: u32;

@group(0) @binding(1)
var<uniform> count: u32;

@group(0) @binding(2)
var<storage, read_write> dispatch: DispatchWorkgroups;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let workgroups = div_ceil(count, group_size);

    dispatch = DispatchWorkgroups(workgroups, 1, 1);
}
