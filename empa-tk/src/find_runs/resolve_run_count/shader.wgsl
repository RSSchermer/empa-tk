@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read> temporary_storage: array<u32>;

@group(0) @binding(2)
var<storage, read_write> run_count: u32;

@compute @workgroup_size(1, 1, 1)
fn main() {
    run_count = temporary_storage[count - 1] + 1;
}
