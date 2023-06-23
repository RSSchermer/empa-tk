@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read> data: array<DATA_TYPE>;

@group(0) @binding(2)
var<storage, read_write> temporary_storage: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index != 0 && index < count {
        if data[index] != data[index - 1] {
            temporary_storage[index] = 1;
        }
    }
}
