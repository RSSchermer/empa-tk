@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read> gather_by: array<BY_TYPE>;

@group(0) @binding(2)
var<storage, read> data_in: array<VALUE_TYPE>;

@group(0) @binding(3)
var<storage, read_write> data_out: array<VALUE_TYPE>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index < count {
        data_out[index] = data_in[gather_by[index]];
    }
}
