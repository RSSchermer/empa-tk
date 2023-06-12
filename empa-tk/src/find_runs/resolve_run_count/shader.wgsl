@group(0) @binding(0)
var<storage, read> temporary_storage: array<u32>;

@group(0) @binding(1)
var<storage, read_write> run_count: u32;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    run_count = temporary_storage[arrayLength(&temporary_storage)] + 1;
}
