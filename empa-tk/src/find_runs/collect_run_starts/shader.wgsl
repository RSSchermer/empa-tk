@group(0) @binding(0)
var<storage, read> temporary_storage: array<u32>;

@group(0) @binding(1)
var<storage, read_write> run_starts: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= arrayLength(&temporary_storage) {
        return;
    }

    if index == 0 || temporary_storage[index] != temporary_storage[index - 1] {
        let run_index = temporary_storage[index];

        run_starts[run_index] = index;
    }
}
