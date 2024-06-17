const RADIX_SIZE = 8u;
const GROUP_SIZE = 256u; // Must be >= 2^RADIX_SIZE

const RADIX_DIGITS = 256u; //1 << RADIX_SIZE;
const RADIX_GROUPS = 4u; //32 / RADIX_SIZE;

@group(0) @binding(0)
var<storage, read_write> global_data: array<array<u32, RADIX_DIGITS>, RADIX_GROUPS>;

var<workgroup> local_data: array<u32, GROUP_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
    let group_index = workgroup_id.x;

    if local_index < RADIX_DIGITS {
        local_data[local_index] = global_data[group_index][local_index];
    }

    workgroupBarrier();

    for (var i = 1u; i < GROUP_SIZE; i <<= 1u) {
        var value: u32;

        if (local_index >= i) {
            value = local_data[local_index] + local_data[local_index - i];
        } else {
            value = local_data[local_index];
        }

        workgroupBarrier();

        local_data[local_index] = value;

        workgroupBarrier();
    }

    workgroupBarrier();

    // Note: our prefix sum is currently inclusive, but needs to be exclusive, so we shift the array to the right. The
    // first element in the global_data output array should already be 0.
    var output = 0u;

    if local_index != 0 {
        output = local_data[local_index - 1];
    }

    if local_index < RADIX_DIGITS {
        global_data[group_index][local_index] = output;
    }
}
