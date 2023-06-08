const GROUP_SIZE = 256u;
const RADIX_SIZE = 8u;

const RADIX_DIGITS = 256u;//1 << RADIX_SIZE;
const RADIX_GROUPS = 4u;//32 / RADIX_SIZE;

const BUCKET_STATUS_NOT_READY = 0u;
const BUCKET_STATUS_LOCAL_OFFSET = 1u;
const BUCKET_STATUS_GLOBAL_OFFSET = 2u;

struct Uniforms {
    radix_offset: u32,
    radix_group: u32
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> data_in: array<u32>;

@group(0) @binding(2)
var<storage, read_write> data_out: array<u32>;

@group(0) @binding(3)
var<storage, read> global_base_bucket_offsets: array<array<u32, RADIX_DIGITS>, RADIX_GROUPS>;

@group(0) @binding(4)
var<storage, read_write> group_state: array<array<atomic<u32>, RADIX_DIGITS>>;

@group(0) @binding(5)
var<storage, read_write> group_counter: atomic<u32>;

@group(0) @binding(6)
var<storage, read_write> debug: array<u32>;

var<workgroup> segment_index: u32;

var<workgroup> local_data: array<u32, GROUP_SIZE>;

var<workgroup> workspace: array<u32, GROUP_SIZE>;

fn extract_radix_digits(value: u32) -> u32 {
    return (value >> uniforms.radix_offset) & (RADIX_DIGITS - 1);
}

fn workspace_prefix_sum_inclusive(local_index: u32) {
    // Hillis-Steele style prefix sum over the workspace
    for (var i = 1u; i < GROUP_SIZE; i <<= 1) {
        var value: u32;

        if (local_index >= i) {
            value = workspace[local_index] + workspace[local_index - i];
        } else {
            value = workspace[local_index];
        }

        workgroupBarrier();

        workspace[local_index] = value;

        workgroupBarrier();
    }
}

fn sort_local_data(local_index: u32) {
    for (var b = 0u; b < RADIX_SIZE; b++) {
        let bit_offset = uniforms.radix_offset + b;

        if local_index == 0 {
            workspace[0] = 0;
        } else {
            let bit_value_prev = (local_data[local_index - 1] >> bit_offset) & 1;

            workspace[local_index] = u32(bit_value_prev == 0);
        }

        workgroupBarrier();

        workspace_prefix_sum_inclusive(local_index);

        let bit_value = (local_data[local_index] >> bit_offset) & 1;
        let last_bit_value = (local_data[GROUP_SIZE - 1] >> bit_offset) & 1;
        let total_false_count = u32(last_bit_value == 0) + workspace[GROUP_SIZE - 1];

        var output_index: u32;

        if bit_value == 0 {
            output_index = workspace[local_index];
        } else {
            output_index = total_false_count + local_index - workspace[local_index];
        }

        // Move the local_data value to its new position. First let all threads read their current into `function`
        // memory, wait for all threads to be done reading, then all threads move their value to the new position.
        let value = local_data[local_index];

        workgroupBarrier();

        local_data[output_index] = value;

        workgroupBarrier();
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) local_index: u32) {
    if local_index == 0 {
        segment_index = atomicAdd(&group_counter, 1);
    }

    let uniform_segment_index = workgroupUniformLoad(&segment_index);
    let segment_offset = uniform_segment_index * GROUP_SIZE;

    if segment_offset >= arrayLength(&data_in) {
        return;
    }

    let data_size = min(GROUP_SIZE, arrayLength(&data_in) - segment_offset);

    if local_index < data_size {
        local_data[local_index] = data_in[segment_offset + local_index];
    } else {
        local_data[local_index] = 0xFFFFFFFFu;
    }

    workgroupBarrier();

    sort_local_data(local_index);

    // Now find "runs" of the same key in the sorted local data, mark the start of runs with `1` in the workspace
    // array, otherwise set to `0`.
    let current_radix = extract_radix_digits(local_data[local_index]);
    let prev_radix = extract_radix_digits(local_data[local_index - 1]);

    let is_run_start = local_index == 0 || current_radix != prev_radix;

    if local_index != 0 && current_radix != prev_radix {
        workspace[local_index] = 1;
    } else {
        workspace[local_index] = 0;
    }

    workgroupBarrier();

    // An inclusive prefix sum over the workspace will now find the index of the "run" each value belongs to
    workspace_prefix_sum_inclusive(local_index);

    let run_index = workspace[local_index];

    workgroupBarrier();

    // Reuse the workspace again to now store the index at which each "run" starts. Before we store the run start
    // indices, first set all positions to `data_size`. Now, after the run starts are written, the position after each
    // run start holds the run end. We use the difference to compute the bucket sizes.

    workspace[local_index] = data_size;

    workgroupBarrier();

    if is_run_start {
        workspace[run_index] = local_index;
    }

    workgroupBarrier();

    // Lookup the bucket counts and the within-bucket-index for each value. Note that the bucket count will only make
    // sense for threads that represent a "run start"; we'll ignore the bucket count value on all other threads.
    let run_start = workspace[run_index];

    var run_end = data_size;

    if run_index < RADIX_DIGITS - 1 {
        run_end = workspace[run_index + 1];
    }

    let bucket_count = run_end - run_start;
    let within_bucket_index = local_index - run_start;

    // We'll determine the bucket number from radix digits of the value in local_data with which the run start is
    // associated
    let bucket_index = current_radix;

    // We're now ready to communicate the bucket sizes to the other workgroups. We'll reuse the workspace again to
    // store the counts for each bucket.

    workgroupBarrier();

    workspace[local_index] = 0;

    workgroupBarrier();

    if is_run_start {
        workspace[bucket_index] = bucket_count;
    }

    workgroupBarrier();

    let local_bucket_count = workspace[local_index];

    // Initially the bucket state will contain the local offset, unless this is the first segment, in which case
    // it will immediately be the global offset.
    var bucket_status = BUCKET_STATUS_LOCAL_OFFSET;

    if segment_index == 0 {
        bucket_status = BUCKET_STATUS_GLOBAL_OFFSET;
    }

    let broadcast_state = (bucket_status << 30) | local_bucket_count;

    atomicStore(&group_state[segment_index][local_index], broadcast_state);

    var accumulated_prefix = 0u;

    for (var i = i32(segment_index) - 1; i >= 0; i -= 1) {
        var state = 0u;

        while (state >> 30) == BUCKET_STATUS_NOT_READY {
            state = atomicLoad(&group_state[i][local_index]);
        }

        let status = state >> 30;
        let value = state & 0x3FFFFFFF;

        accumulated_prefix += value;

        if status == BUCKET_STATUS_GLOBAL_OFFSET {
            let new_value = accumulated_prefix + local_bucket_count;
            let new_broadcast_state = (BUCKET_STATUS_GLOBAL_OFFSET << 30) | new_value;

            atomicStore(&group_state[segment_index][local_index], new_broadcast_state);

            break;
        }
    }

    workgroupBarrier();

    workspace[local_index] = accumulated_prefix;

    workgroupBarrier();

    let global_bucket_offset = global_base_bucket_offsets[uniforms.radix_group][bucket_index] + workspace[bucket_index];
    let output_index = global_bucket_offset + within_bucket_index;

    debug[local_index] = bucket_index;

    if local_index < data_size {
        data_out[output_index] = local_data[local_index];
    }
}
