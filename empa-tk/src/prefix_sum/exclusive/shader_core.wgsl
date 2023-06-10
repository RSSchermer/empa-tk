const GROUP_SIZE = 256u;
const VALUES_PER_THREAD = 8u;

const SEGMENT_SIZE = 2048u; // GROUP_SIZE * VALUES_PER_THREAD;

const GROUP_STATUS_X = 0u;
const GROUP_STATUS_A = 1u;
const GROUP_STATUS_P = 2u;

struct GroupState {
    // Note: aggregate and inclusive_prefix need to be atomic, as global memory in WebGPU's memory model is not
    // coherent. Non-atomic reads therefor risk reading stale values from local (L1) caches. Atomic reads force
    // ignoring any local caching.
    aggregate: atomic<u32>,
    inclusive_prefix: atomic<u32>,
    status: atomic<u32>
}

@group(0) @binding(0)
var<storage, read_write> data: array<DATA_TYPE>;

@group(0) @binding(1)
var<storage, read_write> group_state: array<GroupState>;

@group(0) @binding(2)
var<storage, read_write> group_counter: atomic<u32>;

var<workgroup> shared_data: array<DATA_TYPE, SEGMENT_SIZE>;

var<workgroup> group_index: u32;

var<workgroup> prefix: DATA_TYPE;

var<workgroup> add_prefix: bool;

var<workgroup> done: bool;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) local_index: u32) {
    if local_index == 0 {
        group_index = atomicAdd(&group_counter, 1);
    }

    workgroupBarrier();

    let offset = group_index * SEGMENT_SIZE;

    for (var i = local_index; i < SEGMENT_SIZE; i += GROUP_SIZE) {
        let global_index = offset + i;

        if global_index < arrayLength(&data) {
            shared_data[i] = data[global_index];
        }
    }

    workgroupBarrier();

    for (var i = 1u; i < firstLeadingBit(SEGMENT_SIZE); i += 1) {
        let multiplier = 1u << i;
        let a_offset = multiplier - 1;
        let b_offset = (multiplier >> 1) - 1;

        let threshold = SEGMENT_SIZE >> i;

        for (var j = local_index; j < threshold; j += GROUP_SIZE) {
            let base_offset = j * multiplier;

            shared_data[base_offset + a_offset] += shared_data[base_offset + b_offset];
        }

        workgroupBarrier();
    }

    let last_index = SEGMENT_SIZE - 1;
    let mid_index = (SEGMENT_SIZE >> 1) - 1;

    var status = 0u;
    var aggregate: DATA_TYPE;

    if local_index == 0 {
        // We can compute the aggregate of all elements in this workgroup's data segment now.
        aggregate = shared_data[last_index] + shared_data[mid_index];

        if group_index == 0 {
            atomicStore(&group_state[0].inclusive_prefix, bitcast<u32>(aggregate));

            status = GROUP_STATUS_P;
        } else {
            atomicStore(&group_state[group_index].aggregate, bitcast<u32>(aggregate));

            status = GROUP_STATUS_A;
        }
    }

    storageBarrier();

    if local_index == 0 {
        atomicStore(&group_state[group_index].status, status);

        shared_data[last_index] = shared_data[mid_index];
        shared_data[mid_index] = 0;
    }
    workgroupBarrier();

    for (var i = 1u; i < firstLeadingBit(SEGMENT_SIZE); i += 1) {
        let multiplier = SEGMENT_SIZE >> i;
        let a_offset = multiplier - 1;
        let b_offset = (multiplier >> 1) - 1;

        let threshold = 1u << i;

        for (var j = local_index; j < threshold; j += GROUP_SIZE) {
            let base_offset = j * multiplier;

            let a_index = base_offset + a_offset;
            let b_index = base_offset + b_offset;

            let v = shared_data[a_index];

            shared_data[a_index] += shared_data[b_index];
            shared_data[b_index] = v;
        }

        workgroupBarrier();
    }

    let uniform_group_index = workgroupUniformLoad(&group_index);

    if uniform_group_index != 0 {
        var target_group_index = uniform_group_index - 1;

        loop {
            if local_index == 0 {
                let target_status = atomicLoad(&group_state[target_group_index].status);

                if target_status == GROUP_STATUS_A {
                    let additional_prefix = bitcast<DATA_TYPE>(atomicLoad(&group_state[target_group_index].aggregate));

                    prefix = additional_prefix;
                    aggregate += additional_prefix;
                    add_prefix = true;
                    target_group_index -= 1;
                } else if target_status == GROUP_STATUS_P {
                    let additional_prefix =
                        bitcast<DATA_TYPE>(atomicLoad(&group_state[target_group_index].inclusive_prefix));

                    prefix = additional_prefix;
                    aggregate += additional_prefix;
                    add_prefix = true;
                    done = true;
                } else {
                    add_prefix = false;
                }

                if done {
                    atomicStore(&group_state[group_index].inclusive_prefix, bitcast<u32>(aggregate));
                }
            }

            let uniform_done = workgroupUniformLoad(&done);

            if uniform_done {
                storageBarrier();

                if local_index == 0 {
                    atomicStore(&group_state[group_index].status, GROUP_STATUS_P);
                }
            }

            if add_prefix {
                for (var i = local_index; i < SEGMENT_SIZE; i += GROUP_SIZE) {
                    shared_data[i] += prefix;
                }
            }

            workgroupBarrier();

            if uniform_done {
                break;
            }
        }
    }

    for (var i = local_index; i < SEGMENT_SIZE; i += GROUP_SIZE) {
        let global_index = offset + i;

        if global_index < arrayLength(&data) {
            data[global_index] = shared_data[i];
        }
    }
}
