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

var<workgroup> local_data: array<DATA_TYPE, SEGMENT_SIZE>;

var<workgroup> group_index: u32;

var<workgroup> prefix: DATA_TYPE;

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
            local_data[i] = data[global_index];
        }
    }

    workgroupBarrier();

    for (var i = 1u; i < SEGMENT_SIZE; i <<= 1) {
        var values: array<u32, VALUES_PER_THREAD>;

        for (var j = 0u; j < VALUES_PER_THREAD; j += 1) {
            let index = j * GROUP_SIZE + local_index;

            if (index >= i) {
                values[j] = local_data[index] + local_data[index - i];
            } else {
                values[j] = local_data[index];
            }
        }

        workgroupBarrier();

        for (var j = 0u; j < VALUES_PER_THREAD; j += 1) {
            let index = j * GROUP_SIZE + local_index;

            local_data[index] = values[j];
        }

        workgroupBarrier();
    }

    var status = 0u;
    var aggregate: DATA_TYPE;

    if local_index == 0 {
        // We can compute the aggregate of all elements in this workgroup's data segment now.
        aggregate = local_data[SEGMENT_SIZE - 1];

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
    }

    let uniform_group_index = workgroupUniformLoad(&group_index);

    if uniform_group_index != 0 {
        var target_group_index = uniform_group_index - 1;

        loop {
            if local_index == 0 {
                let target_status = atomicLoad(&group_state[target_group_index].status);

                if target_status == GROUP_STATUS_A {
                    let additional_prefix = bitcast<DATA_TYPE>(atomicLoad(&group_state[target_group_index].aggregate));

                    prefix += additional_prefix;
                    target_group_index -= 1;
                } else if target_status == GROUP_STATUS_P {
                    let additional_prefix =
                        bitcast<DATA_TYPE>(atomicLoad(&group_state[target_group_index].inclusive_prefix));

                    prefix += additional_prefix;

                    atomicStore(&group_state[group_index].inclusive_prefix, bitcast<u32>(prefix + aggregate));

                    done = true;
                }
            }

            let uniform_done = workgroupUniformLoad(&done);

            if uniform_done {
                storageBarrier();

                if local_index == 0 {
                    atomicStore(&group_state[group_index].status, GROUP_STATUS_P);
                }

                break;
            }
        }
    }

    for (var i = local_index; i < SEGMENT_SIZE; i += GROUP_SIZE) {
        let global_index = offset + i;

        if global_index < arrayLength(&data) {
            if OUTPUT_EXCLUSIVE {
                var output_value = prefix;

                if i > 0 {
                    output_value += local_data[i - 1];
                }

                data[global_index] = output_value;
            } else {
                data[global_index] = prefix + local_data[i];
            }
        }
    }
}
