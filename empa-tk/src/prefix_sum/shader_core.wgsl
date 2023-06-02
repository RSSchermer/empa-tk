const SEGMENT_SIZE = 512u;
const GROUP_SIZE = 256u;

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

    // The "up-sweep" phase of the Blelloch algorithm. We implement an unrolled version  of the algorithm for the
    // specific workgroup size of `256`.
    //
    // We can do the first step before the first barrier, as in this step each thread only touches the 2 values it just
    // loaded itself. We then synchronize between each step. Every thread on the group participates in the first step,
    // then for each subsequent step the number of participating threads halves.

    shared_data[local_index * 2 + 1] += shared_data[local_index * 2];
    workgroupBarrier();

    if local_index < 128 {
        shared_data[local_index * 4 + 3] += shared_data[local_index * 4 + 1];
    }
    workgroupBarrier();

    if local_index < 64 {
        shared_data[local_index * 8 + 7] += shared_data[local_index * 8 + 3];
    }
    workgroupBarrier();

    if local_index < 32 {
        shared_data[local_index * 16 + 15] += shared_data[local_index * 16 + 7];
    }
    workgroupBarrier();

    if local_index < 16 {
        shared_data[local_index * 32 + 31] += shared_data[local_index * 32 + 15];
    }
    workgroupBarrier();

    if local_index < 8 {
        shared_data[local_index * 64 + 63] += shared_data[local_index * 64 + 31];
    }
    workgroupBarrier();

    if local_index < 4 {
        shared_data[local_index * 128 + 127] += shared_data[local_index * 128 + 63];
    }
    workgroupBarrier();

    if local_index < 2 {
        shared_data[local_index * 256 + 255] += shared_data[local_index * 256 + 127];
    }
    workgroupBarrier();

    // We skip the final up-sweep step, because this step only sets the last element in the shared array, which would
    // be set to `0` in the "down-sweep" phase.

    // The "down-sweep" phase of the Blelloch algorithm. We again implement an unrolled version of the algorithm for the
    // specific workgroup size of `256`, synchronizing between each step. Only one thread participates on the first
    // step, then for each subsequent step the number of threads that participate doubles, until all threads participate
    // in the final step.

    var status = 0u;
    var aggregate: DATA_TYPE;

    if local_index == 0 {
        // We can compute the aggregate of all elements in this workgroup's data segment now.
        aggregate = shared_data[511] + shared_data[255];

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

        shared_data[511] = shared_data[255];
        shared_data[255] = 0;
    }
    workgroupBarrier();

    if local_index < 2 {
        let v = shared_data[local_index * 256 + 255];

        shared_data[local_index * 256 + 255] += shared_data[local_index * 256 + 127];
        shared_data[local_index * 256 + 127] = v;
    }
    workgroupBarrier();

    if local_index < 4 {
        let v = shared_data[local_index * 128 + 127];

        shared_data[local_index * 128 + 127] += shared_data[local_index * 128 + 63];
        shared_data[local_index * 128 + 63] = v;
    }
    workgroupBarrier();

    if local_index < 8 {
        let v = shared_data[local_index * 64 + 63];

        shared_data[local_index * 64 + 63] += shared_data[local_index * 64 + 31];
        shared_data[local_index * 64 + 31] = v;
    }
    workgroupBarrier();

    if local_index < 16 {
        let v = shared_data[local_index * 32 + 31];

        shared_data[local_index * 32 + 31] += shared_data[local_index * 32 + 15];
        shared_data[local_index * 32 + 15] = v;
    }
    workgroupBarrier();

    if local_index < 32 {
        let v = shared_data[local_index * 16 + 15];

        shared_data[local_index * 16 + 15] += shared_data[local_index * 16 + 7];
        shared_data[local_index * 16 + 7] = v;
    }
    workgroupBarrier();

    if local_index < 64 {
        let v = shared_data[local_index * 8 + 7];

        shared_data[local_index * 8 + 7] += shared_data[local_index * 8 + 3];
        shared_data[local_index * 8 + 3] = v;
    }
    workgroupBarrier();

    if local_index < 128 {
        let v = shared_data[local_index * 4 + 3];

        shared_data[local_index * 4 + 3] += shared_data[local_index * 4 + 1];
        shared_data[local_index * 4 + 1] = v;
    }
    workgroupBarrier();

    let v = shared_data[local_index * 2 + 1];

    shared_data[local_index * 2 + 1] += shared_data[local_index * 2];
    shared_data[local_index * 2] = v;

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
