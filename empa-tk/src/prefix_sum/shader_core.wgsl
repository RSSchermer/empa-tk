// Warning: this algorithm relies on a "weak OBE" forward progress model, as described by Sorensen et al. 2021
// "Specifying and Testing GPU Workgroup Progress Models" (https://arxiv.org/pdf/2109.06132.pdf). This model
// is not guarenteed by WebGPU or any of the underlying API's (Vulkan, D3D, Metal). However, in practice this
// model seems to describe current GPU behavior, and I am not aware of any current counter examples. I have
// however not done compresensive testing of the complete landscape of GPUs that support WebGPU. The lack of
// spec guarantees also means that future GPUs may not exhibit this behavior. While I certainly don't think
// any GPU vendors will take this library into account when designing future hardware, this is also the forward
// progress model on which Unreal Engine 5's Nanite relies, which is something that I suspect GPU vendors will
// take into consideration. Personally, I am willing to bet on this unofficial forward progress model, at least
// until I come across a counter example, however, if you are unwilling to make this same bet, then consider
// using a multi-pass prefix sum algorithm instead.

const GROUP_SIZE = 256u;
const VALUES_PER_THREAD = 8u;
const SEGMENT_SIZE = 2048u; // GROUP_SIZE * VALUES_PER_THREAD;

const GROUP_STATUS_X = 0u;
const GROUP_STATUS_A = 1u;
const GROUP_STATUS_P = 2u;

struct GroupState {
    // We want to store a status flag and a 32 bit payload, where the status flag indicates the meaning of the payload.
    // Traditionally, the flag and payload would be stored at separate addresses: the payload is stored first, then
    // the flag is stored with "release" memory ordering sementics; when reading back the flag is read first with
    // "acquire" memory ordering semantics. However, WGSL only supports "relaxed" memory ordering semantics for
    // atomic operations (Metal has this restriction). Therefor, what we do instead is split the payload in two (into
    // two 16 bit parts). We still use 2 addresses, but each address stores both the status flag, and a payload part.
    // The memory ordering when writing to these 2 adresses does not matter, nor does the memory order matter when
    // reading from those 2 adresses; all that matters is that when reading, the payload may only be considered
    // reconstructable, if the status flag bits for both adresses match. This then leads to a correctly reconstructed
    // payload, with a correct status interpretation. Note that this only behaves correctly because the algorithm
    // changes to a particular group status once, and then never changes back to that status later for any particular
    // group (the status only changes from e.g. from `P` to `A`, but will never change from `P` to `A`, and then back
    // to `P` again).
    state_0: atomic<u32>,
    state_1: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read_write> data: array<DATA_TYPE>;

@group(0) @binding(2)
var<storage, read_write> group_state: array<GroupState>;

@group(0) @binding(3)
var<storage, read_write> group_counter: atomic<u32>;

var<workgroup> local_data: array<DATA_TYPE, SEGMENT_SIZE>;

var<workgroup> group_index: u32;

var<workgroup> prefix: DATA_TYPE;

var<workgroup> done: bool;

fn write_group_state(group_index: u32, status: u32, payload: DATA_TYPE) {
    let status_bits = status << 30;

    let payload_u32 = bitcast<u32>(payload);
    let payload_top = payload_u32 >> 16;
    let payload_bottom = payload_u32 & 0xFFFF;

    let state_0 = status_bits | payload_top;
    let state_1 = status_bits | payload_bottom;

    atomicStore(&group_state[group_index].state_0, state_0);
    atomicStore(&group_state[group_index].state_1, state_1);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) local_index: u32) {
    if local_index == 0 {
        group_index = atomicAdd(&group_counter, 1u);
    }

    workgroupBarrier();

    let offset = group_index * SEGMENT_SIZE;

    for (var i = local_index; i < SEGMENT_SIZE; i += GROUP_SIZE) {
        let global_index = offset + i;

        if global_index < count {
            local_data[i] = data[global_index];
        }
    }

    workgroupBarrier();

    for (var i = 1u; i < SEGMENT_SIZE; i <<= 1u) {
        var values: array<u32, VALUES_PER_THREAD>;

        for (var j = 0u; j < VALUES_PER_THREAD; j += 1u) {
            let index = j * GROUP_SIZE + local_index;

            if (index >= i) {
                values[j] = local_data[index] + local_data[index - i];
            } else {
                values[j] = local_data[index];
            }
        }

        workgroupBarrier();

        for (var j = 0u; j < VALUES_PER_THREAD; j += 1u) {
            let index = j * GROUP_SIZE + local_index;

            local_data[index] = values[j];
        }

        workgroupBarrier();
    }

    if local_index == 0 {
        let status = select(GROUP_STATUS_A, GROUP_STATUS_P, group_index == 0);
        let aggregate = local_data[SEGMENT_SIZE - 1];

        write_group_state(group_index, status, aggregate);

        if group_index != 0 {
            var target_group_index = group_index - 1;

            loop {
                var target_state_0 = atomicLoad(&group_state[target_group_index].state_0);
                var target_state_1 = atomicLoad(&group_state[target_group_index].state_1);

                var target_status = GROUP_STATUS_X;
                var target_payload = 0u;

                while target_status == GROUP_STATUS_X {
                    let target_status_0 = target_state_0 >> 30;
                    let target_status_1 = target_state_1 >> 30;

                    if target_status_0 == GROUP_STATUS_X || target_status_0 != target_status_1 {
                        target_state_0 = atomicLoad(&group_state[target_group_index].state_0);
                        target_state_1 = atomicLoad(&group_state[target_group_index].state_1);
                    } else {
                        target_status = target_status_0;

                        let target_payload_top = target_state_0 << 16;
                        let target_payload_bottom = target_state_1 & 0xFFFF;

                        target_payload = target_payload_top | target_payload_bottom;
                    }
                }

                let additional_prefix = bitcast<DATA_TYPE>(target_payload);

                prefix += additional_prefix;

                if target_status == GROUP_STATUS_A {
                    target_group_index -= 1u;
                } else if target_status == GROUP_STATUS_P {
                    write_group_state(group_index, GROUP_STATUS_P, prefix + aggregate);

                    break;
                }
            }
        }
    }

    workgroupBarrier();

    for (var i = local_index; i < SEGMENT_SIZE; i += GROUP_SIZE) {
        let global_index = offset + i;

        if global_index < count {
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
