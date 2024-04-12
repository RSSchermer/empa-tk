const RADIX_SIZE = 8u;
const GROUP_SIZE = 256u;
const GROUP_ITERATIONS = 4u;

const RADIX_DIGITS = 256u;//1 << RADIX_SIZE;
const SEGMENT_SIZE = 1024;//GROUP_SIZE * GROUP_ITERATIONS;
const RADIX_GROUPS = 4u;//32 / RADIX_SIZE;

@group(0) @binding(0)
var<uniform> max_count: u32;

@group(0) @binding(1)
var<storage, read> data: array<u32>;

@group(0) @binding(2)
var<storage, read_write> global_histograms: array<array<atomic<u32>, RADIX_DIGITS>, RADIX_GROUPS>;

var<workgroup> local_histograms: array<array<atomic<u32>, RADIX_DIGITS>, RADIX_GROUPS>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
    let group_index = workgroup_id.x;
    let count = max(max_count, arrayLength(&data));

    let segment_offset = group_index * SEGMENT_SIZE;

    for (var i = local_index; i < SEGMENT_SIZE; i += GROUP_SIZE) {
        let data_index = segment_offset + i;

        if data_index < count {
            let value = data[data_index];

            for (var j = 0u; j < RADIX_GROUPS; j++) {
                let digits = (value >> (j * RADIX_SIZE)) & (RADIX_DIGITS - 1);

                atomicAdd(&local_histograms[j][digits], 1);
            }
        }
    }

    workgroupBarrier();

    for (var i = local_index; i < RADIX_DIGITS; i += GROUP_SIZE) {
        for (var j = 0u; j < RADIX_GROUPS; j++) {
            let local_bucket_count = atomicLoad(&local_histograms[j][i]);

            if local_bucket_count > 0 {
                atomicAdd(&global_histograms[j][i], local_bucket_count);
            }
        }
    }
}
