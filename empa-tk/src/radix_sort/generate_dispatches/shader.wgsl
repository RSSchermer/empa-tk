struct DispatchWorkgroups {
    x: u32,
    y: u32,
    z: u32
}

struct SegmentSizes {
    histogram: u32,
    scatter: u32,
}

@group(0) @binding(0)
var<uniform> segment_sizes: SegmentSizes;

@group(0) @binding(1)
var<uniform> max_count: u32;

@group(0) @binding(2)
var<storage, read> data: array<u32>;

@group(0) @binding(3)
var<storage, read_write> histogram_dispatch: DispatchWorkgroups;

@group(0) @binding(4)
var<storage, read_write> scatter_dispatch: DispatchWorkgroups;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let count = max(max_count, arrayLength(&data));

    let histogram_workgroups = div_ceil(count, segment_sizes.histogram);

    histogram_dispatch = DispatchWorkgroups(histogram_workgroups, 1, 1);

    let scatter_workgroups = div_ceil(count, segment_sizes.scatter);

    scatter_dispatch = DispatchWorkgroups(scatter_workgroups, 1, 1);
}
