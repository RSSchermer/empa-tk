use std::error::Error;
use std::mem;

use empa::adapter::Feature;
use empa::buffer;
use empa::buffer::Buffer;
use empa::device::DeviceDescriptor;
use empa::native::Instance;
use empa_tk::find_runs::{FindRuns, FindRunsInput, FindRunsOutput};
use futures::FutureExt;

fn main() {
    pollster::block_on(run().map(|res| res.unwrap()));
}

async fn run() -> Result<(), Box<dyn Error>> {
    let instance = Instance::default();
    let adapter = instance.get_adapter(Default::default())?;
    let device = adapter
        .request_device(&DeviceDescriptor {
            required_features: Feature::TimestampQuery | Feature::TimestampQueryInsideEncoders,
            required_limits: Default::default(),
        })
        .await?;

    let counts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000];
    let total = counts.iter().fold(0, |a, b| a + b) as usize;

    let mut data: Vec<u32> = Vec::with_capacity(total);

    for count in counts.iter().copied() {
        for _ in 0..count {
            data.push(count);
        }
    }

    println!("Finding the offset of 'runs' of the same number within a list of numbers.");

    let mut find_runs = FindRuns::init_u32(device.clone()).await;

    let data_buffer: Buffer<[u32], _> =
        device.create_buffer(data, buffer::Usages::storage_binding().and_copy_src());

    let run_count_buffer: Buffer<u32, _> =
        device.create_buffer_zeroed(buffer::Usages::storage_binding().and_copy_src());
    let run_starts_buffer: Buffer<[u32], _> =
        device.create_slice_buffer_zeroed(total, buffer::Usages::storage_binding().and_copy_src());
    let run_mapping_buffer: Buffer<[u32], _> = device.create_slice_buffer_zeroed(
        total,
        buffer::Usages::storage_binding()
            .and_copy_dst()
            .and_copy_src(),
    );

    let run_count_readback_buffer: Buffer<u32, _> =
        device.create_buffer_zeroed(buffer::Usages::map_read().and_copy_dst());
    let run_starts_readback_buffer: Buffer<[u32], _> =
        device.create_slice_buffer_zeroed(total, buffer::Usages::map_read().and_copy_dst());

    let timestamp_query_set = device.create_timestamp_query_set(2);
    let timestamps =
        device.create_slice_buffer_zeroed(2, buffer::Usages::query_resolve().and_copy_src());
    let timestamps_readback =
        device.create_slice_buffer_zeroed(2, buffer::Usages::copy_dst().and_map_read());

    let mut encoder = device.create_command_encoder();

    encoder = encoder.write_timestamp(&timestamp_query_set, 0);
    encoder = find_runs.encode(
        encoder,
        FindRunsInput {
            data: data_buffer.view(),
            count: None,
        },
        FindRunsOutput {
            run_count: run_count_buffer.view(),
            run_starts: run_starts_buffer.view(),
            run_mapping: run_mapping_buffer.view(),
        },
    );
    encoder = encoder.write_timestamp(&timestamp_query_set, 1);

    encoder = encoder
        .copy_buffer_to_buffer_slice(run_starts_buffer.view(), run_starts_readback_buffer.view());
    encoder =
        encoder.copy_buffer_to_buffer(run_count_buffer.view(), run_count_readback_buffer.view());
    encoder = encoder.resolve_timestamp_query_set(&timestamp_query_set, 0, timestamps.view());
    encoder = encoder.copy_buffer_to_buffer_slice(timestamps.view(), timestamps_readback.view());

    device.queue().submit(encoder.finish());

    run_count_readback_buffer.map_read().await?;

    let run_count = *run_count_readback_buffer.mapped() as usize;

    run_count_readback_buffer.unmap();

    run_starts_readback_buffer.map_read().await?;

    let run_starts = run_starts_readback_buffer.mapped();

    println!(
        "The starts of the runs of numbers found on the GPU: {:#?}",
        &run_starts[..run_count]
    );

    println!("Asserting the number of runs found matches the expected number of runs...");

    assert_eq!(run_count, counts.len());

    println!("...successfully!");

    println!("Asserting the run offsets computed on the GPU match the expected offsets...");

    let mut expected_values = Vec::with_capacity(counts.len());

    let mut s = 0;

    for count in counts.iter().copied() {
        expected_values.push(s);

        s += count;
    }

    for i in 0..run_count {
        assert_eq!(run_starts[i], expected_values[i]);
    }

    println!("...successfully!");

    mem::drop(run_starts);

    run_starts_readback_buffer.unmap();

    timestamps_readback.map_read().await?;

    let timestamps = timestamps_readback.mapped();
    let time_elapsed = timestamps[1] - timestamps[0];

    println!("Time elapsed: {} nanoseconds", time_elapsed);

    mem::drop(timestamps);

    timestamps_readback.unmap();

    Ok(())
}
