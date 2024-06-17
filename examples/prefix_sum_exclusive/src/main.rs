use std::error::Error;
use std::mem;

use empa::adapter::Feature;
use empa::buffer;
use empa::buffer::Buffer;
use empa::device::DeviceDescriptor;
use empa::native::Instance;
use empa_tk::prefix_sum::{PrefixSum, PrefixSumInput};
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

    let count = 1_000_000;

    println!(
        "Evaluating an exclusive prefix-sum over a list of {} `1`s.",
        count
    );

    let mut evaluator = PrefixSum::init_exclusive_u32(device.clone()).await;

    let data: Vec<u32> = vec![1; count];

    let data_buffer: Buffer<[u32], _> =
        device.create_buffer(data, buffer::Usages::storage_binding().and_copy_src());
    let readback_buffer: Buffer<[u32], _> =
        device.create_buffer(vec![0; count], buffer::Usages::map_read().and_copy_dst());
    let timestamp_query_set = device.create_timestamp_query_set(2);
    let timestamps =
        device.create_slice_buffer_zeroed(2, buffer::Usages::query_resolve().and_copy_src());
    let timestamps_readback =
        device.create_slice_buffer_zeroed(2, buffer::Usages::copy_dst().and_map_read());

    let mut encoder = device.create_command_encoder();

    encoder = encoder.write_timestamp(&timestamp_query_set, 0);
    encoder = evaluator.encode(
        encoder,
        PrefixSumInput {
            data: data_buffer.view(),
            count: None,
        },
    );
    encoder = encoder.write_timestamp(&timestamp_query_set, 1);

    encoder = encoder.copy_buffer_to_buffer_slice(data_buffer.view(), readback_buffer.view());
    encoder = encoder.resolve_timestamp_query_set(&timestamp_query_set, 0, timestamps.view());
    encoder = encoder.copy_buffer_to_buffer_slice(timestamps.view(), timestamps_readback.view());

    device.queue().submit(encoder.finish());

    readback_buffer.map_read().await?;

    let data = readback_buffer.mapped();

    println!("The first 10 numbers: {:#?}", &data[..10]);
    println!("The last 10 numbers: {:#?}", &data[data.len() - 10..]);

    println!("Asserting the values computed on the GPU match the expected values...");

    for i in 0..count {
        assert_eq!(data[i], i as u32);
    }

    println!("...successfully!");

    mem::drop(data);

    readback_buffer.unmap();

    timestamps_readback.map_read().await?;

    let timestamps = timestamps_readback.mapped();
    let time_elapsed = timestamps[1] - timestamps[0];

    println!("Time elapsed: {} nanoseconds", time_elapsed);

    mem::drop(timestamps);

    timestamps_readback.unmap();

    Ok(())
}
