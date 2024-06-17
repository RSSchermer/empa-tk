use std::error::Error;
use std::mem;

use empa::adapter::Feature;
use empa::buffer;
use empa::buffer::Buffer;
use empa::device::DeviceDescriptor;
use empa::native::Instance;
use empa_tk::radix_sort::{RadixSort, RadixSortInput};
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

    let mut radix_sort = RadixSort::init_u32(device.clone()).await;

    let count = 1_000_000;

    println!("Sorting {} values...", count);

    let mut rng = oorandom::Rand32::new(1);
    let mut data: Vec<u32> = Vec::with_capacity(count);

    for _ in 0..count {
        data.push(rng.rand_u32());
    }

    let data_buffer: Buffer<[u32], _> =
        device.create_buffer(&*data, buffer::Usages::storage_binding().and_copy_src());
    let temp_storage_buffer: Buffer<[u32], _> =
        device.create_slice_buffer_zeroed(count, buffer::Usages::storage_binding().and_copy_src());
    let readback_buffer: Buffer<[u32], _> =
        device.create_buffer(vec![0; count], buffer::Usages::map_read().and_copy_dst());
    let timestamp_query_set = device.create_timestamp_query_set(2);
    let timestamps =
        device.create_slice_buffer_zeroed(2, buffer::Usages::query_resolve().and_copy_src());
    let timestamps_readback =
        device.create_slice_buffer_zeroed(2, buffer::Usages::copy_dst().and_map_read());

    let mut encoder = device.create_command_encoder();

    encoder = encoder.write_timestamp(&timestamp_query_set, 0);
    encoder = radix_sort.encode(
        encoder,
        RadixSortInput {
            data: data_buffer.view(),
            temporary_storage: temp_storage_buffer.view(),
            count: None,
        },
    );
    encoder = encoder.write_timestamp(&timestamp_query_set, 1);

    encoder = encoder.copy_buffer_to_buffer_slice(data_buffer.view(), readback_buffer.view());

    encoder = encoder.resolve_timestamp_query_set(&timestamp_query_set, 0, timestamps.view());
    encoder = encoder.copy_buffer_to_buffer_slice(timestamps.view(), timestamps_readback.view());

    device.queue().submit(encoder.finish());

    data.sort();

    readback_buffer.map_read().await?;

    let readback = readback_buffer.mapped();

    println!(
        "The first 10 numbers computed on the GPU: {:#?}",
        &readback[..10]
    );
    println!(
        "The first 10 numbers computed on the CPU (reference): {:#?}",
        &data[..10]
    );

    println!(
        "The last 10 numbers computed on the GPU: {:#?}",
        &readback[readback.len() - 10..]
    );
    println!(
        "The last 10 numbers computed on the CPU (reference): {:#?}",
        &data[data.len() - 10..]
    );

    println!("Asserting all values produced by the GPU sort match the values produced by the CPU sort...");

    for i in 0..count {
        assert_eq!(&readback[i], &data[i]);
    }

    println!("...successfully!");

    mem::drop(readback);

    readback_buffer.unmap();

    timestamps_readback.map_read().await?;

    let timestamps = timestamps_readback.mapped();
    let gpu_time_elapsed = timestamps[1] - timestamps[0];

    println!("Time elapsed GPU: {} milliseconds", gpu_time_elapsed);

    mem::drop(timestamps);

    timestamps_readback.unmap();

    Ok(())
}
