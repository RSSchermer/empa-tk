use std::error::Error;
use std::mem;

use bytemuck::Zeroable;
use empa::adapter::Feature;
use empa::buffer::Buffer;
use empa::device::DeviceDescriptor;
use empa::native::Instance;
use empa::{abi, buffer};
use empa_tk::radix_sort::{RadixSortBy, RadixSortByInput};
use futures::FutureExt;

#[derive(abi::Sized, Clone, Copy, PartialEq, Default, Debug, Zeroable)]
#[repr(C)]
struct MyValue {
    field_a: u32,
    field_b: f32,
}

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

    let mut radix_sort_by = RadixSortBy::init_u32(device.clone()).await;

    let count = 1_000_000;

    println!("Sorting {} values by their keys...", count);

    let mut rng = oorandom::Rand32::new(1);
    let mut keys: Vec<u32> = Vec::with_capacity(count);
    let mut values: Vec<MyValue> = Vec::with_capacity(count);

    for i in 0..count {
        keys.push(rng.rand_u32());
        values.push(MyValue {
            field_a: i as u32,
            field_b: i as f32,
        });
    }

    let keys_buffer: Buffer<[u32], _> =
        device.create_buffer(&*keys, buffer::Usages::storage_binding().and_copy_src());
    let temp_key_storage_buffer: Buffer<[u32], _> =
        device.create_slice_buffer_zeroed(count, buffer::Usages::storage_binding().and_copy_src());
    let values_buffer: Buffer<[MyValue], _> =
        device.create_buffer(&*values, buffer::Usages::storage_binding().and_copy_src());
    let temp_value_storage_buffer: Buffer<[MyValue], _> =
        device.create_slice_buffer_zeroed(count, buffer::Usages::storage_binding().and_copy_src());

    let value_readback_buffer: Buffer<[MyValue], _> =
        device.create_slice_buffer_zeroed(count, buffer::Usages::map_read().and_copy_dst());
    let timestamp_query_set = device.create_timestamp_query_set(2);
    let timestamps =
        device.create_slice_buffer_zeroed(2, buffer::Usages::query_resolve().and_copy_src());
    let timestamps_readback =
        device.create_slice_buffer_zeroed(2, buffer::Usages::copy_dst().and_map_read());

    let mut encoder = device.create_command_encoder();

    encoder = encoder.write_timestamp(&timestamp_query_set, 0);
    encoder = radix_sort_by.encode(
        encoder,
        RadixSortByInput {
            keys: keys_buffer.view(),
            values: values_buffer.view(),
            temporary_key_storage: temp_key_storage_buffer.view(),
            temporary_value_storage: temp_value_storage_buffer.view(),
            count: None,
        },
    );
    encoder = encoder.write_timestamp(&timestamp_query_set, 1);

    encoder =
        encoder.copy_buffer_to_buffer_slice(values_buffer.view(), value_readback_buffer.view());

    encoder = encoder.resolve_timestamp_query_set(&timestamp_query_set, 0, timestamps.view());
    encoder = encoder.copy_buffer_to_buffer_slice(timestamps.view(), timestamps_readback.view());

    device.queue().submit(encoder.finish());

    let mut permutation = permutation::sort(&keys);

    permutation.apply_slice_in_place(&mut keys);
    permutation.apply_slice_in_place(&mut values);

    value_readback_buffer.map_read().await?;

    let values_readback = value_readback_buffer.mapped();

    println!(
        "The first 10 values computed on the GPU: {:#?}",
        &values_readback[..10]
    );
    println!(
        "The first 10 values computed on the CPU (reference): {:#?}",
        &values[..10]
    );

    println!(
        "The last 10 values computed on the GPU: {:#?}",
        &values_readback[values_readback.len() - 10..]
    );
    println!(
        "The last 10 values computed on the CPU (reference): {:#?}",
        &values[values.len() - 10..]
    );

    println!("Asserting all values produced by the GPU sort match the values produced by the CPU sort...");

    for i in 0..count {
        assert_eq!(&values_readback[i], &values[i]);
    }

    println!("...successfully!");

    mem::drop(values_readback);

    value_readback_buffer.unmap();

    timestamps_readback.map_read().await?;

    let timestamps = timestamps_readback.mapped();
    let gpu_time_elapsed = timestamps[1] - timestamps[0];

    println!("Time elapsed GPU: {} milliseconds", gpu_time_elapsed);

    mem::drop(timestamps);

    timestamps_readback.unmap();

    Ok(())
}
