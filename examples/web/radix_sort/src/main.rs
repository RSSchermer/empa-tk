use std::error::Error;

use arwa::console;
use arwa::window::window;
use empa::adapter::Features;
use empa::arwa::{NavigatorExt, RequestAdapterOptions};
use empa::buffer;
use empa::buffer::Buffer;
use empa::device::DeviceDescriptor;
use empa_tk::prefix_sum::PrefixSum;
use empa_tk::radix_sort::{RadixSort, RadixSortInput};
use futures::FutureExt;

fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    arwa::spawn_local(compute().map(|res| res.unwrap()));
}

async fn compute() -> Result<(), Box<dyn Error>> {
    let window = window();
    let empa = window.navigator().empa();

    let adapter = empa
        .request_adapter(&RequestAdapterOptions::default())
        .await
        .ok_or("adapter not found")?;
    let device = adapter
        .request_device(&DeviceDescriptor {
            required_features: Features::TIMESTAMP_QUERY,
            required_limits: Default::default(),
        })
        .await?;

    let mut radix_sort = RadixSort::init_u32(device.clone());

    let count = 1_000_000;
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

    let global_data_readback: Buffer<[[u32; 256]; 4], _> =
        device.create_buffer_zeroed(buffer::Usages::map_read().and_copy_dst());
    let debug_readback_buffer: Buffer<[u32], _> =
        device.create_buffer(vec![0; count], buffer::Usages::map_read().and_copy_dst());

    let mut encoder = device.create_command_encoder();

    encoder = encoder.write_timestamp(&timestamp_query_set, 0);
    encoder = radix_sort.encode(
        encoder,
        RadixSortInput {
            data: data_buffer.view(),
            temporary_storage: temp_storage_buffer.view(),
        },
    );
    encoder = encoder.write_timestamp(&timestamp_query_set, 1);

    encoder = encoder.copy_buffer_to_buffer_slice(data_buffer.view(), readback_buffer.view());
    encoder =
        encoder.copy_buffer_to_buffer(radix_sort.global_offsets(), global_data_readback.view());
    encoder =
        encoder.copy_buffer_to_buffer_slice(radix_sort.debug(), debug_readback_buffer.view());
    encoder = encoder.resolve_timestamp_query_set(&timestamp_query_set, 0, timestamps.view());
    encoder = encoder.copy_buffer_to_buffer_slice(timestamps.view(), timestamps_readback.view());

    device.queue().submit(encoder.finish());

    readback_buffer.map_read().await?;

    let performance = window.performance();

    let cpu_time_start = performance.now();
    // Note: sort_unstable is about 30% faster, but the GPU radix sort is also a stable sort
    data.sort();
    let cpu_time_end = performance.now();

    let cpu_time_elapsed = cpu_time_end - cpu_time_start;

    {
        let readback = readback_buffer.mapped();

        console::log!(
            "The first 10 numbers computed on the GPU:",
            format!("{:#?}", &readback[..10])
        );
        console::log!(
            "The first 10 numbers computed on the CPU (reference): ",
            format!("{:#?}", &data[..10])
        );

        console::log!(
            "The last 10 numbers computed on the GPU:",
            format!("{:#?}", &readback[readback.len() - 10..])
        );
        console::log!(
            "The last 10 numbers computed on the CPU (reference):",
            format!("{:#?}", &data[data.len() - 10..])
        );

        for i in 0..count {
            assert_eq!(&readback[i], &data[i]);
        }
    }

    // debug_readback_buffer.map_read().await?;
    //
    // {
    //     let debug_data = debug_readback_buffer.mapped();
    //
    //     console::log!("debug data", format!("{:#?}", &*debug_data));
    // }
    //
    // debug_readback_buffer.unmap();
    //
    // global_data_readback.map_read().await?;
    //
    // {
    //     let global_data = global_data_readback.mapped();
    //
    //     console::log!(format!("{:#?}", &global_data[0]));
    // }
    //
    // global_data_readback.unmap();

    timestamps_readback.map_read().await?;

    {
        let timestamps = timestamps_readback.mapped();
        let gpu_time_elapsed = timestamps[1] - timestamps[0];
        let gpu_time_elapsed_ms = (gpu_time_elapsed as f64) / 1_000_000.0;

        console::log!("Time elapsed GPU: %f milliseconds", gpu_time_elapsed_ms);
        console::log!(
            "Time elapsed CPU (Rust `std` sort compiled to WASM): %f milliseconds",
            cpu_time_elapsed
        );
    }

    timestamps_readback.unmap();

    Ok(())
}
