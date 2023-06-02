use std::error::Error;

use arwa::console;
use arwa::window::window;
use empa::arwa::{NavigatorExt, RequestAdapterOptions};
use empa::buffer;
use empa::buffer::Buffer;
use empa::device::DeviceDescriptor;
use empa_tk::prefix_sum::PrefixSum;
use futures::FutureExt;
use empa::adapter::Features;

fn main() {
    arwa::spawn_local(compute().map(|res| res.unwrap()));
}

async fn compute() -> Result<(), Box<dyn Error>> {
    let window = window();
    let empa = window.navigator().empa();

    let adapter = empa
        .request_adapter(&RequestAdapterOptions::default())
        .await
        .ok_or("adapter not found")?;
    let device = adapter.request_device(&DeviceDescriptor {
        required_features: Features::TIMESTAMP_QUERY,
        required_limits: Default::default()
    }).await?;

    let mut evaluator = PrefixSum::init_u32(device.clone());

    let count = 1_000_000;
    let data: Vec<u32> = vec![1; count];

    let data_buffer: Buffer<[u32], _> =
        device.create_buffer(data, buffer::Usages::storage_binding().and_copy_src());
    let readback_buffer: Buffer<[u32], _> = device.create_buffer(
        vec![0; count],
        buffer::Usages::map_read().and_copy_dst(),
    );
    let timestamp_query_set = device.create_timestamp_query_set(2);
    let timestamps = device.create_slice_buffer_zeroed(2, buffer::Usages::query_resolve().and_copy_src());
    let timestamps_readback = device.create_slice_buffer_zeroed(2, buffer::Usages::copy_dst().and_map_read());

    let mut encoder = device.create_command_encoder();

    encoder = encoder.write_timestamp(&timestamp_query_set, 0);
    encoder = evaluator.encode(encoder, data_buffer.view());
    encoder = encoder.write_timestamp(&timestamp_query_set, 1);

    encoder = encoder.copy_buffer_to_buffer_slice(data_buffer.view(), readback_buffer.view());
    encoder = encoder.resolve_timestamp_query_set(&timestamp_query_set, 0, timestamps.view());
    encoder = encoder.copy_buffer_to_buffer_slice(timestamps.view(), timestamps_readback.view());

    device.queue().submit(encoder.finish());

    readback_buffer.map_read().await?;

    {
        let data = readback_buffer.mapped();

        console::log!("The first 10 numbers:", format!("{:#?}", &data[..10]));
        console::log!(
            "The last 10 numbers:",
            format!("{:#?}", &data[data.len() - 10..])
        );
    }

    readback_buffer.unmap();

    timestamps_readback.map_read().await?;

    {
        let timestamps = timestamps_readback.mapped();
        let time_elapsed = timestamps[1] - timestamps[0];

        console::log!("Time elapsed: %i nanoseconds", time_elapsed);
    }

    timestamps_readback.unmap();

    Ok(())
}
