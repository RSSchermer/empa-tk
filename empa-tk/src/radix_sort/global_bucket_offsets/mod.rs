use empa::buffer;
use empa::buffer::Storage;
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};

use crate::radix_sort::{RADIX_DIGITS, RADIX_GROUPS};

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
struct Resources {
    #[resource(binding = 0, visibility = "COMPUTE")]
    global_data: Storage<[[u32; RADIX_DIGITS]; RADIX_GROUPS]>,
}

type ResourcesLayout = <Resources as empa::resource_binding::Resources>::Layout;

pub struct GlobalBucketOffsets {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline: ComputePipeline<(ResourcesLayout,)>,
}

impl GlobalBucketOffsets {
    pub async fn init(device: Device) -> Self {
        let shader = device.create_shader_module(&SHADER);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = device
            .create_compute_pipeline(
                &ComputePipelineDescriptorBuilder::begin()
                    .layout(&pipeline_layout)
                    .compute(&ComputeStageBuilder::begin(&shader, "main").finish())
                    .finish(),
            )
            .await;

        GlobalBucketOffsets {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U0>(
        &mut self,
        encoder: CommandEncoder,
        global_data: buffer::View<[[u32; RADIX_DIGITS]; RADIX_GROUPS], U0>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
    {
        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                global_data: global_data.storage(),
            },
        );

        encoder
            .begin_compute_pass()
            .set_pipeline(&self.pipeline)
            .set_bind_groups(&bind_group)
            .dispatch_workgroups(DispatchWorkgroups {
                count_x: RADIX_GROUPS as u32,
                count_y: 1,
                count_z: 1,
            })
            .end()
    }
}
