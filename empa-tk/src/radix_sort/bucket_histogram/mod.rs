use empa::access_mode::ReadWrite;
use empa::buffer::{Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::{BindGroupLayout, Resources};
use empa::shader_module::{shader_source, ShaderSource};
use empa::{abi, buffer};

use crate::radix_sort::{RADIX_DIGITS, RADIX_GROUPS};

const SHADER_U32: ShaderSource = shader_source!("shader_u32.wgsl");

const GROUP_SIZE: u32 = 256;
const GROUP_ITERATIONS: u32 = 4;
pub const BUCKET_HISTOGRAM_SEGMENT_SIZE: u32 = GROUP_SIZE * GROUP_ITERATIONS;

#[derive(empa::resource_binding::Resources)]
pub struct BucketHistogramResources<'a, T>
where
    T: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    pub max_count: Uniform<'a, u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    pub data: Storage<'a, [T]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    pub global_histograms: Storage<'a, [[u32; RADIX_DIGITS]; RADIX_GROUPS], ReadWrite>,
}

type ResourcesLayout<T> = <BucketHistogramResources<'static, T> as Resources>::Layout;

pub struct BucketHistogram<T>
where
    T: abi::Sized,
{
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout<T>>,
    pipeline: ComputePipeline<(ResourcesLayout<T>,)>,
}

impl<T> BucketHistogram<T>
where
    T: abi::Sized + 'static,
{
    async fn init_internal(device: Device, shader_source: &ShaderSource) -> Self {
        let shader = device.create_shader_module(shader_source);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout<T>>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = device
            .create_compute_pipeline(
                &ComputePipelineDescriptorBuilder::begin()
                    .layout(&pipeline_layout)
                    .compute(ComputeStageBuilder::begin(&shader, "main").finish())
                    .finish(),
            )
            .await;

        BucketHistogram {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U>(
        &mut self,
        encoder: CommandEncoder,
        resources: BucketHistogramResources<T>,
        dispatch_indirect: bool,
        dispatch: buffer::View<DispatchWorkgroups, U>,
        fallback_count: u32,
    ) -> CommandEncoder
    where
        U: buffer::Indirect,
    {
        let bind_group = self
            .device
            .create_bind_group(&self.bind_group_layout, resources);

        let encoder = encoder
            .begin_compute_pass()
            .set_pipeline(&self.pipeline)
            .set_bind_groups(&bind_group);

        if dispatch_indirect {
            encoder.dispatch_workgroups_indirect(dispatch).end()
        } else {
            encoder
                .dispatch_workgroups(DispatchWorkgroups {
                    count_x: fallback_count.div_ceil(BUCKET_HISTOGRAM_SEGMENT_SIZE),
                    count_y: 1,
                    count_z: 1,
                })
                .end()
        }
    }
}

impl BucketHistogram<u32> {
    pub async fn init_u32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_U32).await
    }
}
