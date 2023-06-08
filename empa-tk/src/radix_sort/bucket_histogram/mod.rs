use empa::buffer::{ReadOnlyStorage, Storage};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::{abi, buffer};

use crate::radix_sort::{RADIX_DIGITS, RADIX_GROUPS};

const SHADER_U32: ShaderSource = shader_source!("shader_u32.wgsl");

const GROUP_SIZE: u32 = 256;
const GROUP_ITERATIONS: u32 = 4;
const SEGMENT_SIZE: u32 = GROUP_SIZE * GROUP_ITERATIONS;

#[derive(empa::resource_binding::Resources)]
struct Resources<T>
where
    T: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    data: ReadOnlyStorage<[T]>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    global_histograms: Storage<[[u32; RADIX_DIGITS]; RADIX_GROUPS]>,
}

type ResourcesLayout<T> = <Resources<T> as empa::resource_binding::Resources>::Layout;

pub struct BucketHistogramInput<'a, T, U0, U1> {
    pub data: buffer::View<'a, [T], U0>,
    pub global_histograms: buffer::View<'a, [[u32; RADIX_DIGITS]; RADIX_GROUPS], U1>,
}

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
    T: abi::Sized,
{
    fn init_internal(device: Device, shader_source: &ShaderSource) -> Self {
        let shader = device.create_shader_module(shader_source);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout<T>>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptorBuilder::begin()
                .layout(&pipeline_layout)
                .compute(&ComputeStageBuilder::begin(&shader, "main").finish())
                .finish(),
        );

        BucketHistogram {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U0, U1>(
        &mut self,
        encoder: CommandEncoder,
        input: BucketHistogramInput<T, U0, U1>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
    {
        let BucketHistogramInput {
            data,
            global_histograms,
        } = input;

        let workgroups = (data.len() as u32).div_ceil(SEGMENT_SIZE);

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                data: data.read_only_storage(),
                global_histograms: global_histograms.storage(),
            },
        );

        encoder
            .begin_compute_pass()
            .set_pipeline(&self.pipeline)
            .set_bind_groups(&bind_group)
            .dispatch_workgroups(DispatchWorkgroups {
                count_x: workgroups,
                count_y: 1,
                count_z: 1,
            })
            .end()
    }
}

impl BucketHistogram<u32> {
    pub fn init_u32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_U32)
    }
}
