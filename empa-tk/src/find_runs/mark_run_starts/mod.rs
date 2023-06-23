use empa::buffer::{ReadOnlyStorage, Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::{abi, buffer};

use crate::find_runs::GROUPS_SIZE;

const SHADER_U32: ShaderSource = shader_source!("shader_u32.wgsl");
const SHADER_I32: ShaderSource = shader_source!("shader_i32.wgsl");
const SHADER_F32: ShaderSource = shader_source!("shader_f32.wgsl");

#[derive(empa::resource_binding::Resources)]
pub struct MarkRunStartsResources<T>
where
    T: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    pub count: Uniform<u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    pub data: ReadOnlyStorage<[T]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    pub temporary_storage: Storage<[u32]>,
}

type ResourcesLayout<T> = <MarkRunStartsResources<T> as empa::resource_binding::Resources>::Layout;

pub struct MarkRunStarts<T>
where
    T: abi::Sized,
{
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout<T>>,
    pipeline: ComputePipeline<(ResourcesLayout<T>,)>,
}

impl<T> MarkRunStarts<T>
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

        MarkRunStarts {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U>(
        &self,
        encoder: CommandEncoder,
        resources: MarkRunStartsResources<T>,
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
                    count_x: fallback_count.div_ceil(GROUPS_SIZE),
                    count_y: 1,
                    count_z: 1,
                })
                .end()
        }
    }
}

impl MarkRunStarts<u32> {
    pub fn init_u32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_U32)
    }
}

impl MarkRunStarts<i32> {
    pub fn init_i32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_I32)
    }
}

impl MarkRunStarts<f32> {
    pub fn init_f32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_F32)
    }
}
