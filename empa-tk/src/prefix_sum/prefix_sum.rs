use std::future::join;

use bytemuck::Zeroable;
use empa::buffer::{Buffer, Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::type_flag::{O, X};
use empa::{abi, buffer};

use crate::generate_dispatch::{GenerateDispatch, GenerateDispatchResources};

const GROUPS_SIZE: u32 = 256;
const VALUES_PER_THREAD: u32 = 8;

const SEGMENT_SIZE: u32 = GROUPS_SIZE * VALUES_PER_THREAD;

const EXCLUSIVE_SHADER_U32: ShaderSource = shader_source!("exclusive_shader_u32.wgsl");
const EXCLUSIVE_SHADER_I32: ShaderSource = shader_source!("exclusive_shader_i32.wgsl");
const EXCLUSIVE_SHADER_F32: ShaderSource = shader_source!("exclusive_shader_f32.wgsl");
const INCLUSIVE_SHADER_U32: ShaderSource = shader_source!("inclusive_shader_u32.wgsl");
const INCLUSIVE_SHADER_I32: ShaderSource = shader_source!("inclusive_shader_i32.wgsl");
const INCLUSIVE_SHADER_F32: ShaderSource = shader_source!("inclusive_shader_f32.wgsl");

#[derive(abi::Sized, Clone, Copy, Debug, Zeroable)]
#[repr(C)]
pub struct GroupState {
    state_0: u32,
    state_1: u32,
}

#[derive(empa::resource_binding::Resources)]
struct Resources<T>
where
    T: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    count: Uniform<u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    data: Storage<[T]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    group_state: Storage<[GroupState]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    group_counter: Storage<u32>,
}

type ResourcesLayout<T> = <Resources<T> as empa::resource_binding::Resources>::Layout;

pub struct PrefixSumInput<'a, T, U> {
    pub data: buffer::View<'a, [T], U>,
    pub count: Option<Uniform<u32>>,
}

pub struct PrefixSum<T>
where
    T: abi::Sized,
{
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout<T>>,
    pipeline: ComputePipeline<(ResourcesLayout<T>,)>,
    group_state: Buffer<[GroupState], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
    group_counter: Buffer<u32, buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
    generate_dispatch: GenerateDispatch,
    group_size: Buffer<u32, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>,
    dispatch: Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
}

impl<T> PrefixSum<T>
where
    T: abi::Sized,
{
    async fn init_internal(device: Device, shader_source: &ShaderSource) -> Self {
        let shader = device.create_shader_module(shader_source);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout<T>>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let create_pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptorBuilder::begin()
                .layout(&pipeline_layout)
                .compute(&ComputeStageBuilder::begin(&shader, "main").finish())
                .finish(),
        );
        let group_state =
            device.create_slice_buffer_zeroed(1, buffer::Usages::storage_binding().and_copy_dst());
        let group_counter =
            device.create_buffer(0, buffer::Usages::storage_binding().and_copy_dst());

        let init_generate_dispatch = GenerateDispatch::init(device.clone());
        let group_size = device.create_buffer(SEGMENT_SIZE, buffer::Usages::uniform_binding());
        let dispatch = device.create_buffer(
            DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );

        let (pipeline, generate_dispatch) = join!(create_pipeline, init_generate_dispatch).await;

        PrefixSum {
            device,
            bind_group_layout,
            pipeline,
            group_state,
            group_counter,
            generate_dispatch,
            group_size,
            dispatch,
        }
    }

    pub fn encode<U>(
        &mut self,
        mut encoder: CommandEncoder,
        input: PrefixSumInput<T, U>,
    ) -> CommandEncoder
    where
        U: buffer::StorageBinding,
    {
        let PrefixSumInput { data, count } = input;

        let dispatch_indirect = count.is_some();

        let count = count.unwrap_or_else(|| {
            self.device
                .create_buffer(data.len() as u32, buffer::Usages::uniform_binding())
                .uniform()
        });

        let workgroups = (data.len() as u32).div_ceil(SEGMENT_SIZE);

        if self.group_state.len() < workgroups as usize {
            self.group_state = self
                .device
                .create_slice_buffer_zeroed(workgroups as usize, self.group_state.usage());
        }

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                count: count.clone(),
                data: data.storage(),
                group_state: self.group_state.storage(),
                group_counter: self.group_counter.storage(),
            },
        );

        if dispatch_indirect {
            encoder = self.generate_dispatch.encode(
                encoder,
                GenerateDispatchResources {
                    group_size: self.group_size.uniform(),
                    count,
                    dispatch: self.dispatch.storage(),
                },
            );
        }

        let encoder = encoder
            .clear_buffer(self.group_counter.view())
            .clear_buffer_slice(self.group_state.view())
            .begin_compute_pass()
            .set_pipeline(&self.pipeline)
            .set_bind_groups(&bind_group);

        if dispatch_indirect {
            encoder
                .dispatch_workgroups_indirect(self.dispatch.view())
                .end()
        } else {
            encoder
                .dispatch_workgroups(DispatchWorkgroups {
                    count_x: workgroups,
                    count_y: 1,
                    count_z: 1,
                })
                .end()
        }
    }
}

impl PrefixSum<u32> {
    pub async fn init_exclusive_u32(device: Device) -> Self {
        Self::init_internal(device, &EXCLUSIVE_SHADER_U32).await
    }
    pub async fn init_inclusive_u32(device: Device) -> Self {
        Self::init_internal(device, &INCLUSIVE_SHADER_U32).await
    }
}

impl PrefixSum<i32> {
    pub async fn init_exclusive_i32(device: Device) -> Self {
        Self::init_internal(device, &EXCLUSIVE_SHADER_I32).await
    }
    pub async fn init_inclusive_i32(device: Device) -> Self {
        Self::init_internal(device, &INCLUSIVE_SHADER_I32).await
    }
}

impl PrefixSum<f32> {
    pub async fn init_exclusive_f32(device: Device) -> Self {
        Self::init_internal(device, &EXCLUSIVE_SHADER_F32).await
    }
    pub async fn init_inclusive_f32(device: Device) -> Self {
        Self::init_internal(device, &INCLUSIVE_SHADER_F32).await
    }
}
