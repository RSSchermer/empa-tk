use std::fmt;

use empa::buffer::{Buffer, ReadOnlyStorage, Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::type_flag::{O, X};
use empa::{abi, buffer};
use zeroable::Zeroable;

use crate::radix_sort::{RADIX_DIGITS, RADIX_GROUPS, RADIX_SIZE, SEGMENT_SIZE};

const SHADER_U32: ShaderSource = shader_source!("shader_u32.wgsl");

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u32)]
enum GroupStatus {
    NotReady = 0,
    LocalOffset = 1,
    GlobalOffset = 2,
}

#[derive(abi::Sized, Clone, Copy, Zeroable)]
#[repr(C)]
struct GroupState {
    packed_data: u32,
}

impl fmt::Debug for GroupState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = self.packed_data >> 30;
        let value = self.packed_data & 0x3FFFFFFF;

        let status = match status {
            0 => GroupStatus::NotReady,
            1 => GroupStatus::LocalOffset,
            2 => GroupStatus::GlobalOffset,
            _ => unreachable!(),
        };

        f.debug_struct("GroupState")
            .field("status", &status)
            .field("value", &value)
            .finish()
    }
}

#[derive(abi::Sized, Clone, Copy, Debug, Zeroable)]
#[repr(C)]
pub struct Uniforms {
    radix_offset: u32,
    radix_group: u32,
}

#[derive(empa::resource_binding::Resources)]
struct Resources<T>
where
    T: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    uniforms: Uniform<Uniforms>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    data_in: ReadOnlyStorage<[T]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    data_out: Storage<[T]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    global_base_bucket_offsets: ReadOnlyStorage<[[u32; RADIX_DIGITS]; RADIX_GROUPS]>,
    #[resource(binding = 4, visibility = "COMPUTE")]
    group_state: Storage<[[GroupState; RADIX_DIGITS]]>,
    #[resource(binding = 5, visibility = "COMPUTE")]
    group_counter: Storage<u32>,
    #[resource(binding = 6, visibility = "COMPUTE")]
    debug: Storage<[T]>,
}

type ResourcesLayout<T> = <Resources<T> as empa::resource_binding::Resources>::Layout;

pub struct BucketScatterInput<'a, T, U0, U1, U2> {
    pub data_in: buffer::View<'a, [T], U0>,
    pub data_out: buffer::View<'a, [T], U1>,
    pub global_base_bucket_offsets: buffer::View<'a, [[u32; RADIX_DIGITS]; RADIX_GROUPS], U2>,
    pub radix_group: u32,
}

pub struct BucketScatter<T>
where
    T: abi::Sized,
{
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout<T>>,
    pipeline: ComputePipeline<(ResourcesLayout<T>,)>,
    group_state: Buffer<[[GroupState; RADIX_DIGITS]], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
    group_counter: Buffer<u32, buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
    debug_buffer: Buffer<[T], buffer::Usages<O, O, X, O, O, O, X, X, O, O>>,
}

impl<T> BucketScatter<T>
where
    T: abi::Sized + Zeroable,
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
        let group_state =
            device.create_slice_buffer_zeroed(1, buffer::Usages::storage_binding().and_copy_dst());
        let group_counter =
            device.create_buffer(0, buffer::Usages::storage_binding().and_copy_dst());

        let debug_buffer = device.create_slice_buffer_zeroed(1, buffer::Usages::storage_binding().and_copy_dst().and_copy_src());

        BucketScatter {
            device,
            bind_group_layout,
            pipeline,
            group_state,
            group_counter,
            debug_buffer
        }
    }

    pub fn encode<U0, U1, U2>(
        &mut self,
        encoder: CommandEncoder,
        input: BucketScatterInput<T, U0, U1, U2>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
    {
        let BucketScatterInput {
            data_in,
            data_out,
            global_base_bucket_offsets,
            radix_group,
        } = input;

        let radix_offset = RADIX_SIZE * radix_group;

        let workgroups = (data_in.len() as u32).div_ceil(SEGMENT_SIZE);

        if self.group_state.len() < workgroups as usize {
            self.group_state = self
                .device
                .create_slice_buffer_zeroed(workgroups as usize, self.group_state.usage());
        }

        if self.debug_buffer.len() < data_in.len() {
            self.debug_buffer = self.device.create_slice_buffer_zeroed(data_in.len(), self.debug_buffer.usage());
        }

        let uniforms = self.device.create_buffer(
            Uniforms {
                radix_offset,
                radix_group,
            },
            buffer::Usages::uniform_binding(),
        );

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                uniforms: uniforms.uniform(),
                data_in: data_in.read_only_storage(),
                data_out: data_out.storage(),
                global_base_bucket_offsets: global_base_bucket_offsets.read_only_storage(),
                group_state: self.group_state.storage(),
                group_counter: self.group_counter.storage(),
                debug: self.debug_buffer.storage()
            },
        );

        encoder
            .clear_buffer(self.group_counter.view())
            .clear_buffer_slice(self.group_state.view())
            .clear_buffer_slice(self.debug_buffer.view())
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

    pub fn debug(&self) -> buffer::View<[T], impl buffer::CopySrc> {
        self.debug_buffer.view()
    }
}

impl BucketScatter<u32> {
    pub fn init_u32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_U32)
    }
}
