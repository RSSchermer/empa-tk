use std::fmt;
use std::fmt::Write;

use bytemuck::Zeroable;
use empa::buffer::{Buffer, ReadOnlyStorage, Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::ShaderSource;
use empa::type_flag::{O, X};
use empa::{abi, buffer};

use crate::radix_sort::{RADIX_DIGITS, RADIX_GROUPS, RADIX_SIZE};
use crate::write_value_type::write_value_type;

const SHADER_TEMPLATE_U32: &str = include_str!("shader_template_u32.wgsl");

const GROUP_SIZE: u32 = 256;
const VALUES_PER_THREAD: u32 = 4;

pub const BUCKET_SCATTER_BY_SEGMENT_SIZE: u32 = GROUP_SIZE * VALUES_PER_THREAD;

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
struct Resources<K, V>
where
    K: abi::Sized,
    V: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    count: Uniform<u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    uniforms: Uniform<Uniforms>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    keys_in: ReadOnlyStorage<[K]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    keys_out: Storage<[K]>,
    #[resource(binding = 4, visibility = "COMPUTE")]
    values_in: ReadOnlyStorage<[V]>,
    #[resource(binding = 5, visibility = "COMPUTE")]
    values_out: Storage<[V]>,
    #[resource(binding = 6, visibility = "COMPUTE")]
    global_base_bucket_offsets: ReadOnlyStorage<[[u32; RADIX_DIGITS]; RADIX_GROUPS]>,
    #[resource(binding = 7, visibility = "COMPUTE")]
    group_state: Storage<[[GroupState; RADIX_DIGITS]]>,
    #[resource(binding = 8, visibility = "COMPUTE")]
    group_counter: Storage<u32>,
}

type ResourcesLayout<K, V> = <Resources<K, V> as empa::resource_binding::Resources>::Layout;

pub struct BucketScatterByInput<'a, K, V, U0, U1, U2, U3, U4, U5> {
    pub keys_in: buffer::View<'a, [K], U0>,
    pub keys_out: buffer::View<'a, [K], U1>,
    pub values_in: buffer::View<'a, [V], U2>,
    pub values_out: buffer::View<'a, [V], U3>,
    pub global_base_bucket_offsets: buffer::View<'a, [[u32; RADIX_DIGITS]; RADIX_GROUPS], U4>,
    pub radix_group: u32,
    pub count: Uniform<u32>,
    pub dispatch_indirect: bool,
    pub dispatch: buffer::View<'a, DispatchWorkgroups, U5>,
    pub fallback_count: u32,
}

pub struct BucketScatterBy<K, V>
where
    K: abi::Sized,
    V: abi::Sized,
{
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout<K, V>>,
    pipeline: ComputePipeline<(ResourcesLayout<K, V>,)>,
    group_state: Buffer<[[GroupState; RADIX_DIGITS]], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
    group_counter: Buffer<u32, buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
}

impl<K, V> BucketScatterBy<K, V>
where
    K: abi::Sized,
    V: abi::Sized,
{
    async fn init_internal(device: Device, shader_template: &str) -> Self {
        let mut code = String::new();

        write_value_type::<V>(&mut code);

        write!(code, "{}", shader_template).unwrap();

        let shader_source = ShaderSource::parse(code).unwrap();
        let shader = device.create_shader_module(&shader_source);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout<K, V>>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = unsafe {
            device.create_compute_pipeline(
                &ComputePipelineDescriptorBuilder::begin()
                    .layout(&pipeline_layout)
                    .compute_unchecked(&ComputeStageBuilder::begin(&shader, "main").finish())
                    .finish(),
            )
        }
        .await;

        let group_state =
            device.create_slice_buffer_zeroed(1, buffer::Usages::storage_binding().and_copy_dst());
        let group_counter =
            device.create_buffer(0, buffer::Usages::storage_binding().and_copy_dst());

        BucketScatterBy {
            device,
            bind_group_layout,
            pipeline,
            group_state,
            group_counter,
        }
    }

    pub fn encode<U0, U1, U2, U3, U4, U5>(
        &mut self,
        encoder: CommandEncoder,
        input: BucketScatterByInput<K, V, U0, U1, U2, U3, U4, U5>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
        U4: buffer::StorageBinding,
        U5: buffer::Indirect,
    {
        let BucketScatterByInput {
            keys_in,
            keys_out,
            values_in,
            values_out,
            global_base_bucket_offsets,
            radix_group,
            count,
            dispatch_indirect,
            dispatch,
            fallback_count,
        } = input;

        let radix_offset = RADIX_SIZE * radix_group;

        let fallback_groups = fallback_count.div_ceil(BUCKET_SCATTER_BY_SEGMENT_SIZE);

        if self.group_state.len() < fallback_groups as usize {
            self.group_state = self
                .device
                .create_slice_buffer_zeroed(fallback_groups as usize, self.group_state.usage());
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
                count,
                uniforms: uniforms.uniform(),
                keys_in: keys_in.read_only_storage(),
                keys_out: keys_out.storage(),
                values_in: values_in.read_only_storage(),
                values_out: values_out.storage(),
                global_base_bucket_offsets: global_base_bucket_offsets.read_only_storage(),
                group_state: self.group_state.storage(),
                group_counter: self.group_counter.storage(),
            },
        );

        let encoder = encoder
            .clear_buffer(self.group_counter.view())
            .clear_buffer_slice(self.group_state.view())
            .begin_compute_pass()
            .set_pipeline(&self.pipeline)
            .set_bind_groups(&bind_group);

        if dispatch_indirect {
            encoder.dispatch_workgroups_indirect(dispatch).end()
        } else {
            encoder
                .dispatch_workgroups(DispatchWorkgroups {
                    count_x: fallback_groups,
                    count_y: 1,
                    count_z: 1,
                })
                .end()
        }
    }
}

impl<V> BucketScatterBy<u32, V>
where
    V: abi::Sized,
{
    pub async fn init_u32(device: Device) -> Self {
        Self::init_internal(device, SHADER_TEMPLATE_U32).await
    }
}
