use empa::buffer::{Buffer, Storage};
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

const GROUPS_SIZE: u32 = 256;
const VALUES_PER_THREAD: u32 = 8;

const SEGMENT_SIZE: u32 = GROUPS_SIZE * VALUES_PER_THREAD;

const SHADER_U32: ShaderSource = shader_source!("shader_u32.wgsl");
const SHADER_I32: ShaderSource = shader_source!("shader_i32.wgsl");
const SHADER_F32: ShaderSource = shader_source!("shader_f32.wgsl");

#[derive(abi::Sized, Clone, Copy, Debug, Zeroable)]
#[repr(C)]
pub struct GroupState {
    aggregate: u32,
    inclusive_prefix: u32,
    status: u32,
}

#[derive(empa::resource_binding::Resources)]
struct Resources<T>
where
    T: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    data: Storage<[T]>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    group_state: Storage<[GroupState]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    group_counter: Storage<u32>,
}

type ResourcesLayout<T> = <Resources<T> as empa::resource_binding::Resources>::Layout;

pub struct PrefixSumExclusive<T>
where
    T: abi::Sized,
{
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout<T>>,
    pipeline: ComputePipeline<(ResourcesLayout<T>,)>,
    group_state: Buffer<[GroupState], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
    group_counter: Buffer<u32, buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
}

impl<T> PrefixSumExclusive<T>
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
        let group_state =
            device.create_slice_buffer_zeroed(1, buffer::Usages::storage_binding().and_copy_dst());
        let group_counter =
            device.create_buffer(0, buffer::Usages::storage_binding().and_copy_dst());

        PrefixSumExclusive {
            device,
            bind_group_layout,
            pipeline,
            group_state,
            group_counter,
        }
    }

    pub fn encode<U>(
        &mut self,
        encoder: CommandEncoder,
        data: buffer::View<[T], U>,
    ) -> CommandEncoder
    where
        U: buffer::StorageBinding,
    {
        let workgroups = (data.len() as u32).div_ceil(SEGMENT_SIZE);

        if self.group_state.len() < workgroups as usize {
            self.group_state = self
                .device
                .create_slice_buffer_zeroed(workgroups as usize, self.group_state.usage());
        }

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                data: data.storage(),
                group_state: self.group_state.storage(),
                group_counter: self.group_counter.storage(),
            },
        );

        encoder
            .clear_buffer(self.group_counter.view())
            .clear_buffer_slice(self.group_state.view())
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

impl PrefixSumExclusive<u32> {
    pub fn init_u32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_U32)
    }
}

impl PrefixSumExclusive<i32> {
    pub fn init_i32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_I32)
    }
}

impl PrefixSumExclusive<f32> {
    pub fn init_f32(device: Device) -> Self {
        Self::init_internal(device, &SHADER_F32)
    }
}
