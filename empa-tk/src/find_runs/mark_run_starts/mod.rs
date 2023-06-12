use empa::buffer::{ReadOnlyStorage, Storage};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::{abi, buffer};

const GROUPS_SIZE: u32 = 256;

const SHADER_U32: ShaderSource = shader_source!("shader_u32.wgsl");
const SHADER_I32: ShaderSource = shader_source!("shader_i32.wgsl");
const SHADER_F32: ShaderSource = shader_source!("shader_f32.wgsl");

#[derive(empa::resource_binding::Resources)]
struct Resources<T>
where
    T: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    data: ReadOnlyStorage<[T]>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    temporary_storage: Storage<[u32]>,
}

type ResourcesLayout<T> = <Resources<T> as empa::resource_binding::Resources>::Layout;

pub struct MarkRunStartsInput<'a, T, U0, U1> {
    pub data: buffer::View<'a, [T], U0>,
    pub temporary_storage: buffer::View<'a, [u32], U1>,
}

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

    pub fn encode<U0, U1>(
        &self,
        encoder: CommandEncoder,
        input: MarkRunStartsInput<T, U0, U1>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding + buffer::CopyDst + 'static,
    {
        let MarkRunStartsInput {
            data,
            temporary_storage,
        } = input;

        assert_eq!(data.len(), temporary_storage.len());

        let workgroups = (data.len() as u32).div_ceil(GROUPS_SIZE);

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                data: data.read_only_storage(),
                temporary_storage: temporary_storage.storage(),
            },
        );

        encoder
            .clear_buffer_slice(temporary_storage)
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
