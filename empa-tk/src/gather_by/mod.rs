use std::fmt::Write;

use empa::buffer::{ReadOnlyStorage, Storage};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::ShaderSource;
use empa::{abi, buffer};

use crate::write_value_type::write_value_type;

const SHADER_TEMPLATE: &str = include_str!("shader_template.wgsl");

const GROUP_SIZE: u32 = 256;

#[derive(empa::resource_binding::Resources)]
struct Resources<B, V>
where
    B: abi::Sized,
    V: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    gather_by: ReadOnlyStorage<[B]>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    data_in: ReadOnlyStorage<[V]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    data_out: Storage<[V]>,
}

type ResourcesLayout<K, V> = <Resources<K, V> as empa::resource_binding::Resources>::Layout;

pub struct GatherByInput<'a, B, V, U0, U1> {
    pub gather_by: buffer::View<'a, [B], U0>,
    pub data: buffer::View<'a, [V], U1>,
}

pub struct GatherBy<B, V>
where
    B: abi::Sized,
    V: abi::Sized,
{
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout<B, V>>,
    pipeline: ComputePipeline<(ResourcesLayout<B, V>,)>,
}

impl<B, V> GatherBy<B, V>
where
    B: abi::Sized,
    V: abi::Sized,
{
    fn init_internal(device: Device, by_type: &str, shader_template: &str) -> Self {
        let mut code = String::new();

        write_value_type::<V>(&mut code);

        write!(code, "alias BY_TYPE = {};\n\n{}", by_type, shader_template).unwrap();

        let shader_source = ShaderSource::parse(code).unwrap();
        let shader = device.create_shader_module(&shader_source);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout<B, V>>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = unsafe {
            device.create_compute_pipeline(
                &ComputePipelineDescriptorBuilder::begin()
                    .layout(&pipeline_layout)
                    .compute_unchecked(&ComputeStageBuilder::begin(&shader, "main").finish())
                    .finish(),
            )
        };

        GatherBy {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U0, U1, U2>(
        &mut self,
        encoder: CommandEncoder,
        input: GatherByInput<B, V, U0, U1>,
        output: buffer::View<[V], U2>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
    {
        let GatherByInput { gather_by, data } = input;

        let workgroups = (gather_by.len() as u32).div_ceil(GROUP_SIZE);

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                gather_by: gather_by.read_only_storage(),
                data_in: data.read_only_storage(),
                data_out: output.storage(),
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

impl<V> GatherBy<u32, V>
where
    V: abi::Sized,
{
    pub fn init_u32(device: Device) -> Self {
        Self::init_internal(device, "u32", SHADER_TEMPLATE)
    }
}

impl<V> GatherBy<i32, V>
where
    V: abi::Sized,
{
    pub fn init_i32(device: Device) -> Self {
        Self::init_internal(device, "i32", SHADER_TEMPLATE)
    }
}
