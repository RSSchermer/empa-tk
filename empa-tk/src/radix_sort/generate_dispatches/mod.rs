use empa::abi;
use empa::access_mode::ReadWrite;
use empa::buffer::{Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::{BindGroupLayout, Resources};
use empa::shader_module::{shader_source, ShaderSource};

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(abi::Sized, Clone, Copy)]
#[repr(C)]
pub struct SegmentSizes {
    pub histogram: u32,
    pub scatter: u32,
}

#[derive(empa::resource_binding::Resources)]
pub struct GenerateDispatchesResources<'a, T>
where
    T: abi::Sized,
{
    #[resource(binding = 0, visibility = "COMPUTE")]
    pub segment_sizes: Uniform<'a, SegmentSizes>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    pub max_count: Uniform<'a, u32>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    pub data: Storage<'a, [T]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    pub histogram_dispatch: Storage<'a, DispatchWorkgroups, ReadWrite>,
    #[resource(binding = 4, visibility = "COMPUTE")]
    pub scatter_dispatch: Storage<'a, DispatchWorkgroups, ReadWrite>,
}

type ResourcesLayout<T> = <GenerateDispatchesResources<'static, T> as Resources>::Layout;

pub struct GenerateDispatches<T>
where
    T: abi::Sized,
{
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout<T>>,
    pipeline: ComputePipeline<(ResourcesLayout<T>,)>,
}

impl<T> GenerateDispatches<T>
where
    T: abi::Sized + 'static,
{
    pub async fn init(device: Device) -> Self {
        let shader = device.create_shader_module(&SHADER);

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

        GenerateDispatches {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode(
        &self,
        encoder: CommandEncoder,
        resources: GenerateDispatchesResources<T>,
    ) -> CommandEncoder {
        let bind_group = self
            .device
            .create_bind_group(&self.bind_group_layout, resources);

        encoder
            .begin_compute_pass()
            .set_pipeline(&self.pipeline)
            .set_bind_groups(&bind_group)
            .dispatch_workgroups(DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            })
            .end()
    }
}
