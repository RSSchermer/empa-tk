use empa::buffer::{ReadOnlyStorage, Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
pub struct ResolveRunCountResources {
    #[resource(binding = 0, visibility = "COMPUTE")]
    pub count: Uniform<u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    pub temporary_storage: ReadOnlyStorage<[u32]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    pub run_count: Storage<u32>,
}

type ResourcesLayout = <ResolveRunCountResources as empa::resource_binding::Resources>::Layout;

pub struct ResolveRunCount {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline: ComputePipeline<(ResourcesLayout,)>,
}

impl ResolveRunCount {
    pub async fn init(device: Device) -> Self {
        let shader = device.create_shader_module(&SHADER);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = device
            .create_compute_pipeline(
                &ComputePipelineDescriptorBuilder::begin()
                    .layout(&pipeline_layout)
                    .compute(&ComputeStageBuilder::begin(&shader, "main").finish())
                    .finish(),
            )
            .await;

        ResolveRunCount {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode(
        &self,
        encoder: CommandEncoder,
        resources: ResolveRunCountResources,
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
