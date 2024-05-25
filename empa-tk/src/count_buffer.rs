use empa::buffer::{Buffer, Uniform, Usages};
use empa::device::Device;
use empa::type_flag::{O, X};

pub enum CountBuffer<'a> {
    Binding(Uniform<'a, u32>),
    Buffer(Buffer<u32, Usages<O, O, O, X, O, O, O, O, O, O>>),
}

impl<'a> CountBuffer<'a> {
    pub fn new(binding: Option<Uniform<'a, u32>>, device: &Device, fallback_count: u32) -> Self {
        if let Some(binding) = binding {
            Self::Binding(binding)
        } else {
            let buffer = device.create_buffer(fallback_count, Usages::uniform_binding());

            Self::Buffer(buffer)
        }
    }

    pub fn uniform(&self) -> Uniform<u32> {
        match self {
            CountBuffer::Binding(binding) => binding.clone(),
            CountBuffer::Buffer(buffer) => buffer.uniform(),
        }
    }
}
