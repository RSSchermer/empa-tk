use std::future::{join, Future};

use empa::buffer::{Buffer, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups};
use empa::device::Device;
use empa::type_flag::{O, X};
use empa::{abi, buffer};

use crate::find_runs::collect_run_starts::{CollectRunStarts, CollectRunStartsResources};
use crate::find_runs::mark_run_starts::{MarkRunStarts, MarkRunStartsResources};
use crate::find_runs::resolve_run_count::{ResolveRunCount, ResolveRunCountResources};
use crate::generate_dispatch::{GenerateDispatch, GenerateDispatchResources};
use crate::prefix_sum::{PrefixSum, PrefixSumInput};

mod collect_run_starts;
mod mark_run_starts;
mod resolve_run_count;

const GROUPS_SIZE: u32 = 256;

pub struct FindRunsInput<'a, T, U> {
    pub data: buffer::View<'a, [T], U>,
    pub count: Option<Uniform<u32>>,
}

pub struct FindRunsOutput<'a, U0, U1, U2> {
    pub run_count: buffer::View<'a, u32, U0>,
    pub run_starts: buffer::View<'a, [u32], U1>,
    pub run_mapping: buffer::View<'a, [u32], U2>,
}

pub struct FindRuns<T>
where
    T: abi::Sized,
{
    device: Device,
    mark_run_starts: MarkRunStarts<T>,
    prefix_sum_inclusive: PrefixSum<u32>,
    collect_run_starts: CollectRunStarts,
    resolve_run_count: ResolveRunCount,
    generate_dispatch: GenerateDispatch,
    group_size: Buffer<u32, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>,
    dispatch: Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
}

impl<T> FindRuns<T>
where
    T: abi::Sized,
{
    async fn init_internal(
        device: Device,
        init_mark_run_starts: impl Future<Output = MarkRunStarts<T>>,
    ) -> Self {
        let (
            mark_run_starts,
            prefix_sum_inclusive,
            collect_run_starts,
            resolve_run_count,
            generate_dispatch,
        ) = join!(
            init_mark_run_starts,
            PrefixSum::init_inclusive_u32(device.clone()),
            CollectRunStarts::init(device.clone()),
            ResolveRunCount::init(device.clone()),
            GenerateDispatch::init(device.clone()),
        )
        .await;

        let group_size = device.create_buffer(GROUPS_SIZE, buffer::Usages::uniform_binding());
        let dispatch = device.create_buffer(
            DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );

        FindRuns {
            device,
            mark_run_starts,
            prefix_sum_inclusive,
            collect_run_starts,
            resolve_run_count,
            generate_dispatch,
            group_size,
            dispatch,
        }
    }

    pub fn encode<U0, U1, U2, U3>(
        &mut self,
        mut encoder: CommandEncoder,
        input: FindRunsInput<T, U0>,
        output: FindRunsOutput<U1, U2, U3>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding + buffer::CopyDst + 'static,
    {
        let FindRunsInput { data, count } = input;

        let FindRunsOutput {
            run_count,
            run_starts,
            run_mapping,
        } = output;

        let dispatch_indirect = count.is_some();

        let count = count.unwrap_or_else(|| {
            self.device
                .create_buffer(data.len() as u32, buffer::Usages::uniform_binding())
                .uniform()
        });

        if dispatch_indirect {
            encoder = self.generate_dispatch.encode(
                encoder,
                GenerateDispatchResources {
                    group_size: self.group_size.uniform(),
                    count: count.clone(),
                    dispatch: self.dispatch.storage(),
                },
            )
        }

        encoder = encoder.clear_buffer_slice(run_mapping);
        encoder = self.mark_run_starts.encode(
            encoder,
            MarkRunStartsResources {
                count: count.clone(),
                data: data.read_only_storage(),
                temporary_storage: run_mapping.storage(),
            },
            dispatch_indirect,
            self.dispatch.view(),
            data.len() as u32,
        );
        encoder = self.prefix_sum_inclusive.encode(
            encoder,
            PrefixSumInput {
                data: run_mapping,
                count: if dispatch_indirect {
                    Some(count.clone())
                } else {
                    None
                },
            },
        );
        encoder = self.collect_run_starts.encode(
            encoder,
            CollectRunStartsResources {
                count: count.clone(),
                temporary_storage: run_mapping.read_only_storage(),
                run_starts: run_starts.storage(),
            },
            dispatch_indirect,
            self.dispatch.view(),
            data.len() as u32,
        );
        encoder = self.resolve_run_count.encode(
            encoder,
            ResolveRunCountResources {
                count,
                temporary_storage: run_mapping.read_only_storage(),
                run_count: run_count.storage(),
            },
        );

        encoder
    }
}

impl FindRuns<u32> {
    pub async fn init_u32(device: Device) -> Self {
        let init_mark_run_starts = MarkRunStarts::init_u32(device.clone());

        FindRuns::init_internal(device, init_mark_run_starts).await
    }
}

impl FindRuns<i32> {
    pub async fn init_i32(device: Device) -> Self {
        let init_mark_run_starts = MarkRunStarts::init_i32(device.clone());

        FindRuns::init_internal(device, init_mark_run_starts).await
    }
}

impl FindRuns<f32> {
    pub async fn init_f32(device: Device) -> Self {
        let init_mark_run_starts = MarkRunStarts::init_f32(device.clone());

        FindRuns::init_internal(device, init_mark_run_starts).await
    }
}
