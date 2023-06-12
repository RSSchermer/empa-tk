use empa::command::CommandEncoder;
use empa::device::Device;
use empa::{abi, buffer};

use crate::find_runs::collect_run_starts::{CollectRunStarts, CollectRunStartsInput};
use crate::find_runs::mark_run_starts::{MarkRunStarts, MarkRunStartsInput};
use crate::find_runs::resolve_run_count::{ResolveRunCount, ResolveRunCountInput};
use crate::prefix_sum::PrefixSumInclusive;

mod collect_run_starts;
mod mark_run_starts;
mod resolve_run_count;

pub struct FindRunsInput<'a, T, U0, U1, U2, U3> {
    pub data: buffer::View<'a, [T], U0>,
    pub temporary_storage: buffer::View<'a, [u32], U1>,
    pub run_starts: buffer::View<'a, [u32], U2>,
    pub run_count: buffer::View<'a, u32, U3>,
}

pub struct FindRuns<T>
where
    T: abi::Sized,
{
    mark_run_starts: MarkRunStarts<T>,
    prefix_sum_inclusive: PrefixSumInclusive<u32>,
    collect_run_starts: CollectRunStarts,
    resolve_run_count: ResolveRunCount,
}

impl<T> FindRuns<T>
where
    T: abi::Sized,
{
    pub fn encode<U0, U1, U2, U3>(
        &mut self,
        mut encoder: CommandEncoder,
        input: FindRunsInput<T, U0, U1, U2, U3>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding + buffer::CopyDst + 'static,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
    {
        let FindRunsInput {
            data,
            temporary_storage,
            run_starts,
            run_count,
        } = input;

        encoder = self.mark_run_starts.encode(
            encoder,
            MarkRunStartsInput {
                data,
                temporary_storage,
            },
        );
        encoder = self.prefix_sum_inclusive.encode(encoder, temporary_storage);
        encoder = self.collect_run_starts.encode(
            encoder,
            CollectRunStartsInput {
                temporary_storage,
                run_starts,
            },
        );
        encoder = self.resolve_run_count.encode(
            encoder,
            ResolveRunCountInput {
                temporary_storage,
                run_count,
            },
        );

        encoder
    }
}

impl FindRuns<u32> {
    pub fn init_u32(device: Device) -> Self {
        let mark_run_starts = MarkRunStarts::init_u32(device.clone());
        let prefix_sum_inclusive = PrefixSumInclusive::init_u32(device.clone());
        let collect_run_starts = CollectRunStarts::init(device.clone());
        let resolve_run_count = ResolveRunCount::init(device);

        FindRuns {
            mark_run_starts,
            prefix_sum_inclusive,
            collect_run_starts,
            resolve_run_count,
        }
    }
}

impl FindRuns<i32> {
    pub fn init_i32(device: Device) -> Self {
        let mark_run_starts = MarkRunStarts::init_i32(device.clone());
        let prefix_sum_inclusive = PrefixSumInclusive::init_u32(device.clone());
        let collect_run_starts = CollectRunStarts::init(device.clone());
        let resolve_run_count = ResolveRunCount::init(device);

        FindRuns {
            mark_run_starts,
            prefix_sum_inclusive,
            collect_run_starts,
            resolve_run_count,
        }
    }
}

impl FindRuns<f32> {
    pub fn init_f32(device: Device) -> Self {
        let mark_run_starts = MarkRunStarts::init_f32(device.clone());
        let prefix_sum_inclusive = PrefixSumInclusive::init_u32(device.clone());
        let collect_run_starts = CollectRunStarts::init(device.clone());
        let resolve_run_count = ResolveRunCount::init(device);

        FindRuns {
            mark_run_starts,
            prefix_sum_inclusive,
            collect_run_starts,
            resolve_run_count,
        }
    }
}
