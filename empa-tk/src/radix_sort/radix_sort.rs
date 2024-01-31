use std::future::join;

use empa::buffer::{Buffer, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups};
use empa::device::Device;
use empa::type_flag::{O, X};
use empa::{abi, buffer};

use crate::radix_sort::bucket_histogram::{
    BucketHistogram, BucketHistogramResources, BUCKET_HISTOGRAM_SEGMENT_SIZE,
};
use crate::radix_sort::bucket_scatter::{
    BucketScatter, BucketScatterInput, BUCKET_SCATTER_SEGMENT_SIZE,
};
use crate::radix_sort::generate_dispatches::{
    GenerateDispatches, GenerateDispatchesResources, SegmentSizes,
};
use crate::radix_sort::global_bucket_offsets::GlobalBucketOffsets;
use crate::radix_sort::{RADIX_DIGITS, RADIX_GROUPS};

pub struct RadixSortInput<'a, T, U0, U1> {
    pub data: buffer::View<'a, [T], U0>,
    pub temporary_storage: buffer::View<'a, [T], U1>,
    pub count: Option<Uniform<u32>>,
}

pub struct RadixSort<T>
where
    T: abi::Sized,
{
    device: Device,
    generate_dispatches: GenerateDispatches,
    bucket_histogram: BucketHistogram<T>,
    global_bucket_offsets: GlobalBucketOffsets,
    bucket_scatter: BucketScatter<T>,
    global_bucket_data:
        Buffer<[[u32; RADIX_DIGITS]; RADIX_GROUPS], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
    segment_sizes: Buffer<SegmentSizes, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>,
    histogram_dispatch: Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
    scatter_dispatch: Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
}

impl<T> RadixSort<T>
where
    T: abi::Sized,
{
    pub fn encode<U0, U1>(
        &mut self,
        mut encoder: CommandEncoder,
        input: RadixSortInput<T, U0, U1>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
    {
        self.encode_internal(encoder, input, 4)
    }

    fn encode_internal<U0, U1>(
        &mut self,
        mut encoder: CommandEncoder,
        input: RadixSortInput<T, U0, U1>,
        radix_groups: usize,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
    {
        let RadixSortInput {
            data,
            temporary_storage,
            count,
        } = input;

        let dispatch_indirect = count.is_some();

        let count = count.unwrap_or_else(|| {
            self.device
                .create_buffer(data.len() as u32, buffer::Usages::uniform_binding())
                .uniform()
        });

        let fallback_count = data.len() as u32;

        if dispatch_indirect {
            encoder = self.generate_dispatches.encode(
                encoder,
                GenerateDispatchesResources {
                    segment_sizes: self.segment_sizes.uniform(),
                    count: count.clone(),
                    histogram_dispatch: self.histogram_dispatch.storage(),
                    scatter_dispatch: self.scatter_dispatch.storage(),
                },
            );
        }

        encoder = encoder.clear_buffer(self.global_bucket_data.view());
        encoder = self.bucket_histogram.encode(
            encoder,
            BucketHistogramResources {
                count: count.clone(),
                data: data.read_only_storage(),
                global_histograms: self.global_bucket_data.storage(),
            },
            dispatch_indirect,
            self.histogram_dispatch.view(),
            fallback_count,
        );
        encoder = self
            .global_bucket_offsets
            .encode(encoder, self.global_bucket_data.view());

        let data_a = data;
        let data_b = temporary_storage;

        for i in 0..radix_groups {
            if (i & 1) == 0 {
                encoder = self.bucket_scatter.encode(
                    encoder,
                    BucketScatterInput {
                        data_in: data_a,
                        data_out: data_b,
                        global_base_bucket_offsets: self.global_bucket_data.view(),
                        radix_group: i as u32,
                        count: count.clone(),
                        dispatch_indirect,
                        dispatch: self.scatter_dispatch.view(),
                        fallback_count,
                    },
                );
            } else {
                encoder = self.bucket_scatter.encode(
                    encoder,
                    BucketScatterInput {
                        data_in: data_b,
                        data_out: data_a,
                        global_base_bucket_offsets: self.global_bucket_data.view(),
                        radix_group: i as u32,
                        count: count.clone(),
                        dispatch_indirect,
                        dispatch: self.scatter_dispatch.view(),
                        fallback_count,
                    },
                );
            }
        }

        encoder
    }
}

impl RadixSort<u32> {
    pub async fn init_u32(device: Device) -> Self {
        let global_bucket_data =
            device.create_buffer_zeroed(buffer::Usages::storage_binding().and_copy_dst());

        let (generate_dispatches, bucket_histogram, global_bucket_offsets, bucket_scatter) = join!(
            GenerateDispatches::init(device.clone()),
            BucketHistogram::init_u32(device.clone()),
            GlobalBucketOffsets::init(device.clone()),
            BucketScatter::init_u32(device.clone()),
        )
        .await;

        let segment_sizes = device.create_buffer(
            SegmentSizes {
                histogram: BUCKET_HISTOGRAM_SEGMENT_SIZE,
                scatter: BUCKET_SCATTER_SEGMENT_SIZE,
            },
            buffer::Usages::uniform_binding(),
        );
        let histogram_dispatch = device.create_buffer(
            DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );
        let scatter_dispatch = device.create_buffer(
            DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );

        RadixSort {
            device,
            generate_dispatches,
            bucket_histogram,
            global_bucket_offsets,
            bucket_scatter,
            global_bucket_data,
            segment_sizes,
            histogram_dispatch,
            scatter_dispatch,
        }
    }

    pub fn encode_half_precision<U0, U1>(
        &mut self,
        mut encoder: CommandEncoder,
        input: RadixSortInput<u32, U0, U1>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
    {
        self.encode_internal(encoder, input, 2)
    }
}
