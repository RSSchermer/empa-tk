use std::future::join;

use empa::buffer::{Buffer, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups};
use empa::device::Device;
use empa::type_flag::{O, X};
use empa::{abi, buffer};

use crate::count_buffer::CountBuffer;
use crate::radix_sort::bucket_histogram::{
    BucketHistogram, BucketHistogramResources, BUCKET_HISTOGRAM_SEGMENT_SIZE,
};
use crate::radix_sort::bucket_scatter_by::{
    BucketScatterBy, BucketScatterByInput, BUCKET_SCATTER_BY_SEGMENT_SIZE,
};
use crate::radix_sort::generate_dispatches::{
    GenerateDispatches, GenerateDispatchesResources, SegmentSizes,
};
use crate::radix_sort::global_bucket_offsets::GlobalBucketOffsets;
use crate::radix_sort::{RADIX_DIGITS, RADIX_GROUPS};

pub struct RadixSortByInput<'a, K, V, U0, U1, U2, U3> {
    pub keys: buffer::View<'a, [K], U0>,
    pub values: buffer::View<'a, [V], U1>,
    pub temporary_key_storage: buffer::View<'a, [K], U2>,
    pub temporary_value_storage: buffer::View<'a, [V], U3>,
    pub count: Option<Uniform<'a, u32>>,
}

pub struct RadixSortBy<K, V>
where
    K: abi::Sized,
    V: abi::Sized,
{
    device: Device,
    generate_dispatches: GenerateDispatches<K>,
    bucket_histogram: BucketHistogram<K>,
    global_bucket_offsets: GlobalBucketOffsets,
    bucket_scatter_by: BucketScatterBy<K, V>,
    global_bucket_data:
        Buffer<[[u32; RADIX_DIGITS]; RADIX_GROUPS], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
    segment_sizes: Buffer<SegmentSizes, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>,
    histogram_dispatch: Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
    scatter_dispatch: Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
}

impl<K, V> RadixSortBy<K, V>
where
    K: abi::Sized + 'static,
    V: abi::Sized + 'static,
{
    pub fn encode<U0, U1, U2, U3>(
        &mut self,
        encoder: CommandEncoder,
        input: RadixSortByInput<K, V, U0, U1, U2, U3>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
    {
        self.encode_internal(encoder, input, 4)
    }

    fn encode_internal<U0, U1, U2, U3>(
        &mut self,
        mut encoder: CommandEncoder,
        input: RadixSortByInput<K, V, U0, U1, U2, U3>,
        radix_groups: usize,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
    {
        let RadixSortByInput {
            keys,
            values,
            temporary_key_storage,
            temporary_value_storage,
            count,
        } = input;

        let dispatch_indirect = count.is_some();
        let fallback_count = keys.len() as u32;
        let count = CountBuffer::new(count, &self.device, fallback_count);

        if dispatch_indirect {
            encoder = self.generate_dispatches.encode(
                encoder,
                GenerateDispatchesResources {
                    segment_sizes: self.segment_sizes.uniform(),
                    max_count: count.uniform(),
                    data: keys.storage(),
                    histogram_dispatch: self.histogram_dispatch.storage(),
                    scatter_dispatch: self.scatter_dispatch.storage(),
                },
            );
        }

        encoder = encoder.clear_buffer(self.global_bucket_data.view());
        encoder = self.bucket_histogram.encode(
            encoder,
            BucketHistogramResources {
                max_count: count.uniform(),
                data: keys.storage(),
                global_histograms: self.global_bucket_data.storage(),
            },
            dispatch_indirect,
            self.histogram_dispatch.view(),
            fallback_count,
        );
        encoder = self
            .global_bucket_offsets
            .encode(encoder, self.global_bucket_data.view());

        let keys_a = keys;
        let keys_b = temporary_key_storage;

        let values_a = values;
        let values_b = temporary_value_storage;

        for i in 0..radix_groups {
            if (i & 1) == 0 {
                encoder = self.bucket_scatter_by.encode(
                    encoder,
                    BucketScatterByInput {
                        keys_in: keys_a,
                        keys_out: keys_b,
                        values_in: values_a,
                        values_out: values_b,
                        global_base_bucket_offsets: self.global_bucket_data.view(),
                        radix_group: i as u32,
                        max_count: count.uniform(),
                        dispatch_indirect,
                        dispatch: self.scatter_dispatch.view(),
                        fallback_count,
                    },
                );
            } else {
                encoder = self.bucket_scatter_by.encode(
                    encoder,
                    BucketScatterByInput {
                        keys_in: keys_b,
                        keys_out: keys_a,
                        values_in: values_b,
                        values_out: values_a,
                        global_base_bucket_offsets: self.global_bucket_data.view(),
                        radix_group: i as u32,
                        max_count: count.uniform(),
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

impl<V> RadixSortBy<u32, V>
where
    V: abi::Sized + 'static,
{
    pub async fn init_u32(device: Device) -> Self {
        let global_bucket_data =
            device.create_buffer_zeroed(buffer::Usages::storage_binding().and_copy_dst());

        let (generate_dispatches, bucket_histogram, global_bucket_offsets, bucket_scatter_by) =
            join!(
                GenerateDispatches::init(device.clone()),
                BucketHistogram::init_u32(device.clone()),
                GlobalBucketOffsets::init(device.clone()),
                BucketScatterBy::init_u32(device.clone()),
            )
            .await;

        let segment_sizes = device.create_buffer(
            SegmentSizes {
                histogram: BUCKET_HISTOGRAM_SEGMENT_SIZE,
                scatter: BUCKET_SCATTER_BY_SEGMENT_SIZE,
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

        RadixSortBy {
            device,
            generate_dispatches,
            bucket_histogram,
            global_bucket_offsets,
            bucket_scatter_by,
            global_bucket_data,
            segment_sizes,
            histogram_dispatch,
            scatter_dispatch,
        }
    }

    pub fn encode_half_precision<U0, U1, U2, U3>(
        &mut self,
        encoder: CommandEncoder,
        input: RadixSortByInput<u32, V, U0, U1, U2, U3>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
    {
        self.encode_internal(encoder, input, 2)
    }
}
