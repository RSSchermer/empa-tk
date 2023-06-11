use empa::buffer::Buffer;
use empa::command::CommandEncoder;
use empa::device::Device;
use empa::type_flag::{O, X};
use empa::{abi, buffer};

use crate::radix_sort::bucket_histogram::{BucketHistogram, BucketHistogramInput};
use crate::radix_sort::bucket_scatter_by::{BucketScatterBy, BucketScatterByInput};
use crate::radix_sort::global_bucket_offsets::GlobalBucketOffsets;
use crate::radix_sort::{RADIX_DIGITS, RADIX_GROUPS};

pub struct RadixSortByInput<'a, K, V, U0, U1, U2, U3> {
    pub keys: buffer::View<'a, [K], U0>,
    pub values: buffer::View<'a, [V], U1>,
    pub temporary_key_storage: buffer::View<'a, [K], U2>,
    pub temporary_value_storage: buffer::View<'a, [V], U3>,
}

pub struct RadixSortBy<K, V>
where
    K: abi::Sized,
    V: abi::Sized,
{
    bucket_histogram: BucketHistogram<K>,
    global_bucket_offsets: GlobalBucketOffsets,
    bucket_scatter_by: BucketScatterBy<K, V>,
    global_bucket_data:
        Buffer<[[u32; RADIX_DIGITS]; RADIX_GROUPS], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
}

impl<K, V> RadixSortBy<K, V>
where
    K: abi::Sized,
    V: abi::Sized,
{
    pub fn encode<U0, U1, U2, U3>(
        &mut self,
        mut encoder: CommandEncoder,
        input: RadixSortByInput<K, V, U0, U1, U2, U3>,
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
        } = input;

        assert_eq!(
            keys.len(),
            values.len(),
            "`keys` and `values` must have the same length"
        );
        assert_eq!(
            keys.len(),
            temporary_key_storage.len(),
            "`keys` and `temporary_key_storage` must have the same length"
        );
        assert_eq!(
            values.len(),
            temporary_value_storage.len(),
            "`values` and `temporary_value_storage` must have the same length"
        );
        assert!(
            keys.len() < (1 << 30),
            "data length must be less than 2^30 (1073741824)"
        );

        encoder = encoder.clear_buffer(self.global_bucket_data.view());
        encoder = self.bucket_histogram.encode(
            encoder,
            BucketHistogramInput {
                data: keys,
                global_histograms: self.global_bucket_data.view(),
            },
        );
        encoder = self
            .global_bucket_offsets
            .encode(encoder, self.global_bucket_data.view());

        let keys_a = keys;
        let keys_b = temporary_key_storage;

        let values_a = values;
        let values_b = temporary_value_storage;

        for i in 0..RADIX_GROUPS {
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
                    },
                );
            }
        }

        encoder
    }
}

impl<V> RadixSortBy<u32, V>
where
    V: abi::Sized,
{
    pub fn init_u32(device: Device) -> Self {
        let global_bucket_data =
            device.create_buffer_zeroed(buffer::Usages::storage_binding().and_copy_dst());

        let bucket_histogram = BucketHistogram::init_u32(device.clone());
        let global_bucket_offsets = GlobalBucketOffsets::init(device.clone());
        let bucket_scatter_by = BucketScatterBy::init_u32(device);

        RadixSortBy {
            bucket_histogram,
            global_bucket_offsets,
            bucket_scatter_by,
            global_bucket_data,
        }
    }
}
