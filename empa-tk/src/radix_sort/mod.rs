use empa::buffer::Buffer;
use empa::command::CommandEncoder;
use empa::device::Device;
use empa::type_flag::{O, X};
use empa::{abi, buffer};

use crate::radix_sort::bucket_histogram::{BucketHistogram, BucketHistogramInput};
use crate::radix_sort::bucket_scatter::{BucketScatter, BucketScatterInput};
use crate::radix_sort::global_bucket_offsets::GlobalBucketOffsets;
use zeroable::Zeroable;

mod bucket_histogram;
mod bucket_scatter;
mod global_bucket_offsets;

const RADIX_SIZE: u32 = 8;
const RADIX_DIGITS: usize = 256;
const RADIX_GROUPS: usize = 4;

pub struct RadixSortInput<'a, T, U0, U1> {
    pub data: buffer::View<'a, [T], U0>,
    pub temporary_storage: buffer::View<'a, [T], U1>,
}

pub struct RadixSort<T>
where
    T: abi::Sized,
{
    bucket_histogram: BucketHistogram<T>,
    global_bucket_offsets: GlobalBucketOffsets,
    bucket_scatter: BucketScatter<T>,
    global_bucket_data:
        Buffer<[[u32; RADIX_DIGITS]; RADIX_GROUPS], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
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
        let RadixSortInput {
            data,
            temporary_storage,
        } = input;

        assert_eq!(
            data.len(),
            temporary_storage.len(),
            "`data` and `temporary_storage` must have the same length"
        );
        assert!(
            data.len() < (1 << 30),
            "`data` length must be less than 2^30 (1073741824)"
        );

        encoder = encoder.clear_buffer(self.global_bucket_data.view());
        encoder = self.bucket_histogram.encode(
            encoder,
            BucketHistogramInput {
                data,
                global_histograms: self.global_bucket_data.view(),
            },
        );
        encoder = self
            .global_bucket_offsets
            .encode(encoder, self.global_bucket_data.view());

        let data_a = data;
        let data_b = temporary_storage;

        for i in 0..RADIX_GROUPS {
            if (i & 1) == 0 {
                encoder = self.bucket_scatter.encode(
                    encoder,
                    BucketScatterInput {
                        data_in: data_a,
                        data_out: data_b,
                        global_base_bucket_offsets: self.global_bucket_data.view(),
                        radix_group: i as u32,
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
                    },
                );
            }
        }

        encoder
    }
}

impl RadixSort<u32> {
    pub fn init_u32(device: Device) -> Self {
        let global_bucket_data = device.create_buffer_zeroed(
            buffer::Usages::storage_binding()
                .and_copy_dst(),
        );

        let bucket_histogram = BucketHistogram::init_u32(device.clone());
        let global_bucket_offsets = GlobalBucketOffsets::init(device.clone());
        let bucket_scatter = BucketScatter::init_u32(device);

        RadixSort {
            bucket_histogram,
            global_bucket_offsets,
            bucket_scatter,
            global_bucket_data,
        }
    }
}
