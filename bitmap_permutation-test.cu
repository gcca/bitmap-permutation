#include <gtest/gtest.h>

#include <cstdint>

__device__ std::uint8_t bitmask(const std::uint8_t *bitmap,
                                const std::size_t bitmap_offset)
{
  const std::size_t bitmap_index = bitmap_offset / 8;
  const std::size_t bit_offset = bitmap_offset % 8;

  return bitmap[bitmap_index] & (128 >> bit_offset) ? 1 << bit_offset : 0;
}


__global__ void bitmap_permutation(const std::uint8_t *bitmap,
                                   const std::size_t *map,
                                   const std::size_t map_size,
                                   std::uint8_t *permutation)
{
  const std::size_t chunk_start = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  std::uint8_t result = 0;

  for (std::size_t i = 0; i < 8; i++)
    {
      result ^= bitmask(bitmap, map[chunk_start + i]);
    }

  permutation[chunk_start] = result;
}


TEST(bitmapPermutationTest, FirstAttempt)
{
  std::uint8_t *bitmap;  // TODO: (int) type can be a parameter
  std::uint8_t *permutation;
  cudaMallocManaged(&bitmap, 2 * sizeof(std::uint8_t));
  cudaMallocManaged(&permutation, 2 * sizeof(std::uint8_t));

  bitmap[0] = 0b01010101;
  bitmap[1] = 0b01010101;

  permutation[0] = 0;
  permutation[1] = 0;


  std::size_t *map;  // TODO: type as parameter for indexing
  cudaMallocManaged(&map, 10 * sizeof(std::size_t));

  const std::size_t input_map[] = {7, 6, 5, 4, 3, 2, 1, 0, 15, 14};

  for (std::size_t i = 0; i < 10; i++)
    {
      map[i] = input_map[i];
    }

  // TODO: threads multiple of 8
  bitmap_permutation<<<1, 2>>>(bitmap, map, 10, permutation);

  cudaDeviceSynchronize();

  EXPECT_EQ(0b10101010, permutation[0]);
  EXPECT_EQ(0b10000000, permutation[1]);

  cudaFree(bitmap);
  cudaFree(permutation);

  cudaFree(map);
}
