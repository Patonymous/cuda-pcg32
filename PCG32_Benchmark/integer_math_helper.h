#pragma once

#include <cinttypes>
#include "device_launch_parameters.h"

// returns number of set bits (population count)
__host__ __device__ inline uint32_t popc(uint32_t val)
{
#if defined( __CUDA_ARCH__ )
	uint32_t res;
	asm("popc.u32 %0, %1;\n\t"
		: "=r" (res)
		: "r" (val)
	);
	return res;
#else // taken from http://aggregate.org/MAGIC
	val -= ((val >> 1) & 0x55555555);
	val = (((val >> 2) & 0x33333333) + (val & 0x33333333));
	val = (((val >> 4) + val) & 0x0f0f0f0f);
	val += (val >> 8);
	val += (val >> 16);
	return(val & 0x0000003f);
#endif
}
// returns number of set bits (population count)
__host__ __device__ inline uint32_t popc(uint64_t val)
{
#if defined( __CUDA_ARCH__ )
	uint32_t res;
	asm("popc.u64 %0, %1;\n\t"
		: "=r" (res)
		: "l" (val)
	);
	return res;
#else // adapted from http://aggregate.org/MAGIC
	val -= ((val >> 1) & 0x5555555555555555);
	val = (((val >> 2) & 0x3333333333333333) + (val & 0x3333333333333333));
	val = (((val >> 4) + val) & 0x0f0f0f0f0f0f0f0f);
	val = (((val >> 8) + val) & 0x00ff00ff00ff00ff);
	val += (val >> 16);
	val += (val >> 32);
	return(val & 0x0000000000000fff);
#endif
}
// returns index of highest set bit (effectively takes log of an integer)
__host__ __device__ inline uint32_t bfind(uint32_t val)
{
#if defined( __CUDA_ARCH__ )
	uint32_t res;
	asm("bfind.u32 %0, %1;\n\t"
		: "=r" (res)
		: "r" (val)
	);
	return res;
#else // taken from http://aggregate.org/MAGIC
	val |= (val >> 1);
	val |= (val >> 2);
	val |= (val >> 4);
	val |= (val >> 8);
	val |= (val >> 16);
	return(popc(val) - 1);
#endif
}
// returns index of highest set bit (effectively takes log of an integer)
__host__ __device__ inline uint32_t bfind(uint64_t val)
{
#if defined( __CUDA_ARCH__ )
	uint32_t res;
	asm("bfind.u64 %0, %1;\n\t"
		: "=r" (res)
		: "l" (val)
	);
	return res;
#else // adapted from http://aggregate.org/MAGIC
	val |= (val >> 1);
	val |= (val >> 2);
	val |= (val >> 4);
	val |= (val >> 8);
	val |= (val >> 16);
	val |= (val >> 32);
	return(popc(val) - 1);
#endif
}
