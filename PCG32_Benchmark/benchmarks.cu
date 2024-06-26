#include "benchmarks.h"

#include "cuda_timer.h"
#include "integer_math_helper.h"

__device__ uint64_t dev_next_state;
__constant__ uint64_t dev_const_mul[64];
__constant__ uint64_t dev_const_inc[64];

__global__ void throughput_scal(uint32_t* const dev_arr, pcg32_random_t seed);
__global__ void throughput_vec2(uint32_t* const dev_arr, pcg32_random_t seed);
__global__ void throughput_vec4(uint32_t* const dev_arr, pcg32_random_t seed);
__global__ void multilinear_scal(uint32_t* const dev_arr, pcg32_random_t seed, const unsigned threads_log2, const size_t numbers);
__global__ void multilinear_vec2(uint32_t* const dev_arr, pcg32_random_t seed, const unsigned threads_log2, const size_t numbers);
__global__ void multilinear_vec4(uint32_t* const dev_arr, pcg32_random_t seed, const unsigned threads_log2, const size_t numbers);

void initialize_constants()
{
	uint64_t muls[64];
	muls[0] = PCG32_MAGIC;
	for (int i = 1; i < 64; i++)
		muls[i] = muls[i - 1] * muls[i - 1];

	uint64_t incs[64];
	incs[0] = 1;
	for (int i = 1; i < 64; i++)
		incs[i] = (muls[i - 1] + 1) * incs[i - 1];

	CTE(cudaMemcpyToSymbol(dev_const_mul, muls, 64 * sizeof(uint64_t)));
	CTE(cudaMemcpyToSymbol(dev_const_inc, incs, 64 * sizeof(uint64_t)));
}

statistics benchmark_throughput_scal(const parameters& params)
{
	cuda_timer timer;
    statistics stats;
    stats.blocks = params.numbers / params.threads_per_block;
	timer.start();
    throughput_scal<<<stats.blocks, params.threads_per_block>>>(params.dev_arr, params.seed);
	CTE(cudaGetLastError());
	timer.end();
	stats.calculation_time = timer.wait_and_get_elapsed_time();
	return stats;
}
statistics benchmark_throughput_vec2(const parameters& params)
{
	cuda_timer timer;
	statistics stats;
	stats.blocks = (params.numbers / params.threads_per_block) / 2;
	timer.start();
	throughput_vec2<<<stats.blocks, params.threads_per_block>>>(params.dev_arr, params.seed);
	CTE(cudaGetLastError());
	timer.end();
	stats.calculation_time = timer.wait_and_get_elapsed_time();
	return stats;
}
statistics benchmark_throughput_vec4(const parameters& params)
{
	cuda_timer timer;
	statistics stats;
	stats.blocks = (params.numbers / params.threads_per_block) / 4;
	timer.start();
	throughput_vec4<<<stats.blocks, params.threads_per_block>>>(params.dev_arr, params.seed);
	CTE(cudaGetLastError());
	timer.end();
	stats.calculation_time = timer.wait_and_get_elapsed_time();
	return stats;
}
statistics benchmark_multilinear_scal(const parameters& params)
{
	cuda_timer timer;
	statistics stats;
	stats.blocks = params.total_threads / params.threads_per_block;
	timer.start();
	multilinear_scal<<<stats.blocks, params.threads_per_block>>>(params.dev_arr, params.seed, bfind(params.total_threads), params.numbers);
	CTE(cudaGetLastError());
	timer.end();
	stats.calculation_time = timer.wait_and_get_elapsed_time();
	CTE(cudaMemcpyFromSymbol(&stats.next_seed.state, dev_next_state, sizeof(dev_next_state)));
	stats.next_seed.inc = params.seed.inc;
	return stats;
}
statistics benchmark_multilinear_vec2(const parameters& params)
{
	cuda_timer timer;
	statistics stats;
	stats.blocks = params.total_threads / params.threads_per_block;
	timer.start();
	multilinear_vec2<<<stats.blocks, params.threads_per_block>>>(params.dev_arr, params.seed, bfind(params.total_threads), params.numbers);
	CTE(cudaGetLastError());
	timer.end();
	stats.calculation_time = timer.wait_and_get_elapsed_time();
	CTE(cudaMemcpyFromSymbol(&stats.next_seed.state, dev_next_state, sizeof(dev_next_state)));
	stats.next_seed.inc = params.seed.inc;
	return stats;
}
statistics benchmark_multilinear_vec4(const parameters& params)
{
	cuda_timer timer;
	statistics stats;
	stats.blocks = params.total_threads / params.threads_per_block;
	timer.start();
	multilinear_vec4<<<stats.blocks, params.threads_per_block>>>(params.dev_arr, params.seed, bfind(params.total_threads), params.numbers);
	CTE(cudaGetLastError());
	timer.end();
	stats.calculation_time = timer.wait_and_get_elapsed_time();
	CTE(cudaMemcpyFromSymbol(&stats.next_seed.state, dev_next_state, sizeof(dev_next_state)));
	stats.next_seed.inc = params.seed.inc;
	return stats;
}

__global__ void throughput_scal(uint32_t* const dev_arr, pcg32_random_t seed)
{
	unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
	dev_arr[index] = (uint32_t)(index * seed.state + seed.inc);
}
__global__ void throughput_vec2(uint32_t* const dev_arr, pcg32_random_t seed)
{
	unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned i = index * 2;
	uint2 res;
	res.x = (uint32_t)((i + 0) * seed.state + seed.inc);
	res.y = (uint32_t)((i + 1) * seed.state + seed.inc);
	((uint2*)dev_arr)[index] = res;
}
__global__ void throughput_vec4(uint32_t* const dev_arr, pcg32_random_t seed)
{
	unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned i = index * 4;
	uint4 res;
	res.x = (uint32_t)((i + 0) * seed.state + seed.inc);
	res.y = (uint32_t)((i + 1) * seed.state + seed.inc);
	res.z = (uint32_t)((i + 2) * seed.state + seed.inc);
	res.w = (uint32_t)((i + 3) * seed.state + seed.inc);
	((uint4*)dev_arr)[index] = res;
}

__global__ void multilinear_scal(uint32_t* const dev_arr, pcg32_random_t seed, const unsigned threads_log2, const size_t numbers)
{
	int64_t index = threadIdx.x + blockIdx.x * (int64_t)blockDim.x;
	{
		uint64_t acc_mult = seed.state;
		uint64_t acc_plus = 0;
		int64_t exp = index;
		for (int it = 0; exp != 0; it++, exp >>= 1)
		{
			if (exp & 1)
			{
				acc_mult = acc_mult * dev_const_mul[it];
				acc_plus = acc_plus * dev_const_mul[it] + dev_const_inc[it];
			}
		}
		seed.state = acc_mult + seed.inc * acc_plus;
	}
	const int step = 1U << threads_log2;
	while (index < numbers)
	{
		dev_arr[index] = pcg32_xsr(seed.state);
		seed.state = seed.state * dev_const_mul[threads_log2] + seed.inc * dev_const_inc[threads_log2];
		index += step;
	}
	if (index == numbers)
		dev_next_state = seed.state;
}
__global__ void multilinear_vec2(uint32_t* const dev_arr, pcg32_random_t seed, const unsigned threads_log2, const size_t numbers)
{
	int64_t index = threadIdx.x + blockIdx.x * (int64_t)blockDim.x;
	{
		uint64_t acc_mult = seed.state;
		uint64_t acc_plus = 0;
		int64_t exp = index;
		for (int it = 1; exp != 0; it++, exp >>= 1)
		{
			if (exp & 1)
			{
				acc_mult = acc_mult * dev_const_mul[it];
				acc_plus = acc_plus * dev_const_mul[it] + dev_const_inc[it];
			}
		}
		seed.state = acc_mult + seed.inc * acc_plus;
	}
	const int step = 1U << threads_log2;
	while (index * 2 < numbers)
	{
		uint2 res;
		uint64_t temp = seed.state;
		res.x = pcg32_xsr(temp);
		temp = temp * PCG32_MAGIC + seed.inc;
		res.y = pcg32_xsr(temp);
		((uint2*)dev_arr)[index] = res;
		seed.state = seed.state * dev_const_mul[threads_log2 + 1] + seed.inc * dev_const_inc[threads_log2 + 1];
		index += step;
	}
	if (index * 2 == numbers)
		dev_next_state = seed.state;
}
__global__ void multilinear_vec4(uint32_t* const dev_arr, pcg32_random_t seed, const unsigned threads_log2, const size_t numbers)
{
	int64_t index = threadIdx.x + blockIdx.x * (int64_t)blockDim.x;
	{
		uint64_t acc_mult = seed.state;
		uint64_t acc_plus = 0;
		int64_t exp = index;
		for (int it = 2; exp != 0; it++, exp >>= 1)
		{
			if (exp & 1)
			{
				acc_mult = acc_mult * dev_const_mul[it];
				acc_plus = acc_plus * dev_const_mul[it] + dev_const_inc[it];
			}
		}
		seed.state = acc_mult + seed.inc * acc_plus;
	}
	const int step = 1U << threads_log2;
	while (index * 4 < numbers)
	{
		uint4 res;
		uint64_t temp = seed.state;
		res.x = pcg32_xsr(temp);
		temp = temp * PCG32_MAGIC + seed.inc;
		res.y = pcg32_xsr(temp);
		temp = temp * PCG32_MAGIC + seed.inc;
		res.z = pcg32_xsr(temp);
		temp = temp * PCG32_MAGIC + seed.inc;
		res.w = pcg32_xsr(temp);
		((uint4*)dev_arr)[index] = res;
		seed.state = seed.state * dev_const_mul[threads_log2 + 2] + seed.inc * dev_const_inc[threads_log2 + 2];
		index += step;
	}
	if (index * 4 == numbers)
		dev_next_state = seed.state;
}
