#pragma once

#include "pcg32_minimal.h"

struct parameters
{
	uint32_t* dev_arr;
	pcg32_random_t seed;
	size_t numbers;
	unsigned threads_per_block;
	size_t total_threads;
};

struct statistics
{
	pcg32_random_t next_seed;
	float calculation_time;
	size_t blocks;
	bool correct;

	statistics() : next_seed(), calculation_time(0), blocks(0), correct(true)
	{ next_seed.state = 0; next_seed.inc = 0; }
};

void initialize_constants();

statistics benchmark_throughput(const parameters& params);
statistics benchmark_throughput_vec(const parameters& params);
statistics benchmark_multilinear(const parameters& params);
statistics benchmark_multilinear_vec(const parameters& params);
