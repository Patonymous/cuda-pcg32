#pragma once

#include "cuda_host_helper.h"

class cuda_timer
{
private:
	cudaEvent_t _start;
	cudaEvent_t _end;

public:
	inline cuda_timer() : _start(), _end()
	{
		CTE(cudaEventCreate(&_start));
		CTE(cudaEventCreate(&_end));
	}
	inline ~cuda_timer()
	{
		cudaEventDestroy(_start);
		cudaEventDestroy(_end);
	}

	inline void start()
	{
		CTE(cudaEventRecord(_start));
	}
	inline void end()
	{
		CTE(cudaEventRecord(_end));
	}

	inline float wait_and_get_elapsed_time()
	{
		float time;
		CTE(cudaEventSynchronize(_end));
		CTE(cudaEventElapsedTime(&time, _start, _end));
		return time;
	}
};
