#pragma once

#include <cinttypes>
#include <device_launch_parameters.h>

extern "C"
{
    constexpr uint64_t PCG32_MAGIC = 0x5851F42D4C957F2DULL; // 6364136223846793005ULL;

    struct pcg32_random_t {         // Internals are *Private*.
        uint64_t state;             // RNG state.  All values are possible.
        uint64_t inc;               // Controls which RNG sequence (stream) is
                                    // selected. Must *always* be odd.
    };

    __host__ __device__ inline uint32_t pcg32_xsr(uint64_t state)
    {
        uint32_t result;
        uint32_t xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
        uint32_t rot = state >> 59u;
        result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        return result;
    }

    __host__ __device__ inline uint32_t pcg32_random_r(pcg32_random_t* rng)
    {
        uint64_t oldstate = rng->state;
        rng->state = oldstate * PCG32_MAGIC + rng->inc;
        return pcg32_xsr(oldstate);
    }

    __host__ inline void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
    {
        rng->state = 0U;
        rng->inc = (initseq << 1u) | 1u;
        pcg32_random_r(rng);
        rng->state += initstate;
        pcg32_random_r(rng);
    }
}
