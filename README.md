# CUDA PCG32 

CUDA implementation of minimal PCG32 PRNG standard with benchmarks

## Compile

```
nvcc -O3 -std=c++11 main.cpp benchmarks.cu -o benchmark.exe
```
You may need to add: `--ccbin <path to x64 compiler directory>` or `-D_FORCE_INLINES`

Or simply use VisualStudio

## Run

```
./benchmark.exe [-h|--help|...]
```

## Warning! 

You may want to reduce the size of the benchmark by passing `--numbers-max N` with N less than the default 32.  
This is because an array contaning 2^32 4-byte numbers takes up 16GiB of device memory.

Especially when using `--checks` execution will take **MUCH** longer and also allocate the same amount of host memory as device memory,  
since after each kernel run results will be copied from device to host and CPU will check generated random numbers.  
Running benchmark with `--checks` but without reduced `--numbers-max` will probably take longer than you're willing to wait.

The program can always be stopped with Ctrl-C and benchmark results gathered up to that point should not be lost.
