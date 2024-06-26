#include <fstream>
#include <iomanip>
#include <sstream>

#include "indent.h"
#include "benchmarks.h"
#include "cuda_host_helper.h"

using namespace std;

#define ARR_SIZE(arr) (sizeof(arr) / sizeof(*arr))
#define ARR_LAST(arr) (arr[ARR_SIZE(arr)-1])
#define ARR_FIRST(arr) (*arr)

constexpr uint64_t seed_states[] = { 42, 2809200318112002, 0xAAAABBBBCCCCDDDDULL, };
constexpr uint64_t seed_increments[] = { 54, 1811200228092003, 0xCCCCBBBBAAAADDDDULL, };
constexpr unsigned threads_per_block[] = { 64, 128, 256, 512, }; // only powers of 2

constexpr size_t seed_combinations = ARR_SIZE(seed_states) * ARR_SIZE(seed_increments);

struct _config
{
	static constexpr int numbers_hard_min = 12;
	static constexpr int numbers_hard_max = 32;
	static constexpr int numbers_default_step = 2;
	static constexpr int threads_hard_min = 10; 
	static constexpr int threads_hard_max = 30;
	static constexpr int threads_default_step = 2;

	const char* series_name = "";
	inline string throughput_filename() { return ((string("benchmark_throughput") + ('\0' == *series_name ? "" : "_")) + series_name) + ".csv"; }
	inline string multilinear_filename() { return ((string("benchmark_multilinear") + ('\0' == *series_name ? "" : "_")) + series_name) + ".csv"; }

	bool append = true;
	bool checks = false;
	bool headers = true;

	int numbers_min_log2 = numbers_hard_min;
	int numbers_max_log2 = numbers_hard_max;
	int numbers_step_log2 = numbers_default_step;
	inline size_t test_arr_size() const { return 1ULL << numbers_max_log2; }
	inline size_t test_arr_size_in_bytes() const { return test_arr_size() * sizeof(uint32_t); }

	int threads_min_log2 = threads_hard_min;
	int threads_max_log2 = threads_hard_max;
	int threads_step_log2 = threads_default_step;
} config;

void run_gpu(statistics(*benchmark)(const parameters&), bool(*is_correct)(parameters, const statistics&, uint32_t*), parameters params, statistics& stats, uint32_t* host_arr)
{
	for (int s = 0; s < ARR_SIZE(seed_states); s++)
	{
		for (int i = 0; i < ARR_SIZE(seed_increments); i++)
		{
			pcg32_srandom_r(&params.seed, seed_states[s], seed_increments[i]);

			if (config.checks)
				CTE(cudaMemset(params.dev_arr, 0, params.numbers * sizeof(uint32_t)));
			auto now = benchmark(params);
			if (config.checks)
			{
				CTE(cudaMemcpy(host_arr, params.dev_arr, params.numbers * sizeof(uint32_t), cudaMemcpyDeviceToHost));
				stats.correct = stats.correct && is_correct(params, now, host_arr);
			}
			stats.calculation_time += now.calculation_time;
			if (stats.blocks == 0)
				stats.blocks = now.blocks;
			else if (stats.blocks != now.blocks)
				throw message_exception("Block counts of different runs with the same parameters don't match");
		}
	}
}

bool is_correct_throughput(parameters params, const statistics& stats, uint32_t* host_arr)
{
	for (size_t i = 0; i < params.numbers; i++)
		if (host_arr[i] != (uint32_t)(i * params.seed.state + params.seed.inc))
			return false;
	return true;
}

bool is_correct_pcg32(parameters params, const statistics& stats, uint32_t* host_arr)
{
	for (size_t i = 0; i < params.numbers; i++)
	{
		auto generated = host_arr[i];
		auto expected = pcg32_random_r(&params.seed);
		if (generated != expected)
			return false;
	}
	auto next_correct = params.seed.state == stats.next_seed.state;
	if (!next_correct)
		return false;
	return true;
}

void headers(ofstream& file)
{
	file << "Method name;Numbers generated;Blocks per grid;Threads per block;Execution time [ms];Results correct" << endl;
}

void save(ofstream& file, const string& name, const parameters& params, const statistics& stats)
{
	file << name << ';' << params.numbers << ';' << stats.blocks << ';' << params.threads_per_block << ';';
	file << fixed << setprecision(3) << stats.calculation_time << ';' << (config.checks ? (stats.correct ? "Yes" : "No") : "Skipped") << endl;
}

void open_as_configured(ofstream& file, const string& filename)
{
	bool exists = ifstream(filename, ios::in | ios::binary).is_open();
	file.open(filename, ios::in | ios::out | (config.append ? ios::app : ios::trunc));
	if (!file.is_open())
		throw message_exception("Could not open file" + filename);
	if (config.headers && !(exists && config.append))
		headers(file);
}

bool streq(const char* const s1, const char* const s2)
{
	return strcmp(s1, s2) == 0;
}

bool strneq(const char* const s1, const char* const s2, size_t n)
{
	return strncmp(s1, s2, n) == 0;
}

void usage(ostream& os, const char* const program_name, int exit_code)
{
	os << "USAGE:\n"
		<< program_name << " [OPTIONS]\n"
		<< "\t--checks      \tCheck results correctness with CPU-generated numbers (very slow)\n"
		<< "\t-h|--help     \tShow this message and exit\n"
		<< "\t-o|--overwrite\tIf results file already exists overwrite it instead of appending\n"
		<< "\t--no-headers  \tDon't add column names as first line when creating results file\n"
		<< "\t--numbers-min \tSet N, where 2^N is the amount of numbers the first test will generate\n"
		<< "\t--numbers-max \tSet M, where 2^M is the amount of numbers the last test will generate\n"
		<< "\t--numbers-step\tSet S, where 2^S is the ratio between amount of numbers generated by consecutive tests\n"
		<< "\t  for numbers:\t" << config.numbers_hard_min << " <= N <= M <= " << config.numbers_hard_max << ", default S = " << config.numbers_default_step << "\n"
		<< "\t--series-name \tArbitrary string that will become part of results files' names\n"
		<< "\t--threads-min \tSet N, where 2^N is the minimal amount of threads in a test kernel's grid\n"
		<< "\t--threads-max \tSet M, where 2^M is the maximal amount of threads in a test kernel's grid\n"
		<< "\t--threads-step\tSet S, where 2^S is ratio between amount of threads in consecutive test's runs\n"
		<< "\t  for threads:\t" << config.threads_hard_min << " <= N <= M <= " << config.threads_hard_max << ", default S = " << config.threads_default_step << "\n"
		<< "EXAMPLE:\n"
		<< program_name << "--num-max 28 --thread-min 12 --test-name my_shorter_test -o"
		<< endl;

	exit(exit_code);
}

int arg_to_int(const char* opt, const char* arg, int min, int max)
{
	int res = stoi(arg);
	if (res < min || res > max)
	{
		ostringstream ss;
		ss << "Invalid argument '" << arg << "' for option '" << opt << "'";
		throw message_exception(ss.str());
	}
	return res;
}

int throwing_main(int argc, const char* argv[])
{
	ofstream file;
	indent tabs("  ");
	parameters params;
	uint32_t* host_arr = nullptr;

	cout << "Parsing arguments..." << endl;
	for (int i = 1; i < argc; i++)
	{
		if (streq(argv[i], "--checks"))
			config.checks = true;
		else if (streq(argv[i], "-h") || streq(argv[1], "--help"))
			usage(cout, argv[0], 0);
		else if (streq(argv[i], "-o") || streq(argv[i], "--overwrite"))
			config.append = false;
		else if (streq(argv[i], "--no-headers"))
			config.headers = false;
		else if (streq(argv[i], "--numbers-min"))
		{
			config.numbers_min_log2 = arg_to_int(argv[i], argv[i + 1], config.numbers_hard_min, config.numbers_hard_max);
			i++;
		}
		else if (streq(argv[i], "--numbers-max"))
		{
			config.numbers_max_log2 = arg_to_int(argv[i], argv[i + 1], config.numbers_hard_min, config.numbers_hard_max);
			i++;
		}
		else if (streq(argv[i], "--numbers-step"))
		{
			config.numbers_step_log2 = arg_to_int(argv[i], argv[i + 1], 1, config.numbers_hard_max);
			i++;
		}
		else if (streq(argv[i], "--threads-min"))
		{
			config.threads_min_log2 = arg_to_int(argv[i], argv[i + 1], config.threads_hard_min, config.threads_hard_max);
			i++;
		}
		else if (streq(argv[i], "--threads-max"))
		{
			config.threads_max_log2 = arg_to_int(argv[i], argv[i + 1], config.threads_hard_min, config.threads_hard_max);
			i++;
		}
		else if (streq(argv[i], "--threads-step"))
		{
			config.threads_step_log2 = arg_to_int(argv[i], argv[i + 1], 1, config.threads_hard_max);
			i++;
		}
		else if (streq(argv[i], "--series-name"))
		{
			config.series_name = argv[i + 1];
			i++;
		}
		else if (*(argv[i]) == '-')
			throw message_exception((string("Invalid option '") + argv[i]) + "'");
		else
			throw message_exception((string("Unexpected argument '") + argv[i]) + "'");
	}

	tabs++;
	cout << tabs << "Running with following config:" << endl;
	tabs++;
	cout << tabs << ('\0' == *config.series_name ? "Unnamed benchmark series" : "Benchmark series name: ") << config.series_name << endl;
	cout << tabs << "Numbers min:step:max (log2): " << config.numbers_min_log2 << ':' << config.numbers_step_log2 << ':' << config.numbers_max_log2 << endl;
	cout << tabs << "Threads min:step:max (log2): " << config.threads_min_log2 << ':' << config.threads_step_log2 << ':' << config.threads_max_log2 << endl;
	cout << tabs << "Append results if already exist: " << (config.append ? "true" : "false") << endl;
	cout << tabs << "Headers on first line: " << (config.headers ? "true" : "false") << endl;
	tabs.level = 0;

	cout << "Allocating memory..." << endl;
	bool cuda_malloc_host = false;
	if (config.checks && !(cuda_malloc_host = (cudaMallocHost(&host_arr, config.test_arr_size_in_bytes()) == cudaSuccess)))
	{
		cout << "WARNING: Unable to allocate pinned memory, using normal memory" << endl;
		if (!(host_arr = (uint32_t*)malloc(config.test_arr_size_in_bytes())))
			throw message_exception("Unable to allocate host memory");
	}
	CTE(cudaMalloc(&params.dev_arr, config.test_arr_size_in_bytes()));

	cout << "Measuring throughput..." << endl;
	tabs++;
	bool warmup = true;
	open_as_configured(file, config.throughput_filename());
	for (int n = config.numbers_min_log2; n <= config.numbers_max_log2; n += config.numbers_step_log2)
	{
		params.numbers = 1ULL << n;
		cout << tabs++ << "Generating 2^" << n << " numbers..." << endl;
		for (int t = 0; t < ARR_SIZE(threads_per_block); t++)
		{
			params.threads_per_block = threads_per_block[t];
			cout << tabs << "With " << params.threads_per_block << " threads per block..." << endl;

			if (warmup)
			{
				benchmark_throughput_vec4(params);
				benchmark_throughput_vec2(params);
				benchmark_throughput_scal(params);
				warmup = false;
			}

			statistics scalar, vectorized2, vectorized4;
			switch (params.numbers / params.threads_per_block) // result is a power of 2
			{
			default: // 4 or more
				run_gpu(benchmark_throughput_vec4, is_correct_throughput, params, vectorized4, host_arr);
				vectorized4.calculation_time /= seed_combinations;
				save(file, "Throughput Vec4", params, vectorized4);
			case 2:
				run_gpu(benchmark_throughput_vec2, is_correct_throughput, params, vectorized2, host_arr);
				vectorized2.calculation_time /= seed_combinations;
				save(file, "Throughput Vec2", params, vectorized2);
			case 1:
				run_gpu(benchmark_throughput_scal, is_correct_throughput, params, scalar, host_arr);
				scalar.calculation_time /= seed_combinations;
				save(file, "Throughput Scal", params, scalar);
			case 0:
				break;
			}
		}
		tabs--;
	}
	file.close();
	tabs--;

	cout << "Initializing PCG32 constants..." << endl;
	initialize_constants();

	cout << "Performing multi-linear benchmarks..." << endl;
	tabs++;
	warmup = true;
	open_as_configured(file, config.multilinear_filename());
	for (int n = config.numbers_min_log2; n <= config.numbers_max_log2; n += config.numbers_step_log2)
	{
		params.numbers = 1ULL << n;
		cout << tabs++ << "Generating 2^" << n << " numbers..." << endl;
		for (int tpb = 0; tpb < ARR_SIZE(threads_per_block); tpb++)
		{
			params.threads_per_block = threads_per_block[tpb];
			cout << tabs++ << "With " << params.threads_per_block << " threads per block..." << endl;
			for (int tt = config.threads_min_log2; tt <= config.threads_max_log2 && tt <= n - 2; tt += config.threads_step_log2)
			{
				params.total_threads = 1ULL << tt;
				if (params.threads_per_block > params.total_threads)
					continue;
				cout << tabs << "With 2^" << tt << " threads in total..." << endl;

				if (warmup)
				{
					benchmark_multilinear_vec4(params);
					benchmark_multilinear_vec2(params);
					benchmark_multilinear_scal(params);
					warmup = false;
				}

				statistics scalar, vectorized2, vectorized4;
				switch (params.numbers / params.threads_per_block) // result is a power of 2
				{
				default: // 4 or more
					run_gpu(benchmark_multilinear_vec4, is_correct_pcg32, params, vectorized4, host_arr);
					vectorized4.calculation_time /= seed_combinations;
					save(file, "Multilinear Vec4", params, vectorized4);
				case 2:
					run_gpu(benchmark_multilinear_vec2, is_correct_pcg32, params, vectorized2, host_arr);
					vectorized2.calculation_time /= seed_combinations;
					save(file, "Multilinear Vec2", params, vectorized2);
				case 1:
					run_gpu(benchmark_multilinear_scal, is_correct_pcg32, params, scalar, host_arr);
					scalar.calculation_time /= seed_combinations;
					save(file, "Multilinear Scal", params, scalar);
				case 0:
					break;
				}
			}
			tabs--;
		}
		tabs--;
	}
	file.close();
	tabs--;

	cout << "Freeing memory..." << endl;
	CTE(cudaFree(params.dev_arr));
	if (config.checks)
	{
		if (cuda_malloc_host)
			{ CTE(cudaFreeHost(host_arr)); }
		else
			free(host_arr);
	}
	return 0;
}

int main(int argc, const char* argv[])
{
	try
	{
		throwing_main(argc, argv);
	}
	catch (exception& e)
	{
		cerr << "Exception: " << e.what() << endl;
		return 1;
	}
	return 0;
}
