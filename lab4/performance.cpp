#include "performance.h"
#include <memory>
#include <thread>
#include "num_threads.h"
#include "randomize.h"
#include "vector_mod.h"

std::vector<measurement> run_experiments()
{
	//constexpr std::size_t word_count = (std::size_t(1) << 31) / sizeof(IntegerWord);
	constexpr std::size_t word_count = (std::size_t(1) << 30) / sizeof(IntegerWord);

	constexpr IntegerWord divisor = INTWORD_MAX;
    const size_t thread_count = std::thread::hardware_concurrency();
	auto data = std::make_unique<IntegerWord[]>(word_count);
	std::vector<measurement> results;
	randomize(data.get(), word_count * sizeof(IntegerWord));
	results.reserve(thread_count);
	for (unsigned T = 1; T <= thread_count; ++T)
	{
		set_num_threads(T);
		using namespace std::chrono;
		auto tm0 = steady_clock::now();
		auto result = vector_mod(data.get(), word_count, divisor);
		auto time = duration_cast<milliseconds>(steady_clock::now() - tm0);
		results.emplace_back(measurement{result, time});
	}
	return results;
}