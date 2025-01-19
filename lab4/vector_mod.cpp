#include "vector_mod.h"
#include "mod_ops.h"
#include "num_threads.h"
#include <thread>
#include <vector>
#include <barrier>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <new>
#include <thread>

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
// 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │ ...
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

IntegerWord pow_mod(IntegerWord base, IntegerWord power, IntegerWord mod) {
	IntegerWord result = 1;
	while (power > 0) {
		if (power % 2 != 0) {
			result = mul_mod(result, base, mod);
		}
		power >>= 1;
		base = mul_mod(base, base, mod);
	}
	return result;
}

IntegerWord word_pow_mod(size_t power, IntegerWord mod) {
	return pow_mod((-mod) % mod, power, mod);
}

struct thread_range {
	std::size_t b, e;
};
thread_range vector_thread_range(size_t n, unsigned T, unsigned t) {
	auto b = n % T;
	auto s = n / T;
	if (t < b) b = ++s * t;
	else b += s * t;
	size_t e = b + s;
	return thread_range{ b, e };
}

struct partial_result_t {
	alignas(hardware_destructive_interference_size) IntegerWord value;
};

IntegerWord vector_mod(const IntegerWord* V, std::size_t N, IntegerWord mod) {
	size_t T = get_num_threads();
	std::vector<std::thread> threads(T - 1); //���������, ��� ������� ����� ��������� ������ � t=0
	std::vector<partial_result_t> partial_results(T);
	IntegerWord S = 0;
	std::barrier<> bar(T);

	auto thread_lambda = [V, N, T, mod, &partial_results, &bar](unsigned t) {
		auto [b, e] = vector_thread_range(N, T, t);

		IntegerWord sum = 0;
		// ����� �������
		for (std::size_t i = e; b < i;) {
			//sum = (sum * x + V[e-1-i]) % mod;
			sum = add_mod(times_word(sum, mod), V[--i], mod); // �� �� �����, �� ��� ������������
		}
		partial_results[t].value = sum;
		for (size_t i = 1, ii = 2; i < T; i = ii, ii += ii) {
//            std::cout << "1";
            bar.arrive_and_wait();
			if (t % ii == 0 && t + i < T) {
				auto neighbor = vector_thread_range(N, T, t + i);
				partial_results[t].value = add_mod(partial_results[t].value, mul_mod(partial_results[t + i].value, word_pow_mod(neighbor.b - b, mod), mod), mod);
			}
		}
	};
	// s ������
	for (std::size_t i = 1; i < T; ++i) {
		threads[i - 1] = std::thread(thread_lambda, i);
	}
	thread_lambda(0);

	for (auto& i : threads) {
		i.join();
	}
	return partial_results[0].value;
}