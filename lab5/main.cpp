#include <complex>
#include <bit>
#include <bitset>
#include <iostream>
#include <vector>
#include <thread>
#include <barrier>
#include <fstream>

void bit_shuffle(const std::complex<double>* in, std::complex<double>* out, std::size_t n){
    std::size_t index_len = sizeof(n) * 8 - std::countl_zero(n) - 1;
    for(std::size_t i = 0; i < n; i++){
        std::size_t index = i;
        std::size_t newIndex = 0;

        for(int j = 0; j < index_len; j++){
            newIndex <<= 1;
            newIndex += (index & 1);
            index >>= 1;
        }

        out[newIndex] = in[i];
    }
}

void fft(const std::complex<double>* in, std::complex<double>* out, std::size_t n){
    if(n == 1) {
        out[0] = in[0];
        return;
    }

    fft(in, out, n / 2);
    fft(in + n / 2, out + n / 2, n / 2);
    for(std::size_t i = 0; i < n / 2; i++){
        auto w = std::polar(1.0, -2.0 * i * std::numbers::pi_v<double> / n);
        auto r1 = out[i];
        auto r2 = out[i + n / 2];
        out[i] = r1 + w * r2;
        out[i + n / 2] = r1 - w * r2;
    }
}

void parallel_fft(const std::complex<double>* in, std::complex<double>* out, std::size_t N, std::size_t T){
    std::vector<std::thread> threads(T - 1);

    std::barrier<> bar(T);
    auto thread_lambda = [&in, &out, N, T, &bar](unsigned threadNumber) {
        for(size_t i = threadNumber; i < N; i += T){
            out[i] = in[i];
        }

        for(size_t n = 2; n <= N; n += n){
            bar.arrive_and_wait();
            for (size_t start = threadNumber * n; start + n <= N; start += T * n) {
                for (std::size_t i = 0; i < n / 2; i++) {
                    auto w = std::polar(1.0, -2.0 * i * std::numbers::pi_v<double> / n);
                    auto r1 = out[start + i];
                    auto r2 = out[start + i + n / 2];
                    out[start + i] = r1 + w * r2;
                    out[start + i + n / 2] = r1 - w * r2;
                }
            }
        }
    };

    for (std::size_t i = 1; i < T; ++i) {
        threads[i - 1] = std::thread(thread_lambda, i);
    }
    thread_lambda(0);

    for (auto& i : threads) {
        i.join();
    }
}

void ifft(const std::complex<double>* in, std::complex<double>* out, std::size_t n){
    if(n == 1) {
        out[0] = in[0];
        return;
    }

    ifft(in, out, n / 2);
    ifft(in + n / 2, out + n / 2, n / 2);
    for(std::size_t i = 0; i < n / 2; i++){
        auto w = std::polar(1.0, 2.0 * i * std::numbers::pi_v<double> / n);
        auto r1 = out[i];
        auto r2 = out[i + n / 2];
        out[i] = r1 + w * r2;
        out[i + n / 2] = r1 - w * r2;
    }
}

int main()
{
    std::ofstream output("../output.csv");
    if (!output.is_open())
    {
        std::cout << "Error. Could not open file!\n";
        return -1;
    }

    const std::size_t n = 1 << 20;
    std::vector<std::complex<double>> in(n), shuffled_in(n);
    std::vector<std::complex<double>> out(n), shuffled_out(n), iout(n);

//    for(int i = 0; i < n / 2; i++){
//        in[i] = i;
//        in[n - 1 - i] = i;
//    }

    for(int i = 0; i < n; i++){
        in[i] = i;
    }

//    for(int i = 0; i < n; i++){
//        in[i] = i + 1;
//    }

    bit_shuffle(in.data(), shuffled_in.data(), n);
    size_t trials = 5;
    size_t thread_count = std::thread::hardware_concurrency();

    size_t result[thread_count + 1];

    size_t recursive_time = 0;
    for(int i = 0; i < trials; i++){
        auto tm0 = std::chrono::steady_clock::now();
        fft(shuffled_in.data(), out.data(), n);
        auto time = duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm0);
        recursive_time += time.count();

        // std::cout << "recursive [" << i + 1 << "]: " << time.count() << " ms." << std::endl;
    }
    result[0] = recursive_time / trials;

    for(size_t i = 1; i <= thread_count; i++){
        size_t parallel_time = 0;
        for(int j = 0; j < trials; j++){
            auto tm0 = std::chrono::steady_clock::now();
            parallel_fft(shuffled_in.data(), out.data(), n, i);
            auto time = duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm0);
            parallel_time += time.count();
        }

        result[i] = parallel_time / trials;
    }

    std::cout << "T\t| Duration\t| Acceleration\n";
    output << "T,Duration\n";
    for(size_t i = 0; i <= thread_count; i++){
        std::cout << i << "\t| " << result[i] << "\t| " << std::fixed << result[0] / (double)result[i] << std::endl;
        output << i << "," << result[i] << "\n";
    }

    // parallel_fft(shuffled_in.data(), out.data(), n, 1);

    // std::cout << "AVG parallel: " << sumPar / trials << " ms." << std::endl;
    // std::cout << "AVG recursive: " << sumRec / trials << " ms." << std::endl;

//    bit_shuffle(out.data(), shuffled_out.data(), n);
//    ifft(shuffled_out.data(), iout.data(), n);

//    std::cout << "Out:" << std::endl;
//    for(int i = 0; i < n; i++){
//        std::cout << round(100 * abs(out[i])) / 100 << " ";
//    }
//    std::cout << std::endl;

//    std::cout << "IFFT:" << std::endl;
//    for(int i = 0; i < n; i++){
//        std::cout << std::fixed << iout[i] / static_cast<std::complex<double>>(n) << " ";
//    }
//    std::cout << std::endl;

    return 0;
}