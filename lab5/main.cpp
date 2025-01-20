#include <complex>
#include <bit>
#include <bitset>
#include <iostream>
#include <vector>
#include <thread>
#include <barrier>
#include <fstream>

const size_t experiments = 5;
const std::size_t N = 1 << 20;

void bitShuffle(const std::complex<double>* in, std::complex<double>* out, std::size_t n){
    std::size_t length = sizeof(n) * 8 - std::countl_zero(n) - 1;
    for(std::size_t i = 0; i < n; i++){
        std::size_t index = i;
        std::size_t newIndex = 0;

        for(int j = 0; j < length; j++){
            newIndex <<= 1;
            newIndex += (index & 1);
            index >>= 1;
        }

        out[newIndex] = in[i];
    }
}

void printFft(std::complex<double>* out, size_t N){
    std::cout << "Out:" << std::endl;
    for(int i = 0; i < N; i++){
        std::cout << round(100 * abs(out[i])) / 100 << " ";
    }
    std::cout << std::endl;
}

void printIfft(std::complex<double>* out, size_t N)
{
    std::cout << "IFFT:" << std::endl;
    for(int i = 0; i < N; i++){
        std::cout << std::fixed << out[i] / static_cast<std::complex<double>>(N) << " ";
    }
    std::cout << std::endl;
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
        auto l = out[i];
        auto r = out[i + n / 2];
        out[i] = l + w * r;
        out[i + n / 2] = l - w * r;
    }
}

void parallelFft(const std::complex<double>* in, std::complex<double>* out, std::size_t N, std::size_t T){
    std::vector<std::thread> threads(T - 1);

    std::barrier<> bar(T);

    auto process = [&in, &out, N, T, &bar](unsigned threadNumber) {
        for(size_t i = threadNumber; i < N; i += T){
            out[i] = in[i];
        }

        for(size_t n = 2; n <= N; n += n){
            bar.arrive_and_wait();
            for (size_t start = threadNumber * n; start + n <= N; start += T * n) {
                for (std::size_t i = 0; i < n / 2; i++) {
                    auto w = std::polar(1.0, -2.0 * i * std::numbers::pi_v<double> / n);
                    auto l = out[start + i];
                    auto r = out[start + i + n / 2];
                    out[start + i] = l + w * r;
                    out[start + i + n / 2] = l - w * r;
                }
            }
        }
    };

    for (std::size_t i = 1; i < T; ++i) {
        threads[i - 1] = std::thread(process, i);
    }
    process(0);

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

void outFft(std::complex<double>* out){
    std::cout << "Out:" << std::endl;
    for(int i = 0; i < N; i++){
        std::cout << round(100 * abs(out[i])) / 100 << " ";
    }
    std::cout << std::endl;
}

int main()
{
    std::ofstream output("../output.csv");
    if (!output.is_open())
    {
        std::cout << "Error. Could not open file!\n";
        return -1;
    }

    std::vector<std::complex<double>> in(N), shuffled_in(N);
    std::vector<std::complex<double>> out(N), shuffled_out(N), iout(N);

    for(int i = 0; i < N; i++){
        in[i] = i;
    }

    bitShuffle(in.data(), shuffled_in.data(), N);
    size_t threadCount = std::thread::hardware_concurrency();

    size_t result[threadCount + 1];

    size_t durationRecursive = 0;
    for(int i = 0; i < experiments; i++){
        auto tm0 = std::chrono::steady_clock::now();
        fft(shuffled_in.data(), out.data(), N);
        auto time = duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm0);
        durationRecursive += time.count();
    }
    result[0] = durationRecursive / experiments;

    for(size_t i = 1; i <= threadCount; i++){
        size_t durationParallel = 0;
        for(int j = 0; j < experiments; j++){
            auto tm0 = std::chrono::steady_clock::now();
            parallelFft(shuffled_in.data(), out.data(), N, i);
            auto time = duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm0);
            durationParallel += time.count();
        }

        result[i] = durationParallel / experiments;
    }



    std::cout << "threads\t| duration\t| acceleration\n";
    output << "threads,duration\n";
    for(size_t i = 0; i <= threadCount; i++){
        std::cout << i << "\t| " << result[i] << "\t| " << std::fixed << result[0] / (double)result[i] << std::endl;
        output << i << "," << result[i] << "\n";
    }

    parallelFft(shuffled_in.data(), out.data(), N, 1);


    bitShuffle(out.data(), shuffled_out.data(), N);
    ifft(shuffled_out.data(), iout.data(), N);


//    printFft(out.data(), N);
//    printIfft(iout.data(), N);

    return 0;
}

