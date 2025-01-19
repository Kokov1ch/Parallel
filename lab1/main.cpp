#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include <fstream>
const double N = 100'000'000;

double f(double x)
{
    return x * x + 3;
}

double integrate(double a, double b)
{
    double sum = 0;
    double dx = (b - a) / N;

    for (int i = 0; i < N; i++)
    {
        sum += f(a + i * dx);
    }

    return dx * sum;
}

double integrate_omp(double a, double b)
{
    double sum = 0;
    double dx = (b - a) / N;

#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        double local_sum = 0;

        for (size_t i = t; i < N; i += T)
        {
            local_sum += f(a + i * dx);
        }

#pragma omp critical
        {
            sum += local_sum;
        }
    }

    return dx * sum;
}

int main()
{
    std::ofstream output("../output.csv");
    if (!output.is_open())
    {
        std::cout << "Error. Could not open file!\n";
        return -1;
    }

    const size_t thread_count = std::thread::hardware_concurrency();
    const size_t trial_count = 20;
    const double a = 0;
    const double b = 1;
    const double sToMs = 1'000;

    size_t times[thread_count + 1];
    double values[thread_count + 1];

    double t1 = 0, t2 = 0;
    double total_time = 0;
    double result = 0;

    for(size_t trial = 0; trial < trial_count; trial++){
        double t1 = omp_get_wtime();
        result = integrate(a, b);
        double t2 = omp_get_wtime();
        total_time += t2 - t1;
    }

    times[0] = sToMs * total_time / trial_count;
    values[0] = result;

    for (std::size_t i = 1; i <= thread_count; i++)
    {
        total_time = 0;
        for(size_t trial = 0; trial < trial_count; trial++){
            omp_set_num_threads(i);
            t1 = omp_get_wtime();
            result = integrate_omp(a, b);
            t2 = omp_get_wtime();
            total_time += t2 - t1;
        }

        times[i] = sToMs * total_time / trial_count;
        values[i] = result;
    }

    std::cout << "T\t| Duration\t| Value\n";
    output << "T,Duration\n";

    for(size_t i = 0; i <= thread_count; i++){
        std::cout << i << "\t| " << times[i] << "\t\t| " << values[i] << "\n";
        output << i << "," << times[i] << "\n";
    }

    output.close();
    return 0;
}