#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include <fstream>
#include "vector"

const double N = 100'000'000;
const size_t experiments = 10;


double f(double x)
{
    return x * x -1;
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

double integrateParallel(double a, double b)
{
    double sum = 0;
    double dx = (b - a) / N;

#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        double threadSum = 0;

        for (size_t i = t; i < N; i += T)
        {
            threadSum += f(a + i * dx);
        }

#pragma omp critical
        {
            sum += threadSum;
        }
    }

    return dx * sum;
}

int main()
{
    std::ofstream output("../output.csv");

    if (!output.is_open())
    {
        std::cout << "Couldn't open file!\n";
        return -1;
    }

    const size_t threadCount = std::thread::hardware_concurrency();

    const double a = 0;
    const double b = 1;


    std::vector<size_t> times(threadCount + 1);
    std::vector<double> values(threadCount + 1);

    double t1 = 0, t2 = 0;
    double totalTime = 0;

    double result = 0;

    for(size_t i = 0; i < experiments; ++i){
        double t1 = omp_get_wtime();
        result = integrate(a, b);
        double t2 = omp_get_wtime();
        totalTime += t2 - t1;
    }

    times[0] = 1000 * totalTime / experiments;
    values[0] = result;

    for (std::size_t i = 1; i <= threadCount; i++)
    {
        totalTime = 0;
        for(size_t trial = 0; trial < experiments; trial++){
            omp_set_num_threads(i);
            t1 = omp_get_wtime();
            result = integrateParallel(a, b);
            t2 = omp_get_wtime();
            totalTime += t2 - t1;
        }

        times[i] = 1000 * totalTime / experiments;
        values[i] = result;
    }

    std::cout << "thread\t duration\t value\n";
    output << "thread, duration\n";

    for(size_t i = 0; i <= threadCount; i++){
        std::cout << i << "\t " << times[i] << "\t\t " << values[i] << "\n";
        output << i << "," << times[i] << "\n";
    }

    output.close();
    return 0;
}