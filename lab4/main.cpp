#include "vector_mod.h"
#include "test.h"
#include "performance.h"
#include <iostream>
#include <iomanip>
#include "num_threads.h"
#include <fstream>

int main(int argc, char** argv)
{
    std::ofstream output("../output.csv");
    if (!output.is_open())
    {
        std::cout << "Couldn't open file!\n";
        return -1;
    }

    std::cout << "==Correctness tests. ";
    //set_num_threads(3);
    for (std::size_t iTest = 1; iTest < test_data_count; ++iTest)
    {
        if (test_data[iTest].result != vector_mod(test_data[iTest].dividend, test_data[iTest].dividend_size, test_data[iTest].divisor))
        {
            std::cout << "FAILURE==\n";
            return -1;
        }
    }
    std::cout << "ok.==\n";

    std::cout << "==Performance tests. ";
    auto measurements = run_experiments();
    std::cout << "Done==\n";
    std::cout << std::setfill(' ') << std::setw(2) << "threads:" << " |" << std::setw(3 + 2 * sizeof(IntegerWord)) << "value:" << " | " <<
              std::setw(14) << "duration, ms:" << " | acceleration:\n";
    output << "threads,duration\n";
    for (std::size_t T = 1; T <= measurements.size(); ++T)
    {
        output << T << "," << measurements[T - 1].time.count() << "\n";
        std::cout << std::setw(2) << T << " | 0x" << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0') << std::hex << measurements[T - 1].result;
        std::cout << " | " << std::setfill(' ') << std::setw(14) << std::dec << measurements[T - 1].time.count();
        std::cout << " | " << (static_cast<double>(measurements[0].time.count()) / measurements[T - 1].time.count()) << "\n";
    }

    return 0;
}