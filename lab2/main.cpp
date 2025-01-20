#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <fstream>

const int cols = 1 << 15;
const int rows = 1 << 15;
const size_t batch = 4;

void addMatrix(double* A, const double* B, const double* C, size_t colsc, size_t rowsc)
{
    for (size_t i = 0; i < colsc * rowsc; i++)
    {
        A[i] = B[i] + C[i];
    }
}

void addMatrix256(double* A, const double* B, const double* C, size_t colsc, size_t rowsc)
{
    for (size_t i = 0; i < rowsc * colsc / batch; i++)
    {
        __m256d b = _mm256_loadu_pd(&(B[i * batch]));
        __m256d c = _mm256_loadu_pd(&(C[i * batch]));
        __m256d a = _mm256_add_pd(b, c);

        _mm256_storeu_pd(&(A[i * batch]), a);
    }
}

std::vector<double> B(cols* rows, 1), C(cols* rows, -2), A(cols* rows);

int main(int argc, char** argv)
{
    std::ofstream output("../output.csv");

    if (!output.is_open())
    {
        std::cout << "Couldn't open file!\n";
        return -1;
    }
    auto show_matrix = [](const double* A, size_t colsc, size_t rowsc)
    {
        for (size_t r = 0; r < rowsc; ++r)
        {
            std::cout << "[" << A[r + 0 * rowsc];
            for (size_t c = 1; c < colsc; ++c)
            {
                std::cout << ", " << A[r + c * rowsc];
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    };

    //    show_matrix(B.data(), cols, rows);
    //    show_matrix(C.data(), cols, rows);

    auto t1 = std::chrono::steady_clock::now();
    addMatrix(A.data(), B.data(), C.data(), cols, rows);
    auto t2 = std::chrono::steady_clock::now();
    //    show_matrix(A.data(), cols, rows);
    using namespace std::chrono;
    output << "scalar, vector\n";
    std::cout << "Scalar: " << duration_cast<milliseconds>(t2 - t1).count() << " ms.\n";
    output << duration_cast<milliseconds>(t2 - t1).count() << ", ";

    std::fill_n(A.data(), rows * cols, 0);
    std::fill_n(B.data(), rows * cols, -2);
    std::fill_n(C.data(), rows * cols, 1);

    t1 = std::chrono::steady_clock::now();
    addMatrix256(A.data(), B.data(), C.data(), cols, rows);
    t2 = std::chrono::steady_clock::now();
    //    show_matrix(A.data(), cols, rows);
    std::cout << "Vector: " << duration_cast<milliseconds>(t2 - t1).count() << " ms.\n";
    output << duration_cast<milliseconds>(t2 - t1).count();
    output.close();

    return 0;
}