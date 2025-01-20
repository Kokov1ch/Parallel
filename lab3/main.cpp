#include <assert.h>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>

using namespace std;

void mulMatrix(double* A, size_t cA, size_t rA,
               const double* B, size_t cB, size_t rB,
               const double* C, size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < cA; i++)
    {
        for (size_t j = 0; j < rA; j++)
        {
            A[i * rA + j] = 0;
            for (size_t k = 0; k < cB; k++)
            {
                A[i * rA + j] += B[k * rB + j] * C[i * rC + k];
            }
        }
    }
}

void mulMatrix256(double* A, const double* B, const double* C,
                   size_t cA, size_t rA,
                   size_t cB, size_t rB,
                   size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    const size_t values_per_operation = 4;

    for (size_t i = 0; i < rB / values_per_operation; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                __m256d bCol = _mm256_loadu_pd(B + rB * k + i * values_per_operation);
                __m256d broadcasted = _mm256_set1_pd(C[j * rC + k]);
                sum = _mm256_fmadd_pd(bCol, broadcasted, sum);
            }

            _mm256_storeu_pd(A + j * rA + i * values_per_operation, sum);
        }
    }
}

//pair<vector<double>, vector<double>> getPermutationMatrix(size_t n)
//{
//    vector<size_t> permut(n);
//
//    for (size_t i = 0; i < n; i++)
//    {
//        permut[i] = (n - (1 + 10)) % n;
//    }
//
//    vector<double> vf(n * n), vi(n * n);
//
//    for (size_t c = 0; c < n; c++)
//    {
//        for (size_t r = 0; r < n; r++)
//        {
//            vf[c * n + r] = vi[r * n + c] = 1;
//        }
//    }
//
//    return pair{ vf, vi };
//}

vector<double> getPermutationMatrix(size_t n)
{
    vector<double> matrix(n * n);
    vector<size_t> permut(n);

    for (size_t i = 0; i < n; i++)
    {
        permut[i] = i;
        swap(permut[i], permut[rand() % (i + 1)]);
    }

    for (size_t c = 0; c < n; c++)
    {
        matrix[c * n + permut[c]] = 1;
    }

    return matrix;
}

vector<double> getIdentityMatrix(size_t n)
{
    vector<double> matrix(n * n);

    for (size_t c = 0; c < n; c++)
    {
        matrix[c * n + c] = 1;
    }

    return matrix;
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    auto show_matrix = [](const double* A, size_t colsc, size_t rowsc)
    {
        return;

        for (size_t r = 0; r < rowsc; ++r)
        {
            cout << "[" << A[r + 0 * rowsc];
            for (size_t c = 1; c < colsc; ++c)
            {
                cout << ", " << A[r + c * rowsc];
            }
            cout << "]\n";
        }
        cout << "\n";
    };

    const size_t matrix_size = 64 * (1 << 4);

    vector<double> A(matrix_size * matrix_size),
    //B(matrix_size * matrix_size, 0),
    //C(matrix_size * matrix_size, 0),
    D(matrix_size * matrix_size);

    auto identity_matrix = getIdentityMatrix(matrix_size);
    auto permutation_matrix = getPermutationMatrix(matrix_size);

    auto B = identity_matrix;
    auto C = permutation_matrix;

    //Perform naive multiplication.
    auto t1 = chrono::steady_clock::now();
    mulMatrix(A.data(), matrix_size, matrix_size,
              B.data(), matrix_size, matrix_size,
              C.data(), matrix_size, matrix_size);
    auto t2 = chrono::steady_clock::now();
    //show_matrix(A.data(), matrix_size, matrix_size);
    cout << "Scalar: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " ms\n";

    //Perform vectorized multiplication.
    t1 = chrono::steady_clock::now();
    mulMatrix256(D.data(), B.data(), C.data(), matrix_size, matrix_size, matrix_size, matrix_size, matrix_size,
                  matrix_size);
    t2 = chrono::steady_clock::now();
    //show_matrix(D.data(), matrix_size, matrix_size);
    cout << "Vector: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " ms\n";

    if (!std::memcmp(static_cast<void*>(A.data()),
                     static_cast<void*>(D.data()),
                     matrix_size * matrix_size * sizeof(double)))
    {
        cout << "A = D\n";
    }

    show_matrix(A.data(), matrix_size, matrix_size);
    show_matrix(D.data(), matrix_size, matrix_size);

    return 0;
}