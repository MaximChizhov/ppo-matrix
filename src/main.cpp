#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <immintrin.h>
#include <omp.h>

#define MKL
#ifdef MKL
#include "mkl.h"
#endif

using namespace std;

void generation(double* mat, size_t size)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
    for (size_t i = 0; i < size * size; i++)
        mat[i] = uniform_distance(gen);
}

void matrix_mult(double* a, double* b, double* res, size_t size)
{
    memset(res, 0, size * size * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int k = 0; k < size; k++)
        {
            __m256d a_ik = _mm256_set1_pd(a[i * size + k]); // Loading a[i][k] into a vector
            for (int j = 0; j < size; j += 4) // Processing 4 items at a time
            {
                __m256d b_vec = _mm256_loadu_pd(&b[k * size + j]); // Loading 4 items from b
                __m256d res_vec = _mm256_loadu_pd(&res[i * size + j]); // Loading 4 items from res
                res_vec = _mm256_add_pd(res_vec, _mm256_mul_pd(a_ik, b_vec)); // Multiply and add
                _mm256_storeu_pd(&res[i * size + j], res_vec); // Saving the result
            }
        }
    }
}

int main()
{
    double* mat, * mat_mkl, * a, * b, * a_mkl, * b_mkl;
    size_t size = 1000;
    chrono::time_point<chrono::system_clock> start, end;

    mat = new double[size * size];
    a = new double[size * size];
    b = new double[size * size];
    generation(a, size);
    generation(b, size);
    memset(mat, 0, size * size * sizeof(double));

#ifdef MKL     
    mat_mkl = new double[size * size];
    a_mkl = new double[size * size];
    b_mkl = new double[size * size];
    memcpy(a_mkl, a, sizeof(double) * size * size);
    memcpy(b_mkl, b, sizeof(double) * size * size);
    memset(mat_mkl, 0, size * size * sizeof(double));
#endif

    start = chrono::system_clock::now();
    matrix_mult(a, b, mat, size);
    end = chrono::system_clock::now();


    int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
        (end - start).count();
    cout << "Total time: " << elapsed_seconds / 1000.0 << " sec" << endl;

#ifdef MKL 
    start = chrono::system_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
    end = chrono::system_clock::now();

    elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
        (end - start).count();
    cout << "Total time mkl: " << elapsed_seconds / 1000.0 << " sec" << endl;

    int flag = 0;
    for (unsigned int i = 0; i < size * size; i++)
        if (abs(mat[i] - mat_mkl[i]) > size * 1e-14) {
            flag = 1;
        }
    if (flag)
        cout << "fail" << endl;
    else
        cout << "correct" << endl;

    delete (a_mkl);
    delete (b_mkl);
    delete (mat_mkl);
#endif

    delete (a);
    delete (b);
    delete (mat);

    //system("pause");

    return 0;
}
