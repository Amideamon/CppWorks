#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstddef>
#include <random>

#include <omp.h>

#include "vectors_and_matrices/array_types.hpp"

using std::cin;
using std::cout;
using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

const double PI = 3.141592653589793;

double maxwell_average(ptrdiff_t niter, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist(0.0, 1.0);
    double ave_norm = 0.0;
    #pragma omp parallel
        {
            #pragma omp for private(rng) reduction(+: ave_norm) 
            for (ptrdiff_t i = 0; i < niter; ++i)
            {
                ave_norm += std::sqrt(pow(dist(rng), 2) + pow(dist(rng), 2) + pow(dist(rng), 2));
            }
        }
    ave_norm /= niter;
    return ave_norm;
}

int main(int argc, char** argv)
{
    ptrdiff_t niter;

    cin >> niter;

    double t1 = omp_get_wtime();

    double gaussian3d_abs_average = maxwell_average(niter, 1234);

    double t2 = omp_get_wtime();
    
    cout << "Computed average: " << std::setprecision(16) << gaussian3d_abs_average << std::endl;
    cout << "Exact average: " << std::setprecision(16) << sqrt(8 / PI) << std::endl;
    cout << "Time: " << t2 - t1 << std::endl;

    return 0;
}