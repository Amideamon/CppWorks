#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include <cmath>
#include "vectors_and_matrices/array_types.hpp"

#include <omp.h>

using namespace std;

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

void random_bounds(matrix<double> u, ptrdiff_t ncenters, double ampl, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_charge(-ampl, ampl);
    std::uniform_real_distribution<double> dist_coord(-1, 1);
    
    ptrdiff_t generated=0, nx = u.nrows(), ny = u.ncols();
    double hx = 1.0 / (nx-1), hy = 1.0 / (ny -1);
    while (generated < ncenters)
    {
        double x = dist_coord(rng), y = dist_coord(rng);
        if ((abs(x) > 0.5) && (abs(y) > 0.5))
        {
            double charge = dist_charge(rng);
            for (ptrdiff_t i=0; i < nx; i++)
            {
                u(i, 0) = 0;
                u(i, ny-1) = 0;
            }
            for (ptrdiff_t j=1; j < ny-1; j++)
            {
                u(0, j) = 0;
                u(nx-1, j) = 0;
            }
            for (ptrdiff_t i=0; i < nx; i++)
            {
                double r = hypot(x - (-0.5 + i * hx), y + 0.5);
                u(i, 0) += charge / r;
                r = hypot(x - (-0.5 + i * hx), y - 0.5);
                u(i, ny-1) += charge / r;
            }
            for (ptrdiff_t j=1; j < ny-1; j++)
            {
                double r = hypot(x + 0.5, y - (-0.5 + j * hy));
                u(0, j) += charge / r;
                r = hypot(x - 0.5, y - (-0.5 + j * hy));
                u(nx-1, j) += charge / r;
            }
            generated += 1;
        }
    }
}

int test_laplace(matrix<double> u, double atol)
{
    ptrdiff_t nx = u.nrows(), ny = u.ncols();
    for (ptrdiff_t i = 1; i < nx-1; i++)
    {
        for (ptrdiff_t j = 1; j < ny-1; j++)
        {
            if (abs(u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1) - 4 * u(i, j)) > atol)
            {
                return 0;
            }
        }
    }
    return 1;
}


matrix<double> laplace2d(matrix<double> u, double hx, double hy)
{
    ptrdiff_t nx = u.nrows(), ny = u.ncols();
    matrix<double> u_new(nx, ny);
    u_new = u;
    ptrdiff_t j, i, count_iterations = 0;
    while(test_laplace(u, 1e-6) == 0) 
    {   
        count_iterations += 1;
        #pragma omp parallel
        {
            // count_iterations += 1;
            #pragma omp for private(j) nowait
            for (i = 1; i < nx-1; i++) {
                for (j = 1; j < ny-1; j++) {
                    u_new(i, j) = 0.25 * (u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1));
                }
            }
            u = u_new;    
        }
    }
    cout << "Iterations:" << count_iterations << endl;
    return u;
}


int main(int argc, char* argv[])
{
    ptrdiff_t n;

    std::cin >> n;
    matrix<double> u(n, n);

    random_bounds(u, 100, 5.0, 9876);

    // for (int i=0; i<n; i++){
    //     for (int j=0; j<n; j++) {
    //         cout << u(i,j) << "  ";
    //     }
    //     cout << endl;
    // }

    double t0 = omp_get_wtime();

    u = laplace2d(u, 1.0 / (n-1), 1.0 / (n-1));

    double t1 = omp_get_wtime();

    // cout << endl;
    // for (int i=0; i<n; i++){
    //     for (int j=0; j<n; j++) {
    //         cout << u(i,j) << "  ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "Laplace equation satisfied: " << test_laplace(u, 1e-6)
              << std::endl;
    return 0;
}
