#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <random>
// extern "C"
// {
//     #include <cblas.h>
// }
#include <mkl.h>

using namespace std;

int experiments = 1;
double atol_m = 1e-6;

void dgemm(int N, double *A, double *B, double *C)
{
    for (int i=0; i < N*N; i++) {
        for (int j=0; j < N; j++) {
            C[i] += A[i % N + j * N] * B[(i / N) * N + j];
        }
    }
}

void fill_random(double *X, int N, double xmin, double xmax, int seed)
{
    mt19937_64 rng(seed);
    uniform_real_distribution<double> dist(xmin, xmax);
    for (int i = 0; i < N*N; i++) {
        X[i] = dist(rng);
    }
}

bool compare_matrix(double *A, double *B, int N, double atol_m)
{
    for (int i = 0; i < N*N; i++) {
        if (abs(A[i] - B[i]) > atol_m) return false;
    }
    return true;
}

int main(int argc, char **argv) {

    int N;
    cin >> N;
    cout << "N = " << N << "\n";

    double *A = new double[N*N];
    double *B = new double[N*N];
    double *C = new double[N*N];
    double *C_parallel = new double[N*N];

    double t_sum = 0.;
    double t_sum_par = 0.;

    for (int j=0; j<experiments; j++)
    {
        fill_random(A, N, -10., 10., 1234);
        fill_random(B, N, -10., 10., 9876); 
        for (int i=0; i < N*N; i++) {
            C[i] = 0.0;
            C_parallel[i] = 0.0;
        }

        dgemm(N, A, B, C);

        double t1 = omp_get_wtime();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0.0, C_parallel, N);
        double t2 = omp_get_wtime();

        t_sum_par += (t2-t1);    

        if (!compare_matrix(C, C_parallel, N, atol_m)) {
            cout << "Parallel function is wrong." << "\n";
        }
    }


    // cout << "Dgemm time: " << t_sum / experiments << "\n";

    cout << "Dgemm time: " << t_sum_par / experiments << "\n";

    delete [] A;
    delete [] B;
    delete [] C;
    delete [] C_parallel;

}