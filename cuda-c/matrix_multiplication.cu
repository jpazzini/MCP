#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

void matrix_multiplication(const int* M, const int* N, int* P, const int width){
    // Naive square matrix multiplication algorithm for CPU
    //
    // P = M*N
    // p_i,j = Sum_k m_ik * n_kj

    for (int i = 0; i < width; ++i){
        for (int j = 0; j < width; ++j){
            int p = 0;
            for (int k = 0; k < width; ++k){
               int m = M[i*width + k];
               int n = N[k*width + j];
               p += m*n;
            }
            P[i*width + j] = p;
        }
    }
}

int random_number(){
    return (std::rand()%100);
}

void print_matrix(const int * M, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            printf("%d",M[i*n+j]);
            if (j==n) continue;
            printf("\t");
        }
        printf("\n");
    }
}

int main(int argc, char **argv){

    srand(time(NULL));

    const int n = 1024;                         // Width of the matrices
    
    std::vector<int> A(n*n), B(n*n), C(n*n);    // Local variables, hosted in memory
    std::generate(A.begin(), A.end(), random_number);
    std::generate(B.begin(), B.end(), random_number);
    
    printf("Matrix A\n");
    //print_matrix(A.data(), n);

    printf("Matrix B\n");
    //print_matrix(B.data(), n);

    matrix_multiplication(A.data(), B.data(), C.data(), n);

    printf("Matrix C\n");
    //print_matrix(C.data(), n);

    return 0;

}
