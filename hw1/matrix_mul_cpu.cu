//
// Created by james on 25-8-21.
//
#include <stdio.h>
#include <time.h>

const int DSIZE = 8192;  // Matrix size
const float A_val = 3.0f;
const float B_val = 2.0f;

int main() {
    // Host arrays for matrices
    float *h_A, *h_B, *h_C;

    // Timing variables
    clock_t t0, t1, t2;
    double t1sum = 0.0;
    double t2sum = 0.0;

    // Start timing
    t0 = clock();

    // Allocate memory for matrices
    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];

    // Initialize matrices A and B, and matrix C to 0
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Perform matrix multiplication on the CPU
    t2 = clock();
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            float temp = 0;
            for (int k = 0; k < DSIZE; k++) {
                temp += h_A[i * DSIZE + k] * h_B[k * DSIZE + j];
            }
            h_C[i * DSIZE + j] = temp;
        }
    }
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Done. Compute took %f seconds\n", t2sum);

    // Verify results
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        if (h_C[i] != A_val * B_val * DSIZE) {
            printf("Mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val * B_val * DSIZE);
            return -1;
        }
    }
    printf("Success!\n");

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
