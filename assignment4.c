#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define NUM_TESTS     100     // Test interations
#define MIN_SIZE      0       // Minimum test size (bytes)
#define MAX_SIZE      (1<<20) // Maximum test size (1MB)
#define SIZE_STEP     1024    // Step size
#define WARMUP_ROUNDS 5       // Warm up round

/* Structure for Linear Regression */
struct regression {
    double sum_x;
    double sum_y;
    double sum_xy;
    double sum_x2;
    int count;
};

void init_regression(struct regression *r) {
    r->sum_x = r->sum_y = r->sum_xy = r->sum_x2 = 0.0;
    r->count = 0;
}

void update_regression(struct regression *r, double x, double y) {
    r->sum_x += x;
    r->sum_y += y;
    r->sum_xy += x * y;
    r->sum_x2 += x * x;
    r->count++;
}

void compute_regression(struct regression *r, double *lambda, double *beta) {
    double denominator = r->sum_x2 * r->count - r->sum_x * r->sum_x;
    if (denominator == 0) {
        *lambda = *beta = -1;
        return;
    }
    
    double b = (r->count * r->sum_xy - r->sum_x * r->sum_y) / denominator;
    double a = (r->sum_y - b * r->sum_x) / r->count;
    
    *lambda = a;        // Latency (s)
    *beta = 1.0 / b;    // Bandwidth (bytes/s)
}

int main(int argc, char **argv) {
    int rank, size;
    char *buffer = NULL;
    double start_time, end_time, avg_time;
    struct regression reg;
    MPI_Status status;
    int msg_size, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2 Progress
    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "Err: MUST 2 PROGRESS\n");
        MPI_Finalize();
        return 1;
    }

    init_regression(&reg);

    // Main iteration (only run in process 0)
    if (rank == 0) {
        printf("%10s %12s\n", "SIZE(B)", "AVERAGE TIME(s)");
        
        for (msg_size = MIN_SIZE; msg_size <= MAX_SIZE; msg_size += SIZE_STEP) {
            // 分配对齐内存
            if (msg_size > 0) {
                if (posix_memalign((void **)&buffer, 64, msg_size) != 0) {
                    fprintf(stderr, "内存分配失败: %d bytes\n", msg_size);
                    continue;
                }
            }

            // Warm up
            for (i = 0; i < WARMUP_ROUNDS; i++) {
                if (msg_size > 0) {
                    MPI_Send(buffer, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(buffer, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
                } else {
                    MPI_Send(NULL, 0, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(NULL, 0, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
                }
            }

            // Test Start Here
            avg_time = 0.0;
            for (i = 0; i < NUM_TESTS; i++) {
                start_time = MPI_Wtime();
                
                if (msg_size > 0) {
                    MPI_Send(buffer, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(buffer, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
                } else {
                    MPI_Send(NULL, 0, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(NULL, 0, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
                }
                
                end_time = MPI_Wtime();
                avg_time += (end_time - start_time) / 2;
            }
            
            if (NUM_TESTS > 0)
                avg_time /= NUM_TESTS;

            // obtain result
            if (msg_size > 0) {
                printf("%10d %12.3e\n", msg_size, avg_time);
                update_regression(&reg, msg_size, avg_time);
            }

            if (buffer) {
                free(buffer);
                buffer = NULL;
            }
        }

        // Print Result Here
        double lambda, beta;
        compute_regression(&reg, &lambda, &beta);
        
        printf("\nTest Result:\n");
        printf("  Latency(λ) = %.3f μs\n", lambda * 1e6);
        printf("  Bandwidth(β) = %.3f GB/s\n", beta / 1e9);
    } 
    // Process 1: 响应测试
    else {
        for (msg_size = MIN_SIZE; msg_size <= MAX_SIZE; msg_size += SIZE_STEP) {
            if (msg_size > 0) {
                if (posix_memalign((void **)&buffer, 64, msg_size) != 0) {
                    fprintf(stderr, "内存分配失败: %d bytes\n", msg_size);
                    continue;
                }
            }

            // warm up
            for (i = 0; i < WARMUP_ROUNDS; i++) {
                if (msg_size > 0) {
                    MPI_Recv(buffer, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
                    MPI_Send(buffer, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                } else {
                    MPI_Recv(NULL, 0, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
                    MPI_Send(NULL, 0, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                }
            }

            // main test
            for (i = 0; i < NUM_TESTS; i++) {
                if (msg_size > 0) {
                    MPI_Recv(buffer, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
                    MPI_Send(buffer, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                } else {
                    MPI_Recv(NULL, 0, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
                    MPI_Send(NULL, 0, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                }
            }

            if (buffer) {
                free(buffer);
                buffer = NULL;
            }
        }
    }

    MPI_Finalize();
    return 0;
}