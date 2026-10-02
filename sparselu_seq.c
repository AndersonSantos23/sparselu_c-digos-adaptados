#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define BSIZE 64

typedef struct _block_t {
    double *restrict val;
} block_t;

int NB = 10;
int M = 10;

// Kernels Computacionais Sequenciais
void cpu_lu0(block_t *diag) {
    int i, j, k;
    for (k = 0; k < BSIZE; k++) {
        for (i = k + 1; i < BSIZE; i++) {
            diag->val[i * BSIZE + k] /= diag->val[k * BSIZE + k];
            for (j = k + 1; j < BSIZE; j++) {
                diag->val[i * BSIZE + j] -= diag->val[i * BSIZE + k] * diag->val[k * BSIZE + j];
            }
        }
    }
}

void cpu_bdiv(block_t *diag, block_t *row) {
    int i, j, k;
    for (i = 0; i < BSIZE; i++) {
        for (k = 0; k < BSIZE; k++) {
            for (j = k + 1; j < BSIZE; j++) {
                row->val[i * BSIZE + j] -= row->val[i * BSIZE + k] * diag->val[k * BSIZE + j];
            }
        }
    }
}

void cpu_bmod(block_t *col, block_t *row) {
    int i, j, k;
    for (i = 0; i < BSIZE; i++) {
        for (k = 0; k < BSIZE; k++) {
            for (j = 0; j < BSIZE; j++) {
                row->val[i * BSIZE + j] -= col->val[i * BSIZE + k] * row->val[k * BSIZE + j];
            }
        }
    }
}

void cpu_fwd(block_t *diag, block_t *col, block_t *row, block_t *ik) {
    int i, j, k;
    for (i = 0; i < BSIZE; i++) {
        for (j = 0; j < BSIZE; j++) {
            for (k = 0; k < BSIZE; k++) {
                ik->val[i * BSIZE + j] -= col->val[i * BSIZE + k] * row->val[k * BSIZE + j];
            }
        }
    }
}

block_t **allocate_memory_matrix(int n) {
    block_t **matrix = (block_t **) malloc(n * n * sizeof(block_t *));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (block_t *) malloc(sizeof(block_t));
            matrix[i * n + j]->val = (double *) calloc(BSIZE * BSIZE, sizeof(double));
        }
    }
    return matrix;
}

void init_matrix(block_t **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < BSIZE * BSIZE; k++) {
                matrix[i * n + j]->val[k] = ((double)rand() / (RAND_MAX)) + 1.0;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc > 1) NB = atoi(argv[1]);
    if (argc > 2) M = atoi(argv[2]);

    printf("SparseLU Sequencial (C Puro): Matriz %d x %d blocos (Bloco: %d)\n", NB, NB, BSIZE);

    block_t **matrix = allocate_memory_matrix(NB);
    init_matrix(matrix, NB);

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    // Lógica Sequencial Padrão
    for (int k = 0; k < NB; k++) {
        cpu_lu0(matrix[k * NB + k]);

        for (int j = k + 1; j < NB; j++) {
            if (matrix[k * NB + j]->val != NULL) {
                cpu_bdiv(matrix[k * NB + k], matrix[k * NB + j]);
            }
        }

        for (int i = k + 1; i < NB; i++) {
            if (matrix[i * NB + k]->val != NULL) {
                cpu_bmod(matrix[k * NB + k], matrix[i * NB + k]);
            }
        }

        for (int i = k + 1; i < NB; i++) {
            for (int j = k + 1; j < NB; j++) {
                if (matrix[i * NB + k]->val != NULL && matrix[k * NB + j]->val != NULL) {
                    cpu_fwd(matrix[k * NB + k], matrix[i * NB + k], matrix[k * NB + j], matrix[i * NB + j]);
                }
            }
        }
    }

    gettimeofday(&stop, NULL);
    double duration = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) * 1e-6;
    printf("Tempo total (Sequencial): %.3f segundos\n", duration);

    return 0;
}