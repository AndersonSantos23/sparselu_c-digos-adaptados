#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define BSIZE 64

typedef struct _block_t {
    double *restrict val;
} block_t;

int NB = 10;

// Kernels Computacionais Puros (sem wrappers do StarPU)
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

void cpu_bmod(block_t *diag, block_t *col) {
    int i, j, k;
    for (i = 0; i < BSIZE; i++) {
        for (k = 0; k < BSIZE; k++) {
            for (j = 0; j < BSIZE; j++) {
                col->val[i * BSIZE + j] -= col->val[i * BSIZE + k] * diag->val[k * BSIZE + j];
            }
        }
    }
}

void cpu_fwd(block_t *col, block_t *row, block_t *ik) {
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

    printf("SparseLU OpenMP (Paralelo): Matriz %d x %d blocos (Bloco: %d)\n", NB, NB, BSIZE);

    block_t **matrix = allocate_memory_matrix(NB);
    init_matrix(matrix, NB);

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    // Grafo de Tarefas Dinâmico com OpenMP Task Dependencies
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int k = 0; k < NB; k++) {
                
                // 1. Tarefa LU0
                #pragma omp task depend(inout: matrix[k * NB + k]->val[0])
                cpu_lu0(matrix[k * NB + k]);

                // 2. Tarefas BDIV
                for (int j = k + 1; j < NB; j++) {
                    #pragma omp task depend(in: matrix[k * NB + k]->val[0]) depend(inout: matrix[k * NB + j]->val[0])
                    cpu_bdiv(matrix[k * NB + k], matrix[k * NB + j]);
                }

                // 3. Tarefas BMOD
                for (int i = k + 1; i < NB; i++) {
                    #pragma omp task depend(in: matrix[k * NB + k]->val[0]) depend(inout: matrix[i * NB + k]->val[0])
                    cpu_bmod(matrix[k * NB + k], matrix[i * NB + k]);
                }

                // 4. Tarefas FWD (Modificação Principal)
                for (int i = k + 1; i < NB; i++) {
                    for (int j = k + 1; j < NB; j++) {
                        #pragma omp task depend(in: matrix[i * NB + k]->val[0], matrix[k * NB + j]->val[0]) depend(inout: matrix[i * NB + j]->val[0])
                        cpu_fwd(matrix[i * NB + k], matrix[k * NB + j], matrix[i * NB + j]);
                    }
                }
            }
        }
    }

    gettimeofday(&stop, NULL);
    double duration = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) * 1e-6;
    printf("Tempo total (OpenMP Paralelo): %.3f segundos\n", duration);

    // Liberação de memória
    for (int i = 0; i < NB; i++) {
        for (int j = 0; j < NB; j++) {
            free(matrix[i * NB + j]->val);
            free(matrix[i * NB + j]);
        }
    }
    free(matrix);

    return 0;
}