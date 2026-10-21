#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define TOLERANCE 1e-4

typedef struct _block_t {
    double *restrict val;
} block_t;

int NB = 10;
int BSIZE = 64;

void cpu_lu0(block_t *diag, int bsize) {
    int i, j, k;
    for (k = 0; k < bsize; k++) {
        for (i = k + 1; i < bsize; i++) {
            diag->val[i * bsize + k] /= diag->val[k * bsize + k];
            for (j = k + 1; j < bsize; j++) {
                diag->val[i * bsize + j] -= diag->val[i * bsize + k] * diag->val[k * bsize + j];
            }
        }
    }
}

void cpu_bdiv(block_t *diag, block_t *row, int bsize) {
    int i, j, k;
    for (i = 0; i < bsize; i++) {
        for (k = 0; k < bsize; k++) {
            for (j = k + 1; j < bsize; j++) {
                row->val[i * bsize + j] -= row->val[i * bsize + k] * diag->val[k * bsize + j];
            }
        }
    }
}

void cpu_bmod(block_t *diag, block_t *col, int bsize) {
    int i, j, k;
    for (i = 0; i < bsize; i++) {
        for (k = 0; k < bsize; k++) {
            for (j = 0; j < bsize; j++) {
                col->val[i * bsize + j] -= col->val[i * bsize + k] * diag->val[k * bsize + j];
            }
        }
    }
}

void cpu_fwd(block_t *col, block_t *row, block_t *ik, int bsize) {
    int i, j, k;
    for (i = 0; i < bsize; i++) {
        for (j = 0; j < bsize; j++) {
            for (k = 0; k < bsize; k++) {
                ik->val[i * bsize + j] -= col->val[i * bsize + k] * row->val[k * bsize + j];
            }
        }
    }
}

// Função geradora unificada (genmat): Aloca e inicializa a matriz com uma seed
block_t **genmat(int n, int bsize, unsigned int seed) {
    block_t **matrix = (block_t **) malloc(n * n * sizeof(block_t *));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (block_t *) malloc(sizeof(block_t));
            matrix[i * n + j]->val = (double *) calloc(bsize * bsize, sizeof(double));
        }
    }

    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < bsize * bsize; k++) {
                matrix[i * n + j]->val[k] = ((double)rand() / (RAND_MAX)) + 1.0;
            }
        }
    }

    return matrix;
}

void cpu_sparselu_seq(block_t **matrix, int n, int bsize) {
    for (int k = 0; k < n; k++) {
        cpu_lu0(matrix[k * n + k], bsize);
        for (int j = k + 1; j < n; j++) {
            cpu_bdiv(matrix[k * n + k], matrix[k * n + j], bsize);
        }
        for (int i = k + 1; i < n; i++) {
            cpu_bmod(matrix[k * n + k], matrix[i * n + k], bsize);
        }
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                cpu_fwd(matrix[i * n + k], matrix[k * n + j], matrix[i * n + j], bsize);
            }
        }
    }
}

int verify_matrix(block_t **matrix_par, block_t **matrix_ref, int n, int bsize) {
    double max_diff = 0.0;
    for (int i = 0; i < n * n; i++) {
        for (int k = 0; k < bsize * bsize; k++) {
            double diff = fabs(matrix_par[i]->val[k] - matrix_ref[i]->val[k]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > TOLERANCE) {
                printf("\n[AVISO] Maior desvio encontrado: %e (acima da tolerância %e)\n", max_diff, TOLERANCE);
                return 0;
            }
        }
    }
    return 1;
}

void free_matrix(block_t **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            free(matrix[i * n + j]->val);
            free(matrix[i * n + j]);
        }
    }
    free(matrix);
}

int main(int argc, char **argv) {
    if (argc > 1) NB = atoi(argv[1]);
    if (argc > 2) BSIZE = atoi(argv[2]);

    printf("SparseLU OpenMP (C) | Matriz: %dx%d blocos | Bloco: %dx%d\n", NB, NB, BSIZE, BSIZE);

    unsigned int seed = 42;

    // Geração da matriz paralela e de referência usando a nova função genmat
    block_t **matrix = genmat(NB, BSIZE, seed);
    block_t **matrix_ref = genmat(NB, BSIZE, seed);

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int k = 0; k < NB; k++) {
                #pragma omp task depend(inout: matrix[k * NB + k]->val[0])
                cpu_lu0(matrix[k * NB + k], BSIZE);

                for (int j = k + 1; j < NB; j++) {
                    #pragma omp task depend(in: matrix[k * NB + k]->val[0]) depend(inout: matrix[k * NB + j]->val[0])
                    cpu_bdiv(matrix[k * NB + k], matrix[k * NB + j], BSIZE);
                }

                for (int i = k + 1; i < NB; i++) {
                    #pragma omp task depend(in: matrix[k * NB + k]->val[0]) depend(inout: matrix[i * NB + k]->val[0])
                    cpu_bmod(matrix[k * NB + k], matrix[i * NB + k], BSIZE);
                }

                for (int i = k + 1; i < NB; i++) {
                    for (int j = k + 1; j < NB; j++) {
                        #pragma omp task depend(in: matrix[i * NB + k]->val[0], matrix[k * NB + j]->val[0]) depend(inout: matrix[i * NB + j]->val[0])
                        cpu_fwd(matrix[i * NB + k], matrix[k * NB + j], matrix[i * NB + j], BSIZE);
                    }
                }
            }
        }
    }

    gettimeofday(&stop, NULL);
    double duration = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) * 1e-6;

    cpu_sparselu_seq(matrix_ref, NB, BSIZE);
    int is_correct = verify_matrix(matrix, matrix_ref, NB, BSIZE);

    printf("Tempo total (OpenMP Paralelo): %.4f segundos\n", duration);
    if (is_correct) {
        printf("Verificação de Corretude: SUCESSO (Correto)\n");
    } else {
        printf("Verificação de Corretude: FALHA (Incorreto)\n");
    }

    free_matrix(matrix, NB);
    free_matrix(matrix_ref, NB);

    return 0;
}