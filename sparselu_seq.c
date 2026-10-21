#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define TOLERANCE 1e-4

typedef struct _block_t {
    double *restrict val;
} block_t;

int NB = 10;
int BSIZE = 64;

// Kernels Computacionais Sequenciais (usando BSIZE dinâmico)
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

void cpu_bmod(block_t *col, block_t *row, int bsize) {
    int i, j, k;
    for (i = 0; i < bsize; i++) {
        for (k = 0; k < bsize; k++) {
            for (j = 0; j < bsize; j++) {
                row->val[i * bsize + j] -= col->val[i * bsize + k] * row->val[k * bsize + j];
            }
        }
    }
}

void cpu_fwd(block_t *diag, block_t *col, block_t *row, block_t *ik, int bsize) {
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

void run_sparselu_seq(block_t **matrix, int n, int bsize) {
    for (int k = 0; k < n; k++) {
        cpu_lu0(matrix[k * n + k], bsize);

        for (int j = k + 1; j < n; j++) {
            if (matrix[k * n + j]->val != NULL) {
                cpu_bdiv(matrix[k * n + k], matrix[k * n + j], bsize);
            }
        }

        for (int i = k + 1; i < n; i++) {
            if (matrix[i * n + k]->val != NULL) {
                cpu_bmod(matrix[k * n + k], matrix[i * n + k], bsize);
            }
        }

        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                if (matrix[i * n + k]->val != NULL && matrix[k * n + j]->val != NULL) {
                    cpu_fwd(matrix[k * n + k], matrix[i * n + k], matrix[k * n + j], matrix[i * n + j], bsize);
                }
            }
        }
    }
}

int verify_matrix(block_t **matrix_res, block_t **matrix_ref, int n, int bsize) {
    double max_diff = 0.0;
    for (int i = 0; i < n * n; i++) {
        for (int k = 0; k < bsize * bsize; k++) {
            double diff = fabs(matrix_res[i]->val[k] - matrix_ref[i]->val[k]);
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

    printf("SparseLU Sequencial (C) | Matriz: %dx%d blocos | Bloco: %dx%d\n", NB, NB, BSIZE, BSIZE);

    unsigned int seed = 42;

    // Geração das matrizes usando a função genmat unificada
    block_t **matrix = genmat(NB, BSIZE, seed);
    block_t **matrix_ref = genmat(NB, BSIZE, seed);

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    run_sparselu_seq(matrix, NB, BSIZE);

    gettimeofday(&stop, NULL);
    double duration = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) * 1e-6;

    run_sparselu_seq(matrix_ref, NB, BSIZE);
    int is_correct = verify_matrix(matrix, matrix_ref, NB, BSIZE);

    printf("Tempo total (Sequencial): %.4f segundos\n", duration);
    if (is_correct) {
        printf("Verificação de Corretude: SUCESSO (Correto)\n");
    } else {
        printf("Verificação de Corretude: FALHA (Incorreto)\n");
    }

    free_matrix(matrix, NB);
    free_matrix(matrix_ref, NB);

    return 0;
}