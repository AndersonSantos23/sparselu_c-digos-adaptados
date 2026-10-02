#!/usr/bin/env python3
import time
import argparse
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ==========================================================
# ARGUMENTOS E FUNÇÕES AUXILIARES
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SparseLU Paralelo com Threads (Python)")
    parser.add_argument("--matrix-size", type=int, default=10, help="Tamanho da matriz de blocos (NB)")
    parser.add_argument("--submatrix-size", type=int, default=64, help="Tamanho de cada bloco (BSIZE)")
    parser.add_argument("--nth", type=int, default=4, help="Número de threads")
    parser.add_argument("--niter", type=int, default=3, help="Número de iterações de teste")
    return parser.parse_args()

def print_summary(params, times):
    avg_time = sum(times) / len(times)
    print(f"\n--- Resumo SparseLU (PARALELO - {params.nth} Threads) ---")
    print(f"Matriz: {params.matrix_size}x{params.matrix_size} blocos | Bloco: {params.submatrix_size}x{params.submatrix_size}")
    print(f"Tempos iterativos: {[f'{t:.4f}s' for t in times]}")
    print(f"Tempo médio: {avg_time:.4f} segundos")

def allocate_clean_block(submatrix_size):
    return np.zeros((submatrix_size, submatrix_size), dtype=np.double)

def genmat(matrix_size, submatrix_size):
    BENCH = {}
    for i in range(matrix_size):
        for j in range(matrix_size):
            if (i == j) or (i == j + 1) or (j == matrix_size - 1):
                BENCH[(i, j)] = np.random.rand(submatrix_size, submatrix_size) + 1.0
    return BENCH

# ==========================================================
# KERNELS MATEMÁTICOS
# ==========================================================
def lu0(diag, BSIZE):
    for k in range(BSIZE):
        for i in range(k + 1, BSIZE):
            diag[i, k] /= diag[k, k]
            for j in range(k + 1, BSIZE):
                diag[i, j] -= diag[i, k] * diag[k, j]

def fwd(diag, row, BSIZE):
    for i in range(BSIZE):
        for k in range(BSIZE):
            for j in range(k + 1, BSIZE):
                row[i, j] -= row[i, k] * diag[k, j]

def bdiv(diag, col, BSIZE):
    for i in range(BSIZE):
        for k in range(BSIZE):
            for j in range(BSIZE):
                col[i, j] -= col[i, k] * diag[k, j]

def bmod(col, row, ik, BSIZE):
    for i in range(BSIZE):
        for j in range(BSIZE):
            for k in range(BSIZE):
                ik[i, j] -= col[i, k] * row[k, j]

# ==========================================================
# LÓGICA DE EXECUÇÃO PARALELA (TASK DEPENDENCY)
# ==========================================================
def sparselu_task_dep_call(BENCH, matrix_size, submatrix_size, num_threads):
    locks = {(i, j): threading.Lock() for i in range(matrix_size) for j in range(matrix_size)}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for kk in range(matrix_size):
            futures = []
            diag_key = (kk, kk)

            if diag_key in BENCH and BENCH[diag_key] is not None:
                def task_lu0(dk=diag_key):
                    with locks[dk]: lu0(BENCH[dk], submatrix_size)
                futures.append(executor.submit(task_lu0))

            for f in futures: f.result()
            futures.clear()

            for jj in range(kk + 1, matrix_size):
                key = (kk, jj)
                if key in BENCH and BENCH[key] is not None:
                    def task_fwd(dk=diag_key, k_key=key):
                        with locks[dk], locks[k_key]: fwd(BENCH[dk], BENCH[k_key], submatrix_size)
                    futures.append(executor.submit(task_fwd))

            for ii in range(kk + 1, matrix_size):
                key = (ii, kk)
                if key in BENCH and key is not None:
                    def task_bdiv(dk=diag_key, i_key=key):
                        with locks[dk], locks[i_key]: bdiv(BENCH[dk], BENCH[i_key], submatrix_size)
                    futures.append(executor.submit(task_bdiv))

            for f in futures: f.result()
            futures.clear()

            for ii in range(kk + 1, matrix_size):
                if (ii, kk) in BENCH and BENCH[(ii, kk)] is not None:
                    for jj in range(kk + 1, matrix_size):
                        if (kk, jj) in BENCH and BENCH[(kk, jj)] is not None:
                            target_key = (ii, jj)
                            if target_key not in BENCH or BENCH[target_key] is None:
                                BENCH[target_key] = allocate_clean_block(submatrix_size)
                            def task_bmod(ik=(ii, kk), kj=(kk, jj), target=target_key):
                                with locks[ik], locks[kj], locks[target]:
                                    bmod(BENCH[ik], BENCH[kj], BENCH[target], submatrix_size)
                            futures.append(executor.submit(task_bmod))

            for f in futures: f.result()

def main():
    params = parse_args()
    
    def run():
        BENCH = genmat(params.matrix_size, params.submatrix_size)
        t0 = time.perf_counter_ns()
        sparselu_task_dep_call(BENCH, params.matrix_size, params.submatrix_size, params.nth)
        t1 = time.perf_counter_ns()
        return (t1 - t0) / 1e9

    run() # Warmup
    times = [run() for _ in range(params.niter)]
    print_summary(params, times)

if __name__ == "__main__":
    main()