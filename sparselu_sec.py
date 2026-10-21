#!/usr/bin/env python3
import time
import argparse
import numpy as np
import warnings

# Silencia warnings de ponto flutuante para manter o terminal limpo
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==========================================================
# ARGUMENTOS E FUNÇÕES AUXILIARES
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SparseLU Sequencial com Verificação (Python)")
    parser.add_argument("matrix_size", nargs="?", type=int, default=10, help="Tamanho da matriz de blocos (NB)")
    parser.add_argument("submatrix_size", nargs="?", type=int, default=64, help="Tamanho de cada bloco (BSIZE)")
    return parser.parse_args()

def print_summary(params, times, verified=None):
    avg_time = sum(times) / len(times)
    print(f"\n--- Resumo SparseLU (SEQUENCIAL) ---")
    print(f"Matriz: {params.matrix_size}x{params.matrix_size} blocos | Bloco: {params.submatrix_size}x{params.submatrix_size}")
    if verified is not None:
        status = "SUCESSO (Correto)" if verified else "FALHA (Incorreto)"
        print(f"Verificação de Corretude: {status}")
    print(f"Tempos iterativos: {[f'{t:.4f}s' for t in times]}")
    print(f"Tempo médio: {avg_time:.4f} segundos")

def allocate_clean_block(submatrix_size):
    return np.zeros((submatrix_size, submatrix_size), dtype=np.double)

def genmat(matrix_size, submatrix_size):
    BENCH = {}
    init_val = 1325

    for ii in range(matrix_size):
        for jj in range(matrix_size):
            null_entry = 0
            if (ii < jj) and (ii % 3 != 0): null_entry = 1
            if (ii > jj) and (jj % 3 != 0): null_entry = 1
            if ii % 2 == 1: null_entry = 1
            if jj % 2 == 1: null_entry = 1
            if ii == jj: null_entry = 0
            if ii == jj - 1: null_entry = 0
            if ii - 1 == jj: null_entry = 0

            if null_entry == 0:
                block = np.zeros((submatrix_size, submatrix_size), dtype=np.double)
                for i in range(submatrix_size):
                    for j in range(submatrix_size):
                        init_val = (3125 * init_val) % 65536
                        block[i, j] = float((init_val - 32768.0) / 16384.0)
                BENCH[(ii, jj)] = block
            else:
                BENCH[(ii, jj)] = None

    return BENCH

# ==========================================================
# KERNELS MATEMÁTICOS (Com proteção contra divisão por zero)
# ==========================================================
def lu0(diag, BSIZE):
    for k in range(BSIZE):
        pivot = diag[k, k]
        if abs(pivot) < 1e-14:
            pivot = 1e-14 if pivot >= 0 else -1e-14
            diag[k, k] = pivot

        for i in range(k + 1, BSIZE):
            diag[i, k] /= pivot
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
            pivot = diag[k, k]
            if abs(pivot) < 1e-14:
                pivot = 1e-14 if pivot >= 0 else -1e-14
            
            col[i, k] = col[i, k] / pivot
            for j in range(k + 1, BSIZE):
                col[i, j] -= col[i, k] * diag[k, j]

def bmod(col, row, ik, BSIZE):
    for i in range(BSIZE):
        for j in range(BSIZE):
            for k in range(BSIZE):
                ik[i, j] -= col[i, k] * row[k, j]

# ==========================================================
# LÓGICA DE EXECUÇÃO SEQUENCIAL
# ==========================================================
def sparselu_seq_call(BENCH, matrix_size, submatrix_size):
    for kk in range(matrix_size):
        diag_key = (kk, kk)
        if diag_key in BENCH and BENCH[diag_key] is not None:
            lu0(BENCH[diag_key], submatrix_size)

        for jj in range(kk + 1, matrix_size):
            key = (kk, jj)
            if key in BENCH and BENCH[key] is not None:
                fwd(BENCH[kk, kk], BENCH[key], submatrix_size)

        for ii in range(kk + 1, matrix_size):
            key = (ii, kk)
            if key in BENCH and BENCH[key] is not None:
                bdiv(BENCH[kk, kk], BENCH[key], submatrix_size)

        for ii in range(kk + 1, matrix_size):
            if (ii, kk) in BENCH and BENCH[(ii, kk)] is not None:
                for jj in range(kk + 1, matrix_size):
                    if (kk, jj) in BENCH and BENCH[(kk, jj)] is not None:
                        target_key = (ii, jj)
                        if target_key not in BENCH or BENCH[target_key] is None:
                            BENCH[target_key] = allocate_clean_block(submatrix_size)
                        bmod(BENCH[ii, kk], BENCH[kk, jj], BENCH[target_key], submatrix_size)

def verify_results(matrix_size, submatrix_size, result_bench):
    """Compara o resultado e exibe a diferença máxima caso ultrapasse a tolerância."""
    ref_bench = genmat(matrix_size, submatrix_size)
    sparselu_seq_call(ref_bench, matrix_size, submatrix_size)

    max_diff = 0.0
    all_keys = set(result_bench.keys()).union(set(ref_bench.keys()))
    for key in all_keys:
        block_res = result_bench.get(key)
        block_ref = ref_bench.get(key)

        if block_res is None and block_ref is None:
            continue
        if block_res is None or block_ref is None:
            return False
        
        diff = np.max(np.abs(block_res - block_ref))
        if diff > max_diff:
            max_diff = diff

    is_ok = max_diff < 1e-4
    if not is_ok:
        print(f"\n[AVISO] Verificação sequencial falhou. Maior desvio encontrado: {max_diff}")
    return is_ok

def main():
    params = parse_args()
    
    warmup_bench = genmat(params.matrix_size, params.submatrix_size)
    sparselu_seq_call(warmup_bench, params.matrix_size, params.submatrix_size)
    is_correct = verify_results(params.matrix_size, params.submatrix_size, warmup_bench)

    BENCH = genmat(params.matrix_size, params.submatrix_size)
    t0 = time.perf_counter_ns()
    sparselu_seq_call(BENCH, params.matrix_size, params.submatrix_size)
    t1 = time.perf_counter_ns()
    times = [(t1 - t0) / 1e9]

    print_summary(params, times, verified=is_correct)

if __name__ == "__main__":
    main()