import sys
import math
import time
import argparse
import statistics

EPSILON = 1.0E-6

class UserParameters:
    def __init__(self):
        self.check = 0
        self.succeed = 0
        self.string2display = None
        self.niter = 1
        self.matrix_size = 64
        self.submatrix_size = 64

def allocate_clean_block(submatrix_size):
    return [[0.0 for _ in range(submatrix_size)] for _ in range(submatrix_size)]

def genmat(matrix_size, submatrix_size):
    M = [[None for _ in range(matrix_size)] for _ in range(matrix_size)]
    init_val = 1325

    for ii in range(matrix_size):
        for jj in range(matrix_size):
            null_entry = 0
            if (ii < jj) and (ii % 3 != 0):
                null_entry = 1
            if (ii > jj) and (jj % 3 != 0):
                null_entry = 1
            if ii % 2 == 1:
                null_entry = 1
            if jj % 2 == 1:
                null_entry = 1
            if ii == jj:
                null_entry = 0
            if ii == jj - 1:
                null_entry = 0
            if ii - 1 == jj:
                null_entry = 0

            if null_entry == 0:
                block = [[0.0 for _ in range(submatrix_size)] for _ in range(submatrix_size)]
                for i in range(submatrix_size):
                    for j in range(submatrix_size):
                        init_val = (3125 * init_val) % 65536
                        block[i][j] = float((init_val - 32768.0) / 16384.0)
                M[ii * matrix_size + jj] = block
            else:
                M[ii * matrix_size + jj] = None
    return M

def lu0(diag, submatrix_size):
    for k in range(submatrix_size):
        for i in range(k + 1, submatrix_size):
            diag[i][k] = diag[i][k] / diag[k][k]
            for j in range(k + 1, submatrix_size):
                diag[i][j] = diag[i][j] - diag[i][k] * diag[k][j]

def bdiv(diag, row, submatrix_size):
    for i in range(submatrix_size):
        for k in range(submatrix_size):
            row[i][k] = row[i][k] / diag[k][k]
            for j in range(k + 1, submatrix_size):
                row[i][j] = row[i][j] - row[i][k] * diag[k][j]

def bmod(row, col, inner, submatrix_size):
    for i in range(submatrix_size):
        for j in range(submatrix_size):
            for k in range(submatrix_size):
                inner[i][j] = inner[i][j] - row[i][k] * col[k][j]

def fwd(diag, col, submatrix_size):
    for j in range(submatrix_size):
        for k in range(submatrix_size):
            for i in range(k + 1, submatrix_size):
                col[i][j] = col[i][j] - diag[i][k] * col[k][j]

def checkmat(M, N, submatrix_size):
    for i in range(submatrix_size):
        for j in range(submatrix_size):
            r_err = M[i][j] - N[i][j]
            if r_err == 0.0:
                continue
            if r_err < 0.0:
                r_err = -r_err
            if M[i][j] == 0:
                return 0
            r_err = r_err / M[i][j]
            if r_err > EPSILON:
                return 0
    return 1

def sparselu_check(BENCH_SEQ, BENCH, matrix_size, submatrix_size):
    ok = 1
    for ii in range(matrix_size):
        if not ok:
            break
        for jj in range(matrix_size):
            if not ok:
                break
            idx = ii * matrix_size + jj
            if (BENCH_SEQ[idx] is None) and (BENCH[idx] is not None):
                ok = 0
            elif (BENCH_SEQ[idx] is not None) and (BENCH[idx] is None):
                ok = 0
            elif (BENCH_SEQ[idx] is not None) and (BENCH[idx] is not None):
                ok = checkmat(BENCH_SEQ[idx], BENCH[idx], submatrix_size)
    return ok

def sparselu_seq_call(BENCH, matrix_size, submatrix_size):
    for kk in range(matrix_size):
        lu0(BENCH[kk * matrix_size + kk], submatrix_size)
        
        for jj in range(kk + 1, matrix_size):
            if BENCH[kk * matrix_size + jj] is not None:
                fwd(BENCH[kk * matrix_size + kk], BENCH[kk * matrix_size + jj], submatrix_size)
                
        for ii in range(kk + 1, matrix_size):
            if BENCH[ii * matrix_size + kk] is not None:
                bdiv(BENCH[kk * matrix_size + kk], BENCH[ii * matrix_size + kk], submatrix_size)
                
        for ii in range(kk + 1, matrix_size):
            if BENCH[ii * matrix_size + kk] is not None:
                for jj in range(kk + 1, matrix_size):
                    if BENCH[kk * matrix_size + jj] is not None:
                        idx_ij = ii * matrix_size + jj
                        if BENCH[idx_ij] is None:
                            BENCH[idx_ij] = allocate_clean_block(submatrix_size)
                        bmod(BENCH[ii * matrix_size + kk], BENCH[kk * matrix_size + jj], BENCH[idx_ij], submatrix_size)

def run(params):
    matrix_size = params.matrix_size
    if matrix_size <= 0:
        matrix_size = 64
        params.matrix_size = matrix_size
    
    submatrix_size = params.submatrix_size
    if submatrix_size <= 0:
        submatrix_size = 64
        params.submatrix_size = submatrix_size

    BENCH = genmat(matrix_size, submatrix_size)

    start_time = time.perf_counter()
    sparselu_seq_call(BENCH, matrix_size, submatrix_size)
    end_time = time.perf_counter()
    wtime = end_time - start_time

    if params.check:
        BENCH_SEQ = genmat(matrix_size, submatrix_size)
        sparselu_seq_call(BENCH_SEQ, matrix_size, submatrix_size)
        params.succeed = sparselu_check(BENCH_SEQ, BENCH, matrix_size, submatrix_size)
    else:
        params.succeed = 1

    return wtime, 0, 0

def main():
    parser = argparse.ArgumentParser(description="KaStORS SparsLU Python Port")
    parser.add_argument("-c", action="store_true", help="Ask to check result")
    parser.add_argument("-i", type=int, default=1, help="Number of iterations")
    parser.add_argument("-n", type=int, default=64, help="Matrix size")
    parser.add_argument("-m", type=int, default=64, help="SubMatrix size")
    
    args, unknown = parser.parse_known_args()

    params = UserParameters()
    params.check = 1 if args.c else 0
    params.niter = args.i
    params.matrix_size = args.n
    params.submatrix_size = args.m

    # Warmup
    run(params)

    all_times = []
    mean = 0.0
    meansqr = 0.0
    min_ = float('inf')
    max_ = -1.0

    for _ in range(params.niter):
        cur_time, _, _ = run(params)
        all_times.append(cur_time)
        mean += cur_time
        min_ = min(min_, cur_time)
        max_ = max(max_, cur_time)
        meansqr += cur_time * cur_time
        time.sleep(5)

    mean /= params.niter
    meansqr /= params.niter
    stddev = math.sqrt(meansqr - mean * mean)

    all_times.sort()
    median = statistics.median(all_times)

    progname = "sparselu.py"
    
    print("#Output follows this header :")
    print("##Progname Size Submatrix_size Iterations Threads Time(Mean,sec) startTime endTime Stddev")
    print(f"{progname} {params.matrix_size} {params.submatrix_size} {params.niter} 1 {mean:.6f} 0 0 {stddev:.6f}")
    print(f"#Experience summary : avg : {mean:.6f} :: std : {stddev:.6f} :: min : {min_:.6f} :: max : {max_:.6f} :: median : {median:.6f}")
    
    if params.check:
        status = "success" if params.succeed == 1 else ("not implemented" if params.succeed > 1 else "fail")
        print(f"Check : {status}")
    
    if params.string2display:
        print(params.string2display)

if __name__ == "__main__":
    main()