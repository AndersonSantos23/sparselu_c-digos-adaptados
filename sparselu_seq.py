import sys
import time
import math
import numpy as np

EPSILON = 1.0E-6

class UserParameters:
    def __init__(self):
        self.check = 0
        self.succeed = 0
        self.string2display = None
        self.niter = 1
        self.matrix_size = 64
        self.submatrix_size = 64
        self.blocksize = 0
        self.iblocksize = 0
        self.cutoff_depth = 0
        self.cutoff_size = 0


def allocate_clean_block(submatrix_size):
    """Equivalente a allocate_clean_block em C."""
    return np.zeros(submatrix_size * submatrix_size, dtype=np.float32)


def lu0(diag, submatrix_size):
    """Fatoração LU no bloco diagonal."""
    for k in range(submatrix_size):
        for i in range(k + 1, submatrix_size):
            diag[i * submatrix_size + k] /= diag[k * submatrix_size + k]
            for j in range(k + 1, submatrix_size):
                diag[i * submatrix_size + j] -= diag[i * submatrix_size + k] * diag[k * submatrix_size + j]


def bdiv(diag, row, submatrix_size):
    """Divisão de bloco."""
    for i in range(submatrix_size):
        for k in range(submatrix_size):
            row[i * submatrix_size + k] /= diag[k * submatrix_size + k]
            for j in range(k + 1, submatrix_size):
                row[i * submatrix_size + j] -= row[i * submatrix_size + k] * diag[k * submatrix_size + j]


def bmod(row, col, inner, submatrix_size):
    """Modificação de bloco."""
    for i in range(submatrix_size):
        for j in range(submatrix_size):
            sum_val = 0.0
            for k in range(submatrix_size):
                sum_val += row[i * submatrix_size + k] * col[k * submatrix_size + j]
            inner[i * submatrix_size + j] -= sum_val


def fwd(diag, col, submatrix_size):
    """Substituição progressiva (Forward)."""
    for j in range(submatrix_size):
        for k in range(submatrix_size):
            for i in range(k + 1, submatrix_size):
                col[i * submatrix_size + j] -= diag[i * submatrix_size + k] * col[k * submatrix_size + j]


def genmat(M, matrix_size, submatrix_size):
    """Gera a estrutura e os valores da matriz esparsa."""
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
                block = np.empty(submatrix_size * submatrix_size, dtype=np.float32)
                p_idx = 0
                for i in range(submatrix_size):
                    for j in range(submatrix_size):
                        init_val = (3125 * init_val) % 65536
                        block[p_idx] = float((init_val - 32768.0) / 16384.0)
                        p_idx += 1
                M[ii * matrix_size + jj] = block
            else:
                M[ii * matrix_size + jj] = None


def sparselu_init(matrix_size, submatrix_size):
    """Inicializa a estrutura da matriz esparsa."""
    BENCH = [None] * (matrix_size * matrix_size)
    genmat(BENCH, matrix_size, submatrix_size)
    return BENCH


def checkmat(M, N, submatrix_size):
    """Verifica a precisão entre dois blocos de matriz."""
    for i in range(submatrix_size):
        for j in range(submatrix_size):
            idx = i * submatrix_size + j
            r_err = M[idx] - N[idx]
            if r_err == 0.0:
                continue
            if r_err < 0.0:
                r_err = -r_err
            if M[idx] == 0.0:
                return 0
            r_err = r_err / M[idx]
            if r_err > EPSILON:
                return 0
    return 1


def sparselu_check(BENCH_SEQ, BENCH, matrix_size, submatrix_size):
    """Verifica a integridade de toda a matriz resultante."""
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
    """Chamada sequencial principal do Sparse LU."""
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
                        if BENCH[ii * matrix_size + jj] is None:
                            BENCH[ii * matrix_size + jj] = allocate_clean_block(submatrix_size)
                        bmod(BENCH[ii * matrix_size + kk], BENCH[kk * matrix_size + jj], BENCH[ii * matrix_size + jj], submatrix_size)


def run(params):
    """Executa o benchmark e retorna o tempo de execução e marcações."""
    matrix_size = params.matrix_size
    if matrix_size <= 0:
        matrix_size = 64
        params.matrix_size = matrix_size
    submatrix_size = params.submatrix_size
    if submatrix_size <= 0:
        submatrix_size = 64
        params.submatrix_size = submatrix_size

    BENCH = sparselu_init(matrix_size, submatrix_size)

    start_time = time.perf_counter()
    sparselu_seq_call(BENCH, matrix_size, submatrix_size)
    end_time = time.perf_counter()
    wtime = end_time - start_time

    if params.check:
        BENCH_SEQ = sparselu_init(matrix_size, submatrix_size)
        sparselu_seq_call(BENCH_SEQ, matrix_size, submatrix_size)
        params.succeed = sparselu_check(BENCH_SEQ, BENCH, matrix_size, submatrix_size)
    else:
        params.succeed = 1

    return wtime, int(start_time * 1e9), int(end_time * 1e9)


def parse(argv, params):
    """Analisa os argumentos da linha de comando."""
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-c":
            params.check = 1
        elif arg in ("--help", "-h"):
            print("----------------------------------------------")
            print("-                  KaStORS                   -")
            print("-   Kaapi Starpu OpenMP Runtime task Suite   -")
            print("----------------------------------------------")
            print("-h, --help : Show help information")
            print("-c : Ask to check result")
            print("-i : Number of iterations")
            print("-n : Matrix size")
            print("-m : SubMatrix size")
            sys.exit(0)
        elif arg == "-i":
            i += 1
            if i < len(argv):
                params.niter = int(argv[i])
            else:
                sys.stderr.write("-i requires a number\n")
                sys.exit(1)
        elif arg == "-n":
            i += 1
            if i < len(argv):
                params.matrix_size = int(argv[i])
            else:
                sys.stderr.write("-n requires a number\n")
                sys.exit(1)
        elif arg == "-m":
            i += 1
            if i < len(argv):
                params.submatrix_size = int(argv[i])
            else:
                sys.stderr.write("-m requires a number\n")
                sys.exit(1)
        else:
            sys.stderr.write(f"Unknown parameter : {arg}\n")
        i += 1


def main():
    params = UserParameters()
    params.niter = 1

    parse(sys.argv, params)

    # Warmup
    _, start_time, end_time = run(params)

    mean = 0.0
    meansqr = 0.0
    min_val = float('inf')
    max_val = -1.0
    all_times = []

    for _ in range(params.niter):
        cur_time, st, et = run(params)
        start_time, end_time = st, et
        all_times.append(cur_time)
        mean += cur_time
        min_val = min(min_val, cur_time)
        max_val = max(max_val, cur_time)
        meansqr += cur_time * cur_time
        time.sleep(5)

    mean /= params.niter
    meansqr /= params.niter
    stddev = math.sqrt(meansqr - mean * mean)

    all_times.sort()
    median = all_times[params.niter // 2]

    progname = sys.argv[0].split('/')[-1]

    print("#Output follows this header :")
    print("##Progname Size Submatrix_size Iterations Threads Time(Mean,sec) startTime endTime Stddev")
    print(f"{progname} {params.matrix_size} {params.submatrix_size} {params.niter} 1 {mean} {start_time} {end_time} {stddev}")
    print(f"#Experience summarry : avg : {mean} :: std : {stddev} :: min : {min_val} :: max : {max_val} :: median : {median}")
    
    if params.check:
        status = "success" if params.succeed == 1 else ("not implemented" if params.succeed > 1 else "fail")
        print(f"Check : {status}")
    if params.string2display is not None:
        print(params.string2display)
    print()


if __name__ == "__main__":
    main()