#!/usr/bin/env bash

# ==============================================================================
# USO:
#   ./run_experiment.sh <caminho_do_arquivo> [argumentos_do_calculo...]
#
# EXEMPLOS DE USO:
#   1. C Paralelo:     ./run_experiment.sh ./sparselu_par 10 64
#   2. C Sequencial:   ./run_experiment.sh ./sparselu_seq 10 64
#   3. Python Paralelo: ./run_experiment.sh ./sparselu_par.py 10 64
#   4. Python Seq:     ./run_experiment.sh ./sparselu_seq.py 10 64
# ==============================================================================

if [ -z "$1" ]; then
    echo "Erro: Informe o arquivo a ser executado."
    echo "Uso: $0 <caminho_do_arquivo> [argumentos...]"
    exit 1
fi

# Captura o primeiro argumento como o executável/script e o restante como parâmetros
BENCH="$1"
shift
CLASSES=("$*")  # Exemplo: "10 64"

date=`date +%d_%m_%Y_%Hh_%Mmin_%Ss`
HNAMES=`hostname`

expdir="${HNAMES}_$date"
mkdir -p "$expdir"

# Guardando uma cópia do script utilizado
estescript=`realpath "$0"`
cp "$estescript" "$expdir/"

# Ajustes de ambiente / Hardware
sudo /usr/local/bin/off_turbo.sh
sudo /usr/local/bin/off_ht.sh
sudo /usr/local/bin/gov_performance.sh

source ~/spack/share/spack/setup-env.sh 2>/dev/null || true # ajustar se necessário

origin="$PWD"
pushd "$expdir" > /dev/null

# Lista de threads para códigos paralelos
THREADS_LIST=(1 2 4 8 12 16 24)

# Identifica se é Python ou C
benchname=`basename "$BENCH"`
is_python=0
if [[ "$benchname" == *.py ]]; then
    is_python=1
fi

# Identifica se é versão sequencial
is_seq=0
if [[ "$benchname" == *seq* ]] \vert{}\vert{} [[ "$benchname" == *sec* ]]; then
    is_seq=1
fi

for r in `seq 3`; do  # 3 repetições
    # Se for sequencial, roda apenas com 1 thread. Se for paralelo, itera na lista.
    if [ $is_seq -eq 1 ]; then
        CURRENT_THREADS=(1)
    else
        CURRENT_THREADS=("${THREADS_LIST[@]}")
    fi

    for threads in "${CURRENT_THREADS[@]}"; do
        for class in "${CLASSES[@]}"; do
            classname=${class// /_}
            output="tcc:${threads}-turbo:off-ht:off-${benchname}:${classname}-Rep:${r}"
            
            echo "--------------------------------------------------------"
            echo "Running $output"

            if [ -f "$output.appout" ]; then
                if [ ! -s "$output.appout" ]; then  # Arquivo existe mas tá vazio
                    rm -f "$output.appout"
                else
                    echo "Skip this case ($output.appout já existe)"
                    continue
                fi
            fi

            # Monta o comando de execução dependendo do tipo (Python x C e Paralelo x Seq)
            if [ $is_python -eq 1 ]; then
                if [ $is_seq -eq 1 ]; then
                    # Python Sequencial: python3 script.py NB BSIZE
                    python3 "$origin/$BENCH" $class > "$output.appout" 2>&1
                else
                    # Python Paralelo: python3 script.py NB BSIZE THREADS
                    python3 "$origin/$BENCH" $class $threads > "$output.appout" 2>&1
                fi
            else
                if [ $is_seq -eq 1 ]; then
                    # C Sequencial: ./sparselu_seq NB BSIZE
                    "$origin/$BENCH" $class > "$output.appout" 2>&1
                else
                    # C Paralelo (OpenMP): OMP_NUM_THREADS=X ./sparselu_par NB BSIZE
                    OMP_NUM_THREADS=$threads "$origin/$BENCH" $class > "$output.appout" 2>&1
                fi
            fi

            tail -n 5 "$output.appout"
            echo "Acabou a medicao da repeticao $r"
        done
    done
done

popd > /dev/null