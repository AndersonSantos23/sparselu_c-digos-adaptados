date=`date +%d_%m_%Y_%Hh_%Mmin_%Ss`
HNAMES=`hostname`

expdir="${HNAMES}_$date"

mkdir -p $expdir

BENCHS="mm.py"  # ajustar aqui com o nome da app que tu for usar
CLASSES="500 1000" # pode colocar  aqui os tamanhos

# guardando uma copia do script utilizado
estescript=`realpath "$0"`
cp $estescript $expdir/

# Case 1: on/on

# turbo boost on
sudo /usr/local/bin/off_turbo.sh
# HT on
sudo /usr/local/bin/off_ht.sh
# performance governor
sudo /usr/local/bin/gov_performance.sh

source ~/spack/share/spack/setup-env.sh # ajustar

# por aqui os loads do spack
, 

pushd $expdir

THREADS_LIST=(1 2 4 8 12 16 24)

for r in `seq 3`; do  # 3 repeticoes
    for threads in "${THREADS_LIST[@]}"; do
	for class in $CLASSES; do
	    for bench in $BENCHS; do
		output=tcc:$threads-turbo:on-ht:on-$bench:$class-Rep:$r
		echo "Running $output"
		if [ -f "$output" ]; then
		    echo "$output existe e regular"
		    if [ ! -s "$output" ]; then   # arquivo existe mas ta vazio
			rm -f $output
		    else
			echo "Skip this case ($output)"
			continue # pular ja que existe e nao ta vazio
		    fi
		fi

                # ajstar a linha de execucao
		PYTHON_GIL=0 STARPUPY_OWN_GIL=1 STARPU_NCPU=$threads python3 $bench $class > $output.appout;

		tail $output.appout
		echo "acabou a medicao"
	    done;
	done;
    done;
done;

popd
