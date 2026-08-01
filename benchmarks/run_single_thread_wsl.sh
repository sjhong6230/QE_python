#!/bin/sh
set -eu

run_dir=/home/sjhong6230/QE_porting_test/Si_sym_PZ/python
python_bin=/home/sjhong6230/miniconda3/envs/DFT/bin/python

cd "$run_dir"
export PYTHONPATH=/mnt/c/Users/sjhon/QE_porting
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DOMAIN_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

if [ "$#" -eq 0 ]; then
    set -- 1 2 3 4
fi

: > single_thread_benchmark.progress
for ranks in "$@"
do
    echo "START ranks=$ranks" >> single_thread_benchmark.progress
    /usr/bin/time \
        -f "external_wall=%e maxrss_kb=%M" \
        -o "Si.scf.thread1.$ranks.time" \
        mpirun --bind-to none -np "$ranks" \
        "$python_bin" -m qepy_pw -in Si.scf.in \
        > "Si.scf.thread1.$ranks.out"
    echo "DONE ranks=$ranks" >> single_thread_benchmark.progress
done
echo COMPLETE >> single_thread_benchmark.progress
