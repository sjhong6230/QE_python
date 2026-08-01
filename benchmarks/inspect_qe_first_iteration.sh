#!/bin/sh
set -eu

input_dir=${1:-/home/sjhong6230/QE_porting_test/Si_sym_PZ/QE}
qe_bin=${2:-/home/sjhong6230/qe-7.5/PW/src/pw.x}
ranks=${3:-1}
starting_wfc=${4:-atomic+random}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

cd "$input_dir"
sed \
    -e 's/electron_maxstep = 300/electron_maxstep = 1/' \
    -e "/&electrons/a\\  startingwfc = '$starting_wfc'" \
    -e "s/prefix = 'Si'/prefix = 'Si_diag'/" \
    -e "s#outdir = './temp'#outdir = '$scratch'#" \
    Si.scf.in |
    mpirun --bind-to none -np "$ranks" "$qe_bin" |
    grep -E \
        'iteration #|ethr =|total energy|estimated scf accuracy|avg # of iterations'
