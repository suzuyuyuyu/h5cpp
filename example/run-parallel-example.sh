#!/bin/bash

#========== Slurm Option ==========
#SBATCH -p gr10353b
#SBATCH -t 00:10:00
#SBATCH -J h5cpp-example
#SBATCH --rsc p=4:t=1:c=1
#SBATCH -o stdout/%x.%j
#SBATCH -e stderr/%x.%j

#========= Shell Script ==========
#
# Runs the parallel examples. Like every MPI program here they must go through
# the batch system, never a login node.
#
#   cmake -S example -B example/build/intel-mpi \
#         -DCMAKE_PREFIX_PATH="<parallel h5cpp>;<parallel h5c>;<parallel hdf5>" \
#         -DCMAKE_CXX_COMPILER=mpiicpx
#   cmake --build example/build/intel-mpi --target examples
#   sbatch example/run-parallel-example.sh
#
# Submit from the repository root. stdout/ and stderr/ must already exist:
# Slurm opens them before this script runs.

set -euo pipefail
set -x

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

. /usr/share/Modules/init/bash
export LD_LIBRARY_PATH=${HOME}/.local/opt/intel/phdf5-2.1.0/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1

BUILD_DIR="${BUILD_DIR:-example/build/intel-mpi}"
NRANKS="${NRANKS:-${SLURM_NTASKS:-2}}"

# Executables only: the glob would otherwise pick up the .h5 files the
# examples leave behind in the same directory.
shopt -s nullglob
binaries=()
for f in "${BUILD_DIR}"/example_parallel*; do
    [[ -f "$f" && -x "$f" ]] && binaries+=("$f")
done
shopt -u nullglob

if [[ ${#binaries[@]} -eq 0 ]]; then
    set +x
    echo "error: no example_parallel* binaries in ${BUILD_DIR}." >&2
    echo "The examples need an h5c built with H5C_ENABLE_PARALLEL=ON." >&2
    exit 1
fi

cd "${BUILD_DIR}"
for exe in "${binaries[@]}"; do
    echo "=== $(basename "${exe}") on ${NRANKS} ranks ==="
    srun -n "${NRANKS}" "./$(basename "${exe}")"
done
