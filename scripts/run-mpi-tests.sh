#!/bin/bash

#========== Slurm Option ==========
#SBATCH -p gr10353b
#SBATCH -t 00:30:00
#SBATCH -J h5cpp-mpi
#SBATCH --rsc p=4:t=1:c=1
#SBATCH -o stdout/%x.%j
#SBATCH -e stderr/%x.%j

#========= Shell Script ==========
#
# Runs h5cpp's parallel tests. They must NEVER be launched on a login node, at
# any rank count, which is why they carry the CTest label `mpi` and are
# excluded from `quick`:
#
#   login node:   ctest --test-dir <build> -L quick    # serial only
#   batch:        sbatch scripts/run-mpi-tests.sh      # this script
#
# The tests adapt to whatever rank count they are given, so to change it edit
# `--rsc p=N` above; NRANKS below is derived from the allocation.
#
# The build must already be configured with H5CPP_ENABLE_PARALLEL=ON against a
# parallel h5c. Configure and compile on the login node; this script only
# builds incrementally and runs.
#
# NOTE: stdout/ and stderr/ must exist BEFORE submission. Slurm opens the files
# named by -o/-e before the script runs, so the mkdir below is too late for
# them; the job would fail in under a second with no output at all. Both
# directories are kept in the repository with a .gitkeep for this reason.

set -euo pipefail
set -x

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

# Tolerate submission from either the repository root or scripts/.
if [[ ! -f CMakeLists.txt && -f ../CMakeLists.txt ]]; then
    cd ..
fi
H5CPP_ROOT="$(pwd)"

mkdir -p stdout stderr

ln -sf "./stdout/${SLURM_JOB_NAME}.${SLURM_JOB_ID}" ./out
ln -sf "./stderr/${SLURM_JOB_NAME}.${SLURM_JOB_ID}" ./err

# if $SHELL is not bash and loading modules is necessary, uncomment
. /usr/share/Modules/init/bash
export LD_LIBRARY_PATH=${HOME}/.local/opt/intel/phdf5-2.1.0/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1

BUILD_DIR="${BUILD_DIR:-${H5CPP_ROOT}/build/my-intel-mpi}"
NRANKS="${NRANKS:-${SLURM_NTASKS:-2}}"

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    set +x
    echo "error: ${BUILD_DIR} is not configured." >&2
    echo "Configure it first, on the login node:" >&2
    echo "  cmake --preset my-intel-mpi" >&2
    echo "(or set BUILD_DIR to whichever parallel build you want to run)" >&2
    exit 1
fi

cmake --build "${BUILD_DIR}" -j 4

# srun rather than ctest: ctest would launch with the rank count baked in at
# configure time, while srun inherits this job's allocation.
shopt -s nullglob
binaries=()
for f in "${BUILD_DIR}"/test/test_p*; do
    [[ -x "$f" && ! -d "$f" ]] && binaries+=("$f")
done
shopt -u nullglob

if [[ ${#binaries[@]} -eq 0 ]]; then
    set +x
    echo "error: no test_p* binaries in ${BUILD_DIR}/test." >&2
    echo "Was the build configured with H5CPP_ENABLE_PARALLEL=ON?" >&2
    exit 1
fi

cd "${BUILD_DIR}/test"

failed=0
for exe in "${binaries[@]}"; do
    name="$(basename "${exe}")"
    if srun -n "${NRANKS}" "${exe}"; then
        echo "PASS ${name} (${NRANKS} ranks)"
    else
        echo "FAIL ${name} (${NRANKS} ranks)"
        failed=$((failed + 1))
    fi
done

set +x
echo "=== h5cpp MPI tests: ${#binaries[@]} binaries, ${failed} failed, ${NRANKS} ranks ==="
exit "${failed}"
