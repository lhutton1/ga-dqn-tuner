#$ -V -cwd
#$ -l h_rt=00:01:00
#$ -l coproc_v100=2
#$ -m be

# Load modules
module add singularity/3.6.4
module add cuda/10.1.168

# Run program via Singularity
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H%M")
DATE_TIME=${DATE}_${TIME}

NO_BACKUP_DIR="/nobackup/sc17ljth"

PROJECT_DIR="${NO_BACKUP_DIR}/benchmark-tvm"
TVM_IMG="${PROJECT_DIR}/benchmark-tvm.simg"

SCRIPT_TO_RUN="${PROJECT_DIR}/benchmark/tune_and_benchmark.py"
SCRIPT_ARGS=""

echo "Starting script..."
echo "Swapping to ${PROJECT_DIR}"
cd ${PROJECT_DIR}
singularity exec --nv -H ${NO_BACKUP_DIR} ${TVM_IMG} python3 ${SCRIPT_TO_RUN} ${SCRIPT_ARGS}
echo "Finished script."
