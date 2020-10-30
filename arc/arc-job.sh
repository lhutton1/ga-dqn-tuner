#$ -l h_rt=01:00:00
#$ -l coproc_v100=1

# Load modules
module add singularity/2.4
module add cuda/9.0.176

# Run program via Singularity
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H%M")
DATE_TIME=${DATE}_${TIME}

NO_BACKUP_DIR="/nobackup/sc17ljth"

PROJECT_DIR="${NO_BACKUP_DIR}/benchmark-tvm"
TVM_IMG="${NO_BACKUP_DIR}/benchmark-tvm.simg"

SCRIPT_TO_RUN="${PROJECT_DIR}/tune_and_benchmark.py"
SCRIPT_ARGS=""

echo "Starting script..."
echo "Swapping to ${PROJECT_DIR}"
cd ${PROJECT_DIR}
singularity exec --nv -H ${NO_BACKUP_DIR} ${TVM_IMG} python ${SCRIPT_TO_RUN} ${SCRIPT_ARGS}
echo "Finished script."