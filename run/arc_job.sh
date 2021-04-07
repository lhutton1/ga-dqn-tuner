#$ -V -cwd
#$ -l h_rt=00:10:00
#$ -l coproc_p100=4
#$ -m be

# Load modules
module add singularity/3.6.4
module add cuda/10.1.168

# Run program via Singularity
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H%M")
DATE_TIME=${DATE}_${TIME}

# Change to required username
NO_BACKUP_DIR="/nobackup/sc17ljth"

PROJECT_DIR="${NO_BACKUP_DIR}/rl-tuner"
TVM_IMG="${PROJECT_DIR}/rl-tuner.simg"
TMP_DIR="${NO_BACKUP_DIR}/tmp"

SCRIPT_TO_RUN="${PROJECT_DIR}/driver.py"
METHOD="tune"
CONFIG=`cat <<-EOF
{
    "models": [
        {
            "name": "mobilenet_v2",
            "type": "torchvision"
        }
    ],
    "autotuner_settings": {
        "early_stopping": 250,
        "number": 10,
        "tuning_records": "autotuner_records.json",
        "repeat": 1,
        "trials": 1000,
        "tuner": "xgb"
    },
    "run_settings": {
        "profile": true,
        "device": "gpu",
        "fill_mode": "random",
        "repeat": 4
    },
    "target": "cuda"
}
EOF`

echo "Starting script..."
echo "Swapping to ${PROJECT_DIR}"
cd ${PROJECT_DIR}
singularity exec --nv --containall --bind ${PROJECT_DIR}:/rl-tuner --bind ${TMP_DIR}:/tmp --pwd /rl-tuner -H ${PROJECT_DIR} ${TVM_IMG} python3 ${SCRIPT_TO_RUN} -m=${METHOD} -c="${CONFIG}"
echo "Finished script."
