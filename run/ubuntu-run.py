
METHOD=tune

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
        "tuner": "xgb-rl"
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
python3 benchmark/driver.py -m="${METHOD}" -c="${CONFIG}"
