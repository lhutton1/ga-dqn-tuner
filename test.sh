CONFIG=`cat <<EOF
{
    "models":[
        {
            "name": "mobilenet_v2",
            "type": "torchvision"
        }
    ]
}
EOF`

python3 benchmark/driver.py -m=tune -c="${CONFIG}"
