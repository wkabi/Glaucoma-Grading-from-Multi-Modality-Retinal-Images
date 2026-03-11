set -x 

DATA_ROOT="./data/GAMMA_training_data/val_data"

python tools/convert_fundusdataset.py --data-root ${DATA_ROOT}
