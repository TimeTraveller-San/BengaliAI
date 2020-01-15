python train.py --save_dir "$1" --epochs $2 --run_name seresnext50 --model_name se_resnext50_32x4d --rgb True --activation mish --mixup True --cutmix True --continue_train True
# python train.py --epochs 20 --run_name seresnext50 --model_name efficientnet-b0 --rgb True --activation mish --debug True
