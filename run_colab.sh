# python train.py --save_dir "$1" --epochs $2 --run_name seresnext50 --model_name se_resnext50_32x4d --rgb True --activation mish --mixup True --cutmix True --use_wandb True --continue_train True
python train.py --save_dir "$1" --epochs $2 --run_name effnetb3Test --model_name efficientnet-b3 --rgb True --activation mish --mixup True --cutmix True --use_wandb True --continue_train True
