# python train.py --run_name vanilla --epochs 30 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_1 --verbose True
# python train.py --run_name cutmix --cutmix True --epochs 30 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_1 --verbose True
# python train.py --run_name mixup --mixup True --epochs 30 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_1 --verbose True
# python train.py --alpha 2 --run_name mixup_alpha_2 --mixup True --epochs 30 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_1 --verbose True --min_save_epoch 10000



python train.py --run_name vanilla_no_pretrained --epochs 50 --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name vanilla_pretrained --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name vanilla_pretrained_rgb --rgb True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name vanilla_pretrained_mish --activation mish --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000

python train.py --run_name mixup --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name mixup_2 --alpha 2 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name mixup_3 --alpha 3 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name mixup_5 --alpha 5 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name mixup_7 --alpha 7 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name mixup_10 --alpha 10 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000

python train.py --run_name cutmix --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name cutmix_2 --alpha 2 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name cutmix_3 --alpha 3 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name cutmix_5 --alpha 5 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name cutmix_7 --alpha 7 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
python train.py --run_name cutmix_10 --alpha 10 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
