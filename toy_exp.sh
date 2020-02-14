# python train.py --run_name vanilla --epochs 30 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_1 --verbose True
# python train.py --run_name cutmix --cutmix True --epochs 30 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_1 --verbose True
# python train.py --run_name mixup --mixup True --epochs 30 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_1 --verbose True
# python train.py --alpha 2 --run_name mixup_alpha_2 --mixup True --epochs 30 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_1 --verbose True --min_save_epoch 10000

# python train.py --run_name vanilla_no_pretrained --epochs 50 --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name vanilla_pretrained --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name vanilla_pretrained_rgb --rgb True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name vanilla_pretrained_mish --activation mish --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
#
# python train.py --run_name mixup --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name mixup_2 --alpha 2 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name mixup_3 --alpha 3 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name mixup_5 --alpha 5 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name mixup_10 --alpha 10 --mixup True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
#
# python train.py --run_name cutmix --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name cutmix_2 --alpha 2 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name cutmix_3 --alpha 3 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name cutmix_5 --alpha 5 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --run_name cutmix_10 --alpha 10 --cutmix True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
#
# poweroff
#
# python train.py --run_name vanilla_head3_mish --activation mish --rgb True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --cutmix True --run_name vanilla_head3_mish_cx --activation mish --rgb True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --mixup True --run_name vanilla_head3_mish_mx --activation mish --rgb True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000
# python train.py --cutmix True --mixup True --run_name vanilla_head3_mish_mx_cx --activation mish --rgb True --epochs 50 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_2 --verbose True --min_save_epoch 10000

# head3 normal was best baseline (bb) 0.714
# python train.py --gridmask True --run_name grdm_bb --activation mish --rgb True --epochs 70 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_4 --verbose True --min_save_epoch 10000
# python train.py --mixup True --gridmask True --run_name grdm+mx_bb --activation mish --rgb True --epochs 70 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_4 --verbose True --min_save_epoch 10000
# python train.py --cutmix True --gridmask True --run_name grdm+cx_bb --activation mish --rgb True --epochs 70 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_4 --verbose True --min_save_epoch 10000
# python train.py --mixup True --alpha 1.2 --run_name mx_1.2 --activation mish --rgb True --epochs 70 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_4 --verbose True --min_save_epoch 10000
# python train.py --mixup True --alpha 1.5 --run_name mx_1.5 --activation mish --rgb True --epochs 70 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_4 --verbose True --min_save_epoch 10000
# python train.py --mixup True --alpha 2 --run_name mx_2 --activation mish --rgb True --epochs 70 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_4 --verbose True --min_save_epoch 10000
# python train.py --mixup True --alpha 3 --run_name mx_3 --activation mish --rgb True --epochs 70 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_4 --verbose True --min_save_epoch 10000
# python train.py --mixup True --alpha 5 --run_name mx_5 --activation mish --rgb True --epochs 70 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_4 --verbose True --min_save_epoch 10000
# poweroff


# python train.py --mixup True --gridmask True --scheduler clr --run_name clr-bb --activation mish --rgb True --epochs 150 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_5 --verbose True --min_save_epoch 10000
# python train.py --mixup True --gridmask True --alpha 2 --scheduler clr --run_name clr-bb_al2 --activation mish --rgb True --epochs 150 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_5 --verbose True --min_save_epoch 10000
# python train.py --mixup True --gridmask True --run_name bb_al2 --activation mish --rgb True --epochs 150 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_5 --verbose True --min_save_epoch 10000
# poweroff


# python train.py --mixup True --gridmask True --run_name bb_sgd --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_5 --verbose True --min_save_epoch 10000
# python train.py --run_name sgd-baseline --optmzr sgd --mixup True --gridmask True --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_6 --verbose True --min_save_epoch 10000
# python train.py --run_name sgd-baseline-moment --momentum 0.9 --optmzr sgd --mixup True --gridmask True --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_6 --verbose True --min_save_epoch 10000
# python train.py --run_name sgd-baseline-clr --optmzr sgd --scheduler clr --mixup True --gridmask True --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_6 --verbose True --min_save_epoch 10000
# python train.py --run_name baseline --optmzr adam --mixup True --gridmask True --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_6 --verbose True --min_save_epoch 10000
# python train.py --run_name sgd-baseline-moment-clr --momentum 0.9 --optmzr sgd --scheduler clr --mixup True --gridmask True --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_6 --verbose True --min_save_epoch 10000

# https://arxiv.org/pdf/1710.03740.pdf

# python train.py --run_name baseline-more-multistep-alt4 --optmzr adam --mixup True --gridmask True --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_6 --verbose True --min_save_epoch 10000
# python train.py --run_name baseline-more-multistep-adamw --optmzr adam --mixup True --gridmask True --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_6 --verbose True --min_save_epoch 10000
# python train.py --batch_size 64 --use_apex True --run_name _apex-test --optmzr adam --mixup True --gridmask True --activation mish --rgb True --epochs 200 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_6 --verbose True --min_save_epoch 10000


# python train.py --w1 3 --w2 1 --w3 1 --run_name baseline-311 --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_7 --verbose True --min_save_epoch 10000
# python train.py --w1 4 --w2 1 --w3 1 --run_name baseline-411 --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_7 --verbose True --min_save_epoch 10000
# python train.py --w1 5 --w2 1 --w3 1 --run_name baseline-511 --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_7 --verbose True --min_save_epoch 10000
# python train.py --w1 1 --w2 1 --w3 1 --run_name baseline-111 --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_7 --verbose True --min_save_epoch 10000
# python train.py --w1 2 --w2 1 --w3 1 --run_name baseline --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_7 --verbose True --min_save_epoch 10000

python train.py --run_name baseline_lossCutoff-40 --min_loss_cutoff 40 --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
python train.py --run_name baseline_morph --mixup True --morph True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
python train.py --scheduler oclr --run_name baseline_oclr --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
python train.py --scheduler rlrp --run_name baseline_rlrp --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
python train.py  --grad_acc 2 --run_name baseline_gac2 --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
python train.py  --grad_acc 4 --run_name baseline_gac4 --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
python train.py --run_name baseline_lossCutoff-30 --min_loss_cutoff 40 --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
python train.py --run_name baseline --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
python train.py --run_name baseline_ohem --ohem True --mixup True --gridmask True --activation mish --rgb True --epochs 100 --pretrained True --toy_set True --model_name se_resnext50_32x4d --save_dir test_8 --verbose True --min_save_epoch 10000
