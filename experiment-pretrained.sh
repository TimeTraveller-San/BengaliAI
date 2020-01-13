# In this experiment, I see the effect of pretrained weights. How much difference do they create?

# python train.py -p True --epochs 13 --name pretrained-exp1
# python train.py --epochs 13 --name not-pretrained-exp1


python train.py -p True --epochs 15 --name se_resnext101_32x4d
