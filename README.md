# Imagine classification CNNs for BengaliAI
This is a public record of my failure in the [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19/) kaggle competition. I read and implemented various papers yet couldn't get in the silver zone so I never submitted my solution. 


Might also serve as a repo for the next image classification competition, won't lose again.


# Disclaimer
You will be a fool if you copy anything from this repo. I probably implemented, whatever you're copying, wrong.

An incomplete list of things I failed with:

## Augmentations
- Albumentation .. whatever seemed logical
- Mixup
- Cutmix
- Gridmasks
- Some custom morphological augmentations

## Model
- Transfer learning is the king
- Effnets
- Resnets
- Resnexts
- Custom first convs (variable strides and kernel size)
- Single headed/thee headed models
- Swish amd mish 
- GMPooling
- All other poolings

## Others
- ohem
- gradient accumulation to counter small batch size (x_x limited GPU memory is a pain) (didn't work)
- Optims: SWATS, SGD+M, adam, wadam
- Scheds: OneCycleLR, StepLR, MultistepLR (time waste AF), ReduceLROnplat(Works the best)
- wandb.ai <3
- logs are <3
- I am lazy and never set default types for args... instead kept type casting them
- Nvidia apex <3
- Half precision training <3
- Colab <3
- MultilabelStratifiedShuffleSplit MultilabelStratifiedShuffleSplit MultilabelStratifiedShuffleSplit MultilabelStratifiedShuffleSplit 

# Unit tests
- I've written unit tests for:
    1. `model.py`: Forward pass (to see if all my dimensions are correct).
    2. `model.py`: Check of convergence for a very small toy set (also test some hyperparameters)
    3. `augmentations.py`: Data augmentations
    4. `loaddata.py`: Data loaders, shape of data, time etc

Gambatte!!
<img src="https://i.imgur.com/CUCieJl.jpg" width="400"/>