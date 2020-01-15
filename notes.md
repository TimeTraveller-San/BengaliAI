# Notes as progress
I scribble here whatever I feel like. There is no order. This is just a direct imprint of my chaotic mind struggling to understand things.

pip install iterative-stratification

# Todo
- increase batch size by gradient accumulation

# Ideas
Steal from here: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94687



# Feels
- DAMN! `se_resnext50_32x4d` with mish and mixup got 0.9 val score only on 2 epochs (~55 minutes)! This is amazing. I will be training it for ~40-60 epochs.

# Model?
- I love efficientnets. Mainly because they are efficient. EUREKA!
- effnet-b0 got me a score of ~0.94 with all the vanilla settings. Althought, I won't submit until I get ~0.97 so I don't know my CB score (however it should be close because its close for everyone in the discussions)
- effnet-b0 is very small. I need a bigger model.
- I should try effnet-b4 but I easily get influenced by people in kaggle. they are getting good score for `se_resnext50_32x4d` so I will train that now too (RIP my GPU)


# Activation
- This guy from India, same age as me, FREAKING INVENTED a new activation function - mish
- idk how he found it. swish used RL search. He didn't tell what he did. Nor does his paper provide a mathematical explanation of why/how mish works. Or what was the motivation behind it. I would call it more of a discovery than invention. Maybe he used some sort of search like people at google did for swish. I WANT TO KNOW!!
- Its math and implementation is pretty simple. idk why it works but it does.
- Because its a composite function of three functions, it's a little slower than both swish and RELU
- I implemented it for my se-resnext. idk about the improvement because I dont have a baseline but for a single epoch, its slower by ~2-3 minutes
- I think no one on kaggle is doing this for now. But mish is pretty famous. fast.ai implemented it. I am sure people on kaggle are using it but no one disclosed it yet. This could be a part of "magic"... maybe...


# Clean code
- I think I've grown past my phase of writing all the code in jupyter notebooks
- However, I still have a lot to learn. I need to structure my code well.
- For now, all my models, augmentations, custom activations, losses etc lie in same folder
- I should make a hierarchy of folders
- Maybe use nvidia ignite? fastai? catalyst? and callbacks for cleaner code
- I think I will start with nvidia ignite. Let me first get a baseline result. I want LB ~0.97 with a single model

# Unit tests
- I read a few blogs by some kagglers like the kaggleGM-legendary-russian-name_starts_from_V author of augmentations library
- I've learned a lot of things but one very important thing is: unit tests
- In jupyter notebooks, unit tests were basically just running the notebook and on getting error, restarting everything and spamming shift+enter
- It was a tedious and ugly process. I felt disgusted.
- However, for separate py-scripts I can write unit tests for any function/module I want in the main
- I've written unit tests for:
    1. `model.py`: Forward pass (to see if all my dimensions are correct).
    2. `model.py`: Check of convergence for a very small toy set (also test some hyperparameters)
    3. `augmentations.py`: Data augmentations
    4. `loaddata.py`: Data loaders, shape of data, time etc


# Overfitting-Regularization
- My effnet-b0 model was overfitting. I need some kind of regularization. Dropout? Cutout? Mixup?
- After reading the paper and going through facebook research's official github, I've realized that mixup achieves SOTA
- I read another paper that applies mixup to the output of conv layers (features). I am not sure if I should apply this after every conv or just first or something else?
- A kaggle kernel has applied mixup after every single block of resnet. I am not sure if its a good idea.
- The whole idea of applying mixup at features was that mixup at input doesn't make much sense because brightness, contrast, pixel values may vary a lot. But for MNIST-type data, I think this should not be a problem
- I have decided to applying mixup on input layer
- input = `lambda * input1 + (1 - lambda) * input2`... we want lambda to be close to 0 or 1 so that we train on not-so-noisy images (so we use beta distribution)
- Unit testing shows that the seresnext-50 model does converge after mixup (slower than without mixup but that was expected). It was ~12.5% slower (not in terms of time but in terms of reduction in loss)
- My Mixup implementation adds ~30 seconds per epoch so its negligible
- Mixup also requires more number of epochs ~60
- One epoch for me takes ~27 minutes
- 60 epochs ~ 30 hours. RIP my GPU.
- I should find some place else to train my models. colab? paperspace?
- Well, my training scores and metrics don't make much sense while while using mixup because the data isn't real data at all, its an interpolation
- So, during validation, I am validating on both train_loader and valid_loader now.... this means MORE TIME PER EPOCH. As if I wasn't doomed enough already.

# Beta distribution
- It's just power function normalized by beta function
- For mixup, we take alpha=beta=something to make the distribution symmetric
- If I set alpha small. like ~0.0001 then it will almost always be 1 or 0
- I set it to 1 (because of the official facebook research code)
- 1 is still a small value. For 1, most of the times the random number will be close to 1 or 0 which means that it will be close to either image 1 or 2. which is a good thing. because we don't want to learn a lot of noise (mix of two images) but just a little noise (eg mostly image 1 with some image 2)
## Dirichlet distribution is pretty cool too.
- It's just a generalization for beta if I want to interpolate more than 2 images.
- still having all the parameters ~1 means having a tuple of random numbers that are close to some single image.
- I don't really understand its math very well but I will learn about all the distributions in a few days. I will explore more of it then. For now, I focus on this competition.


# CUDA out of memory
- I got CUDA out of memory for my se_resnext50_32x4d
- Usually, I dismiss it at saying - my GPU is only 6GB
- This time I investigated

# Datasplit
- MultilabelStratifiedShuffleSplit
- such a cool name, once more
- MultilabelStratifiedShuffleSplit
- its like the name of a super move
- MultilabelStratifiedShuffleSplit
- I will go annoy my friend by messaging him this and not explaining what this is
- MultilabelStratifiedShuffleSplit 
