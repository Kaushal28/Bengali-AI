# [Bengali.AI](https://www.kaggle.com/c/bengaliai-cv19): Different approaches

## Experiment #1 (0.9619 on public LB):

- Used Model: Pretrained EfficientNet-B1
- Image Augmentation technique(s): GridMask
- Other settings:
  - Image size 128 x 128
  - 5-fold Cross Validation
  - 30 epochs for each fold
  - Batch size: 256

TODO:
  - ~~Commit Pretrained Models~~ Link to Kaggle dataset: https://www.kaggle.com/kaushal2896/trainedmodeleffnet
  
## Experiment #2:

- Used Model: Pretrained EfficientNet-B1
- Image Augmentation technique(s): None
- Other settings:
  - Image size 128 x 128
  - 5-fold Cross Validation
  - 40 epochs for each fold
  - Batch size: 256
  
TODO:
  - Commit Pretrained Models

  
References:
  - https://www.youtube.com/watch?v=8J5Q4mEzRtY (Part1 and Part2)
  - https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn
  - GridMask: https://arxiv.org/pdf/2001.04086.pdf, https://www.kaggle.com/haqishen/gridmask
  - AugMix: https://arxiv.org/pdf/1912.02781.pdf, https://www.kaggle.com/haqishen/augmix-based-on-albumentations
