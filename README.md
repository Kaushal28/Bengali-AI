# Bengali-AI

### A possible approach for Kaggle Competition: https://www.kaggle.com/c/bengaliai-cv19

Used Models:
  - Pretrained EfficientNet-B1 (Achieves 0.9619 on public LB)
  - Pretrained ResNet-34
  
Image Augmentation techniques: 
  - GridMask
  - AugMix (TODO)
  
 Other settings:
 
  - Image size 128 x 128
  - 5-fold Cross Validation
  - 30 epochs for each fold
  - Batch size: 256

TODOs:
  - ~~Commit Pretrained Models~~ Link to Kaggle dataset: https://www.kaggle.com/kaushal2896/trainedmodeleffnet
  
References:
  - https://www.youtube.com/watch?v=8J5Q4mEzRtY (Part1 and Part2)
  - https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn
  - GridMask: https://arxiv.org/pdf/2001.04086.pdf, https://www.kaggle.com/haqishen/gridmask
  - AugMix: https://arxiv.org/pdf/1912.02781.pdf, https://www.kaggle.com/haqishen/augmix-based-on-albumentations
