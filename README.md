# Bengali-AI

### A possible approach for Kaggle Competition: https://www.kaggle.com/c/bengaliai-cv19

Used Models:
  - Pretrained EfficientNet-B1
  - Pretrained ResNet-34
  
Image Augmentation techniques: 
  - GridMask: https://arxiv.org/pdf/2001.04086.pdf
  - AugMix (TODO): https://arxiv.org/pdf/1912.02781.pdf
  
 Other settings:
 
  - Image size 128 x 128
  - 5-fold Cross Validation
  - 30 epochs for each fold
  - Batch size: 256
