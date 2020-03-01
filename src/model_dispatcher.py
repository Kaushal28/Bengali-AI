import models

MODEL_DISPATCHER = {
    'resnet34': models.ResNet34,
    'efficientNet': models.EfficientNetWrapper
}  