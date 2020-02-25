import os
import ast

import torch
import torch.nn as nn

from tqdm import tqdm

from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain

DEVICE = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAINING_FOLDS_CSV = os.getenv('TRAINING_FOLDS_CSV')
IMG_HEIGHT = int(os.getenv('IMG_HEIGHT'))
IMG_WIDTH = int(os.getenv('IMG_WIDTH'))
EPOCHS = int(os.getenv('EPOCHS'))
TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))
VALIDATION_BATCH_SIZE = int(os.getenv('TEST_BATCH_SIZE'))
MODEL_MEAN = ast.literal_eval(os.getenv('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.getenv('MODEL_STD'))
TRAINING_FOLDS = ast.literal_eval(os.getenv('IMG_HEIGHT'))
VALIDATION_FOLDS = ast.literal_eval(os.getenv('IMG_HEIGHT'))
BASE_MODEL = str(os.getenv('IMG_HEIGHT'))

def calculate_loss(outputs, targets):
    pred_root, pred_vowel, pred_consonant = outputs
    target_root, target_vowel, target_consonant = targets

    return (nn.CrossEntropyLoss()(pred_root, target_root) + 
            nn.CrossEntropyLoss()(pred_vowel, target_vowel) + 
            nn.CrossEntropyLoss()(pred_consonant, target_consonant)) / 3


def train(dataset, data_loader, model, optimizer):
    # Set model to training mode first
    model.train()
    for bi, data in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        image = data['image']
        grapheme_root = data['grapheme_root']
        vowel_diacritic = data['vowel_diacritic']
        consonant_diacritic = data['consonant_diacritic']

        # Load into GPU
        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = calculate_loss(outputs, targets)

        loss.backward()
        optimizer.step()

def evaluate(dataset, data_loader, model):
    model.eval()
    final_loss, counter = 0, 0

    for bi, data in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        counter += 1 
        image = data['image']
        grapheme_root = data['grapheme_root']
        vowel_diacritic = data['vowel_diacritic']
        consonant_diacritic = data['consonant_diacritic']

        # Load into GPU
        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = calculate_loss(outputs, targets)
        final_loss += loss
    return final_loss / counter

def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(folds=TRAINING_FOLDS,
                                        img_height=IMG_HEIGHT,
                                        img_width=IMG_WIDTH,
                                        mean=MODEL_MEAN,
                                        std=MODEL_STD)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=TRAIN_BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=4)

    valid_dataset = BengaliDatasetTrain(folds=VALIDATION_FOLDS,
                                        img_height=IMG_HEIGHT,
                                        img_width=IMG_WIDTH,
                                        mean=MODEL_MEAN,
                                        std=MODEL_STD)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                            batch_size=VALIDATION_BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode='min', 
                                                            patience=5, 
                                                            factor=0.3)

    model = nn.DataParallel(model)

    for _ in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model)
        lr_scheduler.step(val_score)
    torch.save(model.state_dict(), f'{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin')
    

if __name__ == '__main__':
    main()