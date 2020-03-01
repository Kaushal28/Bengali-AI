import os
import ast
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics

from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain
from tqdm import tqdm
from pytorchtools import EarlyStopping

from matplotlib import pyplot as plt

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")



def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()

    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, total: {final_score}, y {y.shape}')
    
    return final_score


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 + l2 + l3) / 3



def train(dataset, data_loader, model, optimizer):
    model.train()
    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        counter = counter + 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        final_loss += loss

        o1, o2, o3 = outputs
        t1, t2, t3 = targets
        final_outputs.append(torch.cat((o1,o2,o3), dim=1))
        final_targets.append(torch.stack((t1,t2,t3), dim=1))

    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    print("=================Train=================")
    macro_recall_score = macro_recall(final_outputs, final_targets)
    
    return final_loss/counter , macro_recall_score


def evaluate(dataset, data_loader, model):
    with torch.no_grad():
        model.eval()
        final_loss = 0
        counter = 0
        final_outputs = []
        final_targets = []
        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
            counter = counter + 1
            image = d["image"]
            grapheme_root = d["grapheme_root"]
            vowel_diacritic = d["vowel_diacritic"]
            consonant_diacritic = d["consonant_diacritic"]

            image = image.to(DEVICE, dtype=torch.float)
            grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            final_loss += loss

            o1, o2, o3 = outputs
            t1, t2, t3 = targets

            final_outputs.append(torch.cat((o1,o2,o3), dim=1))
            final_targets.append(torch.stack((t1,t2,t3), dim=1))
        
        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        print("=================Validation=================")
        macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss/counter , macro_recall_score


def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height = IMG_HEIGHT,
        img_width = IMG_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size= TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        img_height = IMG_HEIGHT,
        img_width = IMG_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size= TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode="min", 
                                                            patience=5, 
                                                            factor=0.3,verbose=True)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    best_score = -1

    print("FOLD : ", VALIDATION_FOLDS[0] )
    
    val_scores = []
    for epoch in range(1, EPOCHS+1):
        train_loss, train_score = train(train_dataset,train_loader, model, optimizer)
        val_loss, val_score = evaluate(valid_dataset, valid_loader, model)

        scheduler.step(val_loss)


        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), "{}_fold{}.pth".format(BASE_MODEL, VALIDATION_FOLDS[0]))

        epoch_len = len(str(EPOCHS))
        print_msg = (f'[{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'train_score: {train_score:.5f} ' +
                     f'valid_loss: {val_loss:.5f} ' +
                     f'valid_score: {val_score:.5f}'
                    )
        val_scores.append(val_score)
        print(print_msg)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, len(val_scores)), val_scores, label='val_scores')

    plt.title('Recall Score')
    plt.xlabel('# of epochs')
    plt.ylabel('Recall Score')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()