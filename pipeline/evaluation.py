#!/usr/bin/env python3

import sys
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from . import device
from .savingandloading import load_checkpoint
from .model import BERT

def evaluation(model_path, destination_folder, test_loader):
    y_pred = []
    y_true = []
    
    bestmodel = BERT(model_path).to(device)
    load_checkpoint(destination_folder + '/model.pt', bestmodel)

    bestmodel.eval()
    with torch.no_grad():
        for (labels, title, content, titlecontent), _ in test_loader:

            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            titlecontent = titlecontent.type(torch.LongTensor)
            titlecontent = titlecontent.to(device)
            output = bestmodel(titlecontent, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])


