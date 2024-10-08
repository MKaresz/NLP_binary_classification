import file_utils   
import text_utils     
import vectorizers  
from custom_types import ReviewLabelDataset
from custom_types import ModelParams

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix


def evaluate_model(model: nn.Module, test_loader: DataLoader, sentiment_threshold: float):
    """
    Evaluate the Model after training on the test set.
    
    Args:
        model: the model to evaluate
        test_loader: dataloader for the test data
        device: cuda or cpu
        sentiment_threshold: the threshold for the prediction
    """
    print("Evaluating...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    # disable gradient calculations
    with torch.no_grad():
        test_labels = []
        predicted_labels = []
        y_scores = []
        num_labels = len(test_loader.dataset)
        correct = 0
        for reviews, labels in test_loader:
            # send test data to device-agnostic place
            reviews, labels = reviews.to(device), labels.to(device)
            # Input the test data into the model
            outputs = model(reviews).squeeze()
            y_scores.extend(outputs.tolist())
            test_labels.extend(labels.tolist())
            predicted = (outputs > sentiment_threshold).int()
            predicted_labels.extend(predicted.tolist())
            correct += (predicted == labels).sum().item()

        accuracy = correct / num_labels
        print(f'Accuracy: {accuracy * 100:.2f}%')

        #BUG DEBUG
        _calculate_metrics(test_labels, predicted_labels, y_scores)

def _show_cm(cm, true_labels):
    """ 
    Plot confusion matrix.
    
    Args:
        cm: values in the confusion matrix
        true_labels: 

    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(set(true_labels)))
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    # Labeling the matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def _calculate_metrics(test_labels, predicted_labels, y_scores):
    """
    Evaluates a model on the test input and plots the confusion matrix and 
    the precision-recall curve.

    Args:
        test_labels: the valid labels that should be the output of the model
        predicted_labels: the prediction determined by the threshold
        y_scores: the calculated raw output of the model
    """
    
    cm = confusion_matrix(test_labels, predicted_labels)
    print(cm)
    _show_cm(cm, test_labels)

    tn, fp, fn, tp = cm.ravel()
    print(tn, fp, fn, tp)
    # The precision is the ratio tp / (tp + fp): the ability of the classifier not to label a negative sample as positive.
    precision_actual = tp / (tp + fp)
    print(f'Precision: {precision_actual:.2f}')

    # The recall is the ratio tp / (tp + fn): the ability of the classifier to find all the positive samples.
    recall_actual = tp / (tp + fn)
    print(f'Recall: {recall_actual:.2f}')

    # F1 score is the harmonic mean of the precision and recall
    # F1 score reaches its best value at 1 and worst score at 0
    f1 = f1_score(test_labels, predicted_labels, average='binary')
    print(f'F1-score: {f1:.2f}')

    print(classification_report(test_labels, predicted_labels, labels=[1, 0], target_names=["positive", "negative"]))
    # The support is the number of occurrences of each class in y_true.

    #calculate precision and recall for different thresholds
    precision, recall, _ = precision_recall_curve(test_labels, y_scores)

    #create precision recall curve
    _, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    #display plot
    plt.show()

def plot_training_history( model_params: ModelParams ):
    """Plots the loss on training and validation data during the training."""
    epochs = range(1, len(model_params.train_history['training_loss']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, model_params.train_history['training_loss'], label='Training Loss')
    plt.plot(epochs, model_params.train_history['validation_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
