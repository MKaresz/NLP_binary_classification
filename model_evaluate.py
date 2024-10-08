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


'''
def create_dataloader(dataset: str, paragraph_size: int, vocab_name: str, batch_size: int):
    vocab = file_utils.load_vocabulary(vocab_name)
    vocab_to_idx = text_utils.build_word_to_idx_vocab(vocab)
    
    df_from_raw = pd.read_csv(dataset)
    df_from_raw['sentiment'] = df_from_raw['sentiment'].map({'positive': 1, 'negative': 0})

    cleaned_corpus = text_utils.clean_corpus(corpus=df_from_raw['review'].tolist())
    X_vec = vectorizers.build_word_to_idx_vector(cleaned_corpus, vocab_to_idx, paragraph_size)

    y_vec = df_from_raw['sentiment'].tolist()
    dataset = ReviewLabelDataset(X_vec, y_vec)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader
'''


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
    #device = "cpu"
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
    precision, recall, thresholds = precision_recall_curve(test_labels, y_scores)

    #create precision recall curve
    fig, ax = plt.subplots()
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




############################################################

"""

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:",device)

# Load the dataset
df = pd.read_csv('data\IMDB_100Sample_Dataset.csv')

# Preprocess the data
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

#DEBUG
#DB train
#print(len(X_train)), print(type(X_train))
#print(len(y_train)), print(type(y_train))
#DB test
#print(len(X_test)), print(type(X_test))
#print(len(y_test)), print(type(y_test))

# Convert text to numerical data using CountVectorizer
# applies preprocessing, tokenization and stop words removal
# raw text creates a matrix in the form of (document_id, tokens) in which the values are the token count
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

print("Shape of the generated vector:", X_train_vec.shape)

#DEBUG
#print(type(X_test_vec), type(X_test_vec))

#Step 3: Create a Custom Dataset Class
#Define a custom dataset class to handle the data.
class IMDBDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = torch.tensor(self.reviews[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return review, label

#Step 4: Create DataLoader Instances
#Create DataLoader instances for the training and test sets.
train_dataset = IMDBDataset(X_train_vec, y_train.values)
test_dataset = IMDBDataset(X_test_vec, y_test.values)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Step 5: Define the Perceptron Model
#Define a simple Perceptron model using PyTorch.

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

input_dim = X_train_vec.shape[1]
model = Perceptron(input_dim).to(device)

#Step 6: Define Loss Function and Optimizer
#Set up the loss function and optimizer.
criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01)
# using Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

#Step 7: Train the Model
#Train the Perceptron model.

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for reviews, labels in train_loader:
        # send data to device-agnostic place
        reviews, labels = reviews.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(reviews).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

evaluate_model(model)
"""


# during training:
# import matplotlib.pyplot as plt

# Plotting actual training and validation loss
# https://learn.codesignal.com/preview/lessons/2243/deep-model-evaluation-with-pytorch
# During training, calculate at each epoch the loss for both the training set and test set, 
# use as validation, and store these values in our history dictionary. 
# This helps us monitor the model's performance and ensure it is not overfitting to the training data.

# 1. Train the model
# num_epochs = 150
# history = {'loss': [], 'val_loss': []}
# criterion = nn.CrossEntropyLoss()

# 2. in every epoch at the end:
# model.eval()
# with torch.no_grad():
#     outputs_val = model(X_test)
#     val_loss = criterion(outputs_val, y_test)
#     history['val_loss'].append(val_loss.item())
# print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')

# 3.  Plotting actual training and validation loss
# epochs = range(1, num_epochs + 1)
# train_loss = history['loss']
# val_loss = history['val_loss']

# plt.figure(figsize=(8, 5))
# plt.plot(epochs, train_loss, label='Training Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.title('Model Loss During Training')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
