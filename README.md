# Sentiment Analysis on Movie Reviews

This project performs sentiment analysis on the Large Movie Review Dataset v1.0 using various models implemented in Python with the PyTorch framework. It includes a command-line interface (CLI) for testing custom text input sentiment and supports training and evaluating different models.

## Features

•  [**Models**] Linear, Multi-layer Linear, RNN, and LSTM models.

•  [**Vectorization**] RNN and LSTM models use the Gensim GloVe vectorizer.

•  [**Test custom text input**] Test custom text input sentiment with a CLI interface.

•  [**Training and Evaluation**] Train and evaluate models on the Large Movie Review Dataset v1.0.


## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/sentiment-analysis-project.git
cd sentiment-analysis-project

2. Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:

pip install -r requirements.txt

if install fails install manually the following packages for windows:
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- pip install nltk==3.9.1
- pip install pandas==2.2.3
- pip install numpy==2.1.1
- pip install gensim==4.3.3
- pip install scikit-learn==1.5.2
- pip install matplotlib==3.9.2

4. Start CLI interface:
python menu.py
```

2. Using the CLI interface:
```
Start CLI interface by calling "python menu.py"

Please choose a model:
    "1" "Simple perceptron"
    "2" "Multi linear"
    "3" "RNN"
    "4" "LSTM"
    "5" "Quit"

Please choose an option:
    "1" "Evaluate this model"
    "2" "Predict"
    "3" "Train"
    "4" "Back to the Models"
    "5" "Quit"

choosing "1" > Evaluate this model:
Please choose a dataset:
    "1" "IMDB"
    "2" "AMAZON"
    "3" "YELP"
    "4" "Quit"

choosing "2" > Predict:
Write a review to get a sentiment prediction from the model:
An overhyped film that doesn’t live up to its expectations, plagued by a weak script and clichéd characters. It’s a frustrating watch that ultimately feels like a waste of time.

outputs tensor(0.0569, device='cuda:0')
 negative

You can try more from the review_samples.txt


choosing "3" > Train:
(The chosen model type train process starts.)
```

**Requirements:**

•  Python 3.7+

•  PyTorch

•  Gensim

•  NLTK

•  Other dependencies listed in requirements.txt

**License**

Please see the LICENCE file for more information.

**Acknowledgements**

•  *Dataset used for training and testing:* is the "The Large Movie Review Dataset v1.0" by Andrew L. Maas et al(2011). Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics and the 11th Meeting of the European Chapter of the Association for Computational Linguistics. https://ai.stanford.edu/~amaas/data/sentiment/

•  *The GloVe vectors:* by the Stanford NLP Group. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 1532-1543).

•  *Transfer datasets:* for AMAZON, and YELP: Kotzias, D., et al. (2015). From Group to Individual Labels using Deep Features. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2015).

