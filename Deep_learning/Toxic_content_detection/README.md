## Toxic Comment Classification with BERT

This project detects multiple types of toxic comments using **BERT** for **multi-label classification**. It predicts whether a comment belongs to one or more of the following categories:
toxic, severe toxic, obscene, threat, insult, identity hate.  

### Features
- **Model:** Based on pretrained [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny), fine-tuned for toxic comment detection.  
- **Multi-label Output:** Each comment can belong to several categories at once (sigmoid activation).  
- **Evaluation:** Uses multilabel confusion matrix and class-wise F1 optimization.  
- **Threshold Optimization:** Finds the best decision thresholds per label to improve F1 score.  
- **Pretrained Model:** Saved as `toxic_model.h5`.  

### Dataset 📁
The dataset is a subset of the [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). It is loaded from `toxic_subset.csv` and contains two parts:  
- `comment_text` → input (text)  
- six binary columns → labels  

### Technologies
`Python 3.9+`, `TensorFlow`, `Keras`, `Hugging Face Transformers`, `NumPy`, `Pandas`, `Scikit-learn`  

## ⚡ Quick Start
1. Add the dataset file `toxic_subset.csv` to the project folder.  
2. Add pretrained model `toxic_model.h5` (or train your own by uncommenting the training section in `main.py`).  
3. Install dependencies: ```pip install -r requirements.txt```
4. Run: `python main.py` to to evaluate the model, generate predictions, confusion matrices, and optimized thresholds.
5. Run: `python plots.py` to create training accuracy and confusion matrix plots.


