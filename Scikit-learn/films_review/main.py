import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, auc

imdb = pd.read_csv('imdb.csv')

# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")

imdb["review_processed"] = imdb["review"].str.lower()                                           # zamiana wszystkich liter na małe
imdb["review_processed"] = imdb["review_processed"].str.replace(r"[^a-zA-Z\s]", "", regex=True) # pozbycie się wszystkiego co nie jest spacją lub literą
imdb["review_tokens"] = imdb["review_processed"].apply(word_tokenize)                           # poddanie każdej recenzji tokenizacji

"""lematyzacja i usuwanie stopwords"""

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

processed_reviews = []
for review in imdb["review_tokens"]:
    processed_review = []
    for word in review:
        if word not in stop_words and word != 'br':
            lemmatized_word = lemmatizer.lemmatize(word)
            processed_review.append(lemmatized_word)
    processed_reviews.append(processed_review)

# print(processed_reviews[0])

imdb["review_processed"] = [' '.join(review) for review in processed_reviews]
#
# imdb.to_csv('imdb_preprocessed.csv')

# imdb = pd.read_csv('imdb_preprocessed.csv')

'''Eksploracyjna analiza danych'''
positive_reviews = imdb[imdb['sentiment'] == 'positive']['review_processed']
negative_reviews = imdb[imdb['sentiment'] == 'negative']['review_processed']

all_pos_words = ' '.join(positive_reviews).split()
all_neg_words = ' '.join(negative_reviews).split()

pos_word_freq = Counter(all_pos_words)
neg_word_freq = Counter(all_neg_words)

pos_common_words = pos_word_freq.most_common(20)
neg_common_words = neg_word_freq.most_common(20)

"""Bar ploty opisujące częstość występowania słów"""
# def plot_word_frequency(word_freq, title, ax):
#     word, count = zip(*word_freq)
#     ax.barh(word, count, color='gold')
#     ax.set_xlabel("Częstość występowania słów")
#     ax.set_title(title)
#     ax.invert_yaxis()
#
# fig, axes = plt.subplots(1, 2, figsize=(15,6))
#
# plot_word_frequency(pos_common_words, "20 najczęściej występujących słów - recenzje pozytywne", ax=axes[0])
# plot_word_frequency(neg_common_words, "20 najczęściej występujących słów - recenzje negatywne", ax=axes[1])
# plt.savefig('barplots')

imdb["review_length"] = imdb["review_processed"].str.count(r'\w+')

"""Histogramy zliczające długości poszczególnych recenzji"""
# imdb[imdb['sentiment'] == 'positive']['review_length'].hist(alpha=0.5, label="Pozytywne recenzje", color="purple", bins=60, density=True)
# imdb[imdb['sentiment'] == 'negative']['review_length'].hist(alpha=0.5, label="Negatywne recenzje", color="yellow", bins=60, density=True)
# plt.legend()
# plt.title('Rozkład długości recenzji')
# plt.xlabel('Długość recenzji (liczba słów)')
# plt.ylabel('Częstość występowania słów')
# plt.savefig('histogram_review_length')

y = imdb['sentiment'].map({'positive': 1, 'negative': 0})
X = imdb['review_processed']

"""wektoryzacja danych tekstowych"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

"""klasyfikacja recenzji"""
models = {
    "Naive Bayes": MultinomialNB(alpha=1.0, fit_prior=True),
    "SVM": SGDClassifier(loss='hinge', alpha=0.1, max_iter=1000, penalty='l2', learning_rate='optimal', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8)
}

metrics = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}
confusion_matrices = {name: None for name in models}


# def evaluate_model(y_true, y_pred):
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='macro')
#     recall = recall_score(y_true, y_pred, average='macro')
#     f1 = f1_score(y_true, y_pred, average='macro')
#     return accuracy, precision, recall, f1


for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics['Model'].append(model_name)
    metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['Precision'].append(report["1"]["precision"])
    metrics['Recall'].append(report["1"]["recall"])
    metrics['F1'].append(report["1"]["f1-score"])

    confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

results_df = pd.DataFrame(metrics)
# print(results_df)
