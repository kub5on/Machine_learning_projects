import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from typing import Any, Union
from dataclasses import dataclass

@dataclass
class Metric:
    model: Union[SGDClassifier, svm.SVC, DecisionTreeClassifier]
    accuracy: Any
    f1: Any
    mean_abs_error: Any
    r2: Any

dataset = pd.read_csv('titanic_train.csv')
dataset_x = dataset.drop(columns="Survived")
dataset_y = dataset['Survived']

dataset_x = dataset_x.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

most_frequent = dataset_x['Embarked'].mode()[0]
dataset_x['Age'] = dataset_x['Age'].fillna(dataset_x['Age'].mean())
dataset_x['Embarked'] = dataset_x['Embarked'].fillna(most_frequent)

label_encoder = LabelEncoder()
dataset_x['Sex'] = label_encoder.fit_transform(dataset_x['Sex'])
dataset_x['Embarked'] = label_encoder.fit_transform(dataset_x['Embarked'])

train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, train_size=0.8, random_state=42)

scaler = StandardScaler()
train_x = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns, index=train_x.index)
test_x = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns, index=test_x.index)

model_sgd = SGDClassifier(loss='hinge', penalty='l2')
model_sgd.fit(train_x, train_y)
y_pred_sgd = model_sgd.predict(test_x)
# print(y_pred_sgd)

model_svc = svm.SVC()
model_svc.fit(train_x, train_y)
y_pred_svc = model_svc.predict(test_x)
# print(y_pred_svc)

model_tree = DecisionTreeClassifier()
model_tree.fit(train_x, train_y)
y_pred_tree = model_tree.predict(test_x)
# print(y_pred_tree)


metrics = {key: None for key in ["SGDClassifier", "SVC", "DecisionTreeClassifier"]}

# Tworzenie obiektów Metric
metrics["SGDClassifier"] = Metric(
    model=model_sgd,
    accuracy=accuracy_score(test_y, y_pred_sgd),
    f1=f1_score(test_y, y_pred_sgd),
    mean_abs_error=mean_absolute_error(test_y, y_pred_sgd),
    r2=r2_score(test_y, y_pred_sgd)
)

metrics["SVC"] = Metric(
    model=model_svc,
    accuracy=accuracy_score(test_y, y_pred_svc),
    f1=f1_score(test_y, y_pred_svc),
    mean_abs_error=mean_absolute_error(test_y, y_pred_svc),
    r2=r2_score(test_y, y_pred_svc)
)

metrics["DecisionTreeClassifier"] = Metric(
    model=model_tree,
    accuracy=accuracy_score(test_y, y_pred_tree),
    f1=f1_score(test_y, y_pred_tree),
    mean_abs_error=mean_absolute_error(test_y, y_pred_tree),
    r2=r2_score(test_y, y_pred_tree)
)

# for name, metric in metrics.items():
#     print(f"{name}: accuracy={metric.accuracy:.3f}, f1={metric.f1:.3f}, MAE={metric.mean_abs_error:.3f}, R²={metric.r2:.3f}")
