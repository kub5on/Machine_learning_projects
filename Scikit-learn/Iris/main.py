import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv('iris.csv')

X = df.drop(columns='species')
y = df['species']

scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

encoder = LabelEncoder()
encoded_iris = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, encoded_iris, test_size=0.2, random_state=0)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


