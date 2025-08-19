import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier

df = pd.read_csv('iris.csv')
X = df.drop(columns='species')
y = df['species']

scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X))

encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, encoded_y, test_size=0.2, random_state=0)

model_gaussian = GaussianProcessClassifier(random_state=0)
model_gaussian.fit(X_train, y_train)
y_pred = model_gaussian.predict(X_test)

gaussian_pred = encoder.inverse_transform(y_pred)


