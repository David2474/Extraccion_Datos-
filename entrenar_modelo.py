from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

X = np.array([
    [1, 2, 3, 1, 2, 3],
    [2, 3, 1, 2, 3, 1],
    [3, 1, 2, 3, 1, 2],
    [1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3],
    [3, 1, 2, 3, 1, 2],
    [2, 3, 1, 2, 3, 1],
    [1, 1, 1, 1, 2, 1],
    [2, 3, 1, 2, 3, 1],
    [1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3],
    [3, 1, 2, 3, 1, 2],
])
y = ['visual', 'auditivo', 'kinestesico', 'visual', 'auditivo']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
clf.fit(X_train, y_train)


joblib.dump(clf, "modelo_estilo.pkl")
print("Modelo entrenado y guardado como 'modelo_estilo.pkl'")

