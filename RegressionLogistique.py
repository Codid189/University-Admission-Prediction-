import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(linear)
            
            dw = (1 / n_samples) * np.dot(X.T, (h - y))
            db = (1 / n_samples) * np.sum(h - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ---------------------------------------------------
# 1. Préparation des données
# ---------------------------------------------------
data = pd.read_csv('datadash.csv')
data['Admitted'] = data['Chance_of_Admit'].apply(lambda x: 1 if x >= 0.5 else 0)

X = data[['GRE_Score', 'LOR', 'CGPA']].values
y = data['Admitted'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# 2. Entraînement des modèles
# ---------------------------------------------------
# Modèle manuel
manual_model = LogisticRegressionManual(learning_rate=0.1, iterations=3000)
manual_model.fit(X_train_scaled, y_train)

# Modèle scikit-learn
sklearn_model = LogisticRegression(max_iter=3000)
sklearn_model.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# 3. Évaluation des performances
# ---------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\nPerformance du modèle {name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Matrice de confusion:")
    print(confusion_matrix(y_test, y_pred))

evaluate_model("manuel", manual_model, X_test_scaled, y_test)
evaluate_model("scikit-learn", sklearn_model, X_test_scaled, y_test)

# ---------------------------------------------------
# 4. Interface de prédiction interactive
# ---------------------------------------------------
def predict_interactive():
    print("\nPrédiction interactive (tapez 'q' pour quitter)")
    while True:
        try:
            # Saisie utilisateur
            gre = input("\nGRE Score (260-340): ")
            if gre.lower() == 'q': break
            
            lor = input("LOR (1.0-5.0): ")
            cgpa = input("CGPA (6.0-10.0): ")
            
            # Préparation des données
            raw_data = np.array([[float(gre), float(lor), float(cgpa)]])
            scaled_data = scaler.transform(raw_data)
            
            # Prédictions
            manual_proba = manual_model.predict_proba(scaled_data)[0]
            manual_pred = manual_model.predict(scaled_data)[0]
            
            sklearn_proba = sklearn_model.predict_proba(scaled_data)[0][1]
            sklearn_pred = sklearn_model.predict(scaled_data)[0]
            
            # Affichage comparé
            print("\n" + "="*55)
            print(f"Modèle Manuel → Probabilité: {manual_proba:.1%} | Décision: {'ADMIS' if manual_pred else 'NON ADMIS'}")
            print(f"Scikit-learn → Probabilité: {sklearn_proba:.1%} | Décision: {'ADMIS' if sklearn_pred else 'NON ADMIS'}")
            print("="*55)
            
        except Exception as e:
            print(f"Erreur: {str(e)} - Réessayez")

predict_interactive()