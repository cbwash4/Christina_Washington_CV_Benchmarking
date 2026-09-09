
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


RANDOM_SEED = 42


def get_classical_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf"
            ))
        ]),

        "Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=100,
                early_stopping=True,
                random_state=RANDOM_SEED
            ))
        ])
    }

    return models


def train_classical_models(X_train, y_train):
    models = get_classical_models()

    trained_models = {}
    failed_models = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model

        except Exception as exc:
            failed_models[name] = str(exc)

    return trained_models, failed_models
