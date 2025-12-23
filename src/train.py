import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_data, preprocess
mlflow.set_experiment("Project_1_ML_TRAINING")

with mlflow.start_run():
    df = load_data("data/dataset.csv")
    x_train, x_test, y_train, y_test = preprocess(df)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Accuracy: {acc}")