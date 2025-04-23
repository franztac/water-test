import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml
import logging

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s - %(filename)s] %(message)s"
)


def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
            logging.info(f"params: n_estimators loaded from {params_path}")
            return params["model_building"]["n_estimators"]

    except Exception as e:
        logging.info(f"error loading parameters from {params_path}: {e}")
        raise


# with open("params.yaml", "r") as f:
#     params = yaml.safe_load(f)
# n_estimators = params["model_building"]["n_estimators"]


def load_data(filepath: str) -> pd.DataFrame:
    try:
        logging.info(f"loading data from {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"error loading data from {filepath}: {e}")
        raise


# train_data = pd.read_csv("./data/processed/train_processed.csv ")

# X_train = train_data.iloc[:, :-1].values
# y_train = train_data.iloc[:, -1].values


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        logging.info("preparing data: separating features and target")
        X = data.drop(columns=["Potability"])
        y = data.Potability
        return X, y
    except Exception as e:
        logging.error(f"error preparing data: {e}")
        raise


# X_train = train_data.drop(columns=["Potability"])
# y_train = train_data.Potability


def train_model(
    X: pd.DataFrame, y: pd.Series, n_estimators: int
) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        logging.info(f"training {clf} model...")
        clf.fit(X, y)
        return clf
    except Exception as e:
        logging.error(f"error training model: {e}")
        raise


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    try:
        with open(filepath, "wb") as file:
            logging.info(f"saving model to {filepath}")
            pickle.dump(model, file)
    except Exception as e:
        logging.error(f"error saving model to {filepath}: {e}")
        raise


# with open("model.pkl", "wb") as f:
#     pickle.dump(clf, f)


def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_processed.csv"
        model_name = "model.pkl"

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        # clf = RandomForestClassifier(n_estimators=n_estimators)
        # clf.fit(X_train, y_train)
        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_name)

    except Exception as e:
        logging.error(f"an error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
