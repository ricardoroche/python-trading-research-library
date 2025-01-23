import json
import logging
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import FEATURES_DATA_DIR

logger = logging.getLogger(__name__)


def train_lasso_pipeline(
    df,
    target_column,
    alpha=0.1,
    test_size=0.1,
    save_model_path=None,
    save_metrics_path=None,
):
    """
    Train a Lasso regression model using an sklearn pipeline with normalization.

    Args:
        df (pd.DataFrame): Input DataFrame with features and target.
        target_column (str): Name of the column to predict.
        alpha (float): Regularization strength for Lasso.
        test_size (float): Proportion of data to use for testing.
        save_model_path (Path): Optional path to save the pipeline.
        save_metrics_path (Path): Optional path to save evaluation metrics.

    Returns:
        tuple: Trained pipeline, DataFrame with predictions, and evaluation metrics.
    """
    # Define feature columns
    numerical_features = [col for col in df if col[:5] == "feat_"] + ["sided_quantity"]
    categorical_features = ["action"]
    boolean_features = []
    other_features = ["time_sin", "time_cos"]

    target_column = "vwap_change"

    # Separate features and target
    X = df[
        numerical_features + categorical_features + boolean_features + other_features
    ]
    y = df[target_column]

    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),  # Scale numerical features
            (
                "cat",
                OneHotEncoder(),
                categorical_features,
            ),  # One-hot encode categorical features
        ],
        remainder="passthrough",
    )

    # Fit and transform the feature matrix
    X_preprocessed = preprocessor.fit_transform(X)

    # Get the feature names after preprocessing
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )
    feature_names = (
        numerical_features
        + other_features
        + list(cat_feature_names)
        + boolean_features
    )

    # Convert the preprocessed features to a DataFrame for visualization
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed_df,
        y,
        test_size=test_size,
        shuffle=False,
    )

    # Create the pipeline
    pipeline = Pipeline(
        [
            ("lasso", Lasso(alpha=alpha)),  # Lasso regression
        ]
    )

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Predict on test set
    predictions = pipeline.predict(X_test)

    # Evaluate the model
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, predictions),
        "mean_absolute_error": mean_absolute_error(y_test, predictions),
        "r2_score": r2_score(y_test, predictions),
    }
    logger.info(f"Evaluation Metrics:\n{json.dumps(metrics, indent=4)}")

    # Save pipeline
    if save_model_path:
        import joblib

        joblib.dump(pipeline, save_model_path)
        logger.info(f"Pipeline saved to {save_model_path}")

    # Save metrics
    if save_metrics_path:
        with open(save_metrics_path, "w") as f:
            json.dump(metrics, f)
        logger.info(f"Evaluation metrics saved to {save_metrics_path}")

    # Prepare predictions DataFrame
    predictions_df = pd.DataFrame(
        {"true": y_test, "predicted": predictions}
    ).reset_index(drop=True)

    return pipeline, predictions_df, metrics


def run_model_training():
    """
    Loads the features and target data, trains a Lasso regression model,
    evaluates the model, and saves predictions and model metrics.
    """
    logger.info("Starting Lasso model training...")
    features_file = FEATURES_DATA_DIR / "features_with_target.csv"
    data = pd.read_csv(features_file)

    # Train the model
    model_path = FEATURES_DATA_DIR / "lasso_model.json"
    metrics_path = FEATURES_DATA_DIR / "lasso_metrics.json"
    trained_model, predictions, metrics = train_lasso_pipeline(
        data,
        target_column="vwap_change",
        alpha=0.05,
        save_model_path=model_path,
        save_metrics_path=metrics_path,
    )

    # Save predictions
    predictions_output_path = FEATURES_DATA_DIR / "lasso_predictions.csv"
    predictions.to_csv(predictions_output_path, index=False)
    logger.info(
        f"Model training completed. Predictions saved to {predictions_output_path}."
    )
