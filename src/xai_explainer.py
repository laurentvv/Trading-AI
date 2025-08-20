import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import shap

logger = logging.getLogger(__name__)

def explain_model_prediction(model, scaler, X: pd.DataFrame, feature_names: list, instance_index: int = -1) -> dict:
    """
    Explains a model's prediction for a specific instance using SHAP.

    Args:
        model: The trained scikit-learn model.
        scaler: The fitted StandardScaler used on the training data.
        X (pd.DataFrame): The dataset containing the features (already scaled if needed).
        feature_names (list): List of feature names corresponding to the model's input.
        instance_index (int): The index of the instance in X to explain. Defaults to the last row (-1).

    Returns:
        dict: A dictionary containing:
            - 'shap_values': The SHAP values for the instance.
            - 'expected_value': The model's expected value (baseline).
            - 'instance_prediction': The model's prediction for the instance.
            - 'instance_probability': The model's prediction probability for the instance (if available).
            - 'feature_names': The list of feature names.
            - 'instance_data': The feature values for the explained instance.
    """
    try:
        # Select the instance to explain
        instance = X.iloc[[instance_index]]  # Double brackets to keep it as a DataFrame

        # Get model prediction and probability for the instance
        instance_prediction = model.predict(instance)[0]
        instance_probability = None
        if hasattr(model, "predict_proba"):
            instance_probability = model.predict_proba(instance)[0]

        # Create a SHAP explainer
        # For tree-based models like RandomForest or GradientBoosting
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(instance)
            # For binary classification, shap_values is a list [class_0, class_1]. We take class 1.
            if isinstance(shap_values, list):
                shap_values = shap_values[1] 
            expected_value = explainer.expected_value
            if isinstance(expected_value, list): # Handle list for binary classification
                expected_value = expected_value[1] 
        # For linear models like LogisticRegression, use LinearExplainer with a proper masker
        elif isinstance(model, LogisticRegression):
            # Use maskers.Independent for 'interventional' style perturbation.
            # We pass the background data (X) to the masker.
            masker = shap.maskers.Independent(data=X)
            explainer = shap.LinearExplainer(model, masker=masker)
            shap_values = explainer.shap_values(instance)
            expected_value = explainer.expected_value
        else:
            logger.warning(f"SHAP explanation not implemented for model type: {type(model)}. Returning None.")
            return None

        # Flatten shap_values if it's 2D (e.g., (1, n_features))
        if shap_values.ndim == 2:
            shap_values = shap_values.flatten()

        explanation_result = {
            'shap_values': shap_values,
            'expected_value': expected_value,
            'instance_prediction': instance_prediction,
            'instance_probability': instance_probability,
            'feature_names': feature_names,
            'instance_data': instance.iloc[0].values # Get the row as a 1D array
        }

        logger.info("SHAP explanation generated successfully.")
        return explanation_result

    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
        return None

def plot_shap_waterfall(explanation: dict, max_display: int = 10, save_path: str = None):
    """
    Plots a waterfall chart of SHAP values for an explanation.

    Args:
        explanation (dict): The dictionary returned by `explain_model_prediction`.
        max_display (int): Maximum number of features to display.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    if not explanation:
        logger.warning("No explanation data provided for plotting.")
        return

    try:
        shap_values = explanation['shap_values']
        expected_value = explanation['expected_value']
        feature_names = explanation['feature_names']
        instance_data = explanation['instance_data']
        instance_prediction = explanation['instance_prediction']
        instance_probability = explanation['instance_probability']

        # Create a SHAP Explanation object for plotting
        # We need to reshape data for the Explanation object
        shap_values_reshaped = np.reshape(shap_values, (1, len(shap_values)))
        instance_data_reshaped = np.reshape(instance_data, (1, len(instance_data)))
        
        shap_explanation = shap.Explanation(
            values=shap_values_reshaped,
            base_values=expected_value,
            data=instance_data_reshaped,
            feature_names=feature_names
        )

        # Plot waterfall
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation[0], max_display=max_display, show=False)
        plt.title(f"SHAP Waterfall Plot for Prediction: {instance_prediction} (Prob: {instance_probability.max() if instance_probability is not None else 'N/A':.3f})")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"SHAP waterfall plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting SHAP waterfall: {e}")

def plot_shap_summary(explanations: list, max_display: int = 10, save_path: str = None):
    """
    Plots a summary plot of SHAP values from multiple explanations.

    Args:
        explanations (list): A list of dictionaries returned by `explain_model_prediction`.
        max_display (int): Maximum number of features to display.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    if not explanations:
        logger.warning("No explanation data provided for summary plotting.")
        return

    try:
        # Aggregate data from all explanations
        all_shap_values = []
        all_instance_data = []
        feature_names = explanations[0]['feature_names'] # Assume consistent features

        for exp in explanations:
            all_shap_values.append(exp['shap_values'])
            all_instance_data.append(exp['instance_data'])

        # Convert to numpy arrays
        all_shap_values = np.array(all_shap_values)
        all_instance_data = np.array(all_instance_data)

        # Create a SHAP Explanation object for summary plotting
        shap_explanation = shap.Explanation(
            values=all_shap_values,
            data=all_instance_data,
            feature_names=feature_names
        )

        # Plot summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_explanation.values, shap_explanation.data, feature_names=shap_explanation.feature_names, max_display=max_display, show=False)
        plt.title("SHAP Summary Plot")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting SHAP summary: {e}")