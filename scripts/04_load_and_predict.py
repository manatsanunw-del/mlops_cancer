import mlflow
from sklearn.datasets import load_breast_cancer
 
 
def load_and_predict():
    """
    Simulates a production scenario by loading a model using an alias
    from the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"  # MLflow 3 ใช้ Alias แทน Stage เดิม (เช่น staging, champion)
 
    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")
 
    # Load the model from the Model Registry ด้วย Alias URI
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI.")
        return
 
    # Prepare new sample data (as_frame=True เพื่อให้ชื่อคอลัมน์ตรงกับ signature ของโมเดล)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    sample_malignant = X[y == 0].iloc[0:1]
    sample_benign = X[y == 1].iloc[0:1]
 
    # Use the loaded model to make a prediction
    # No manual preprocessing is needed because we logged the entire pipeline
    prediction_malignant = model.predict(sample_malignant)
    prediction_benign = model.predict(sample_benign)
 
    label_names = {
    0: "malignant",
    1: "benign",
    }
    actual_malignant = 0
    actual_benign = 1

    predicted_malignant = int(prediction_malignant[0])
    predicted_benign = int(prediction_benign[0])

    print("-" * 60)
    print("Prediction Results:")
    print("-" * 60)

    print(
    f"Actual: malignant | "
    f"Predicted: {label_names[predicted_malignant]} | "
    f"Correct: {predicted_malignant == actual_malignant}"
    )
    print(
    f"Actual: benign | "
    f"Predicted: {label_names[predicted_benign]} | "
    f"Correct: {predicted_benign == actual_benign}"
)
    print("-" * 60)
 
if __name__ == "__main__":
    load_and_predict()

