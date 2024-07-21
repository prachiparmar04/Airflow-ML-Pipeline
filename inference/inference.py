from .embedding import tokenization
import warnings
warnings.filterwarnings("ignore")
import glob
import os
import joblib

def load_latest_model(model_dir):
    # Find the latest model file based on the naming convention
    model_files = glob.glob(os.path.join(model_dir, '*_latest.pkl'))
    if not model_files:
        raise FileNotFoundError("No model files found with '_latest.pkl' in the name")
    
    # Sort files and pick the latest one
    latest_model_file = max(model_files, key=os.path.getctime)
    print(f"Loading model from: {latest_model_file}")
    
    return joblib.load(latest_model_file)

# Load the latest model
model = load_latest_model('/app/models')
def model_inference(features):
    
    """
    Performs inference on the given features to predict churn status.

    Parameters:
    features (list): List of dictionaries, where each dictionary contains the features.

    Returns:
    list: List of churn predictions ("No Churn" or "Churn").
    """
    
    # Step 1: Tokenize the features
    # Convert the input features into a format suitable for model prediction
    print('features --->>>',features)
    embedding = tokenization(features)
    print('embedding --->>>',embedding)
    # Step 2: Predict scores using the pre-trained LightGBM model
    # The model predicts the scores based on the tokenized features
    scores = model.predict(embedding)
    print('scores ----->',scores)
    # Step 3: Define the mapping from prediction scores to human-readable labels
    # Here, 0 is mapped to "No Churn" and 1 is mapped to "Churn"
    mapping = {0: "No Churn", 1: "Churn"}
    ids = [feature['id'] for feature in features]
    # Step 4: Map the prediction scores to human-readable labels
    # Iterate over the prediction scores and map them using the defined mapping
    predictions = [mapping[element] for element in scores]
    print('predictions --->',predictions)
    results = [{'id': ids[x], 'isChurn': predictions[x]} for x in range(len(ids))]
    # Step 5: Return the results
    # The final output is a list of churn predictions ("No Churn" or "Churn")
    return results