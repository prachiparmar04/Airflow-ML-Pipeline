from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'max_active_runs': 1,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'Churn-Pipeline',
    default_args=default_args,
    description='A simple DAG to load and transform CSV data',
    schedule_interval=None,
)

# def load_csv(**kwargs):
#     print('LOADING CSV')
#     df = pd.read_csv('/opt/airflow/data/airport_codes.csv')
#     print('reading file done --------->>>>>',len(df))
#     # df_json = df.to_json(orient='split')  # Convert DataFrame to JSON
#     # kwargs['ti'].xcom_push(key='loaded_df', value=df_json)
#     # return df


def transform_data(**kwargs):
   
    # Load the dataset
    data_frame = pd.read_csv('/opt/airflow/data/rawdata/Telco_customer_churn.csv')
    
    # Specify the columns to keep
    columns_to_keep = ['Contract', 'Tenure Months', 'Online Security', 'Tech Support', 'Dependents', 'Total Charges']
    
    # Fill NaN values with an empty string and strip white spaces from 'Total Charges' column
    data_frame = data_frame.fillna('')
    data_frame['Total Charges'] = data_frame['Total Charges'].apply(lambda x: str(x).strip())
    
    # Count empty values per column
    empty_values_count = (data_frame == '').sum()
    
    # Loop through each column and drop rows with empty string values
    for column in columns_to_keep:
        if empty_values_count[column] > 0:
            data_frame = data_frame[data_frame[column] != '']
    
    # Update the count of empty values per column
    empty_values_count = (data_frame == '').sum()
    
    # Select the specified columns for features and target variable
    X = data_frame[columns_to_keep]
    y_labels = data_frame['Churn Value'].values.tolist()
    
    # Convert 'Total Charges' column to float
    X['Total Charges'] = X['Total Charges'].astype(float)
    
    # Initialize a dictionary to store the label mappings
    label_mappings = {}
    
    # Apply LabelEncoder to each categorical column
    categorical_columns = X.select_dtypes(include=['object']).columns
    X_encoded = X.copy()
    
    for column in categorical_columns:
        label_encoder = LabelEncoder()
        X_encoded[column] = label_encoder.fit_transform(X[column])
        label_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    X_encoded = X_encoded.values.tolist()
    encodings = {'X': X_encoded, 'y': y_labels}
    file_path = '/opt/airflow/data/embedding/'
    
    # Saving the encodings dictionary as a pickle file
    with open(file_path+'training_encoding.pickle', 'wb') as file:
        pickle.dump(encodings, file)

    print(f"Encodings saved to ------>>>>> {file_path}")
    # Saving the encodings dictionary as a pickle file
    with open(file_path+'label_mapping.pickle', 'wb') as file:
        pickle.dump(label_mappings, file)
    
    print(f"Label Mappings saved to ------>>>>> {file_path}")


def train_model(**kwargs):

    # Load encoding data
    with open('/opt/airflow/data/embedding/training_encoding.pickle', 'rb') as file:
        encoding = pickle.load(file)

    # Assuming encoding dictionary is defined with 'X' and 'y'
    X = encoding['X']
    y = encoding['y']

    # Perform train-test split (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a list of models to evaluate
    models = [
        LogisticRegression(max_iter=500),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        LGBMClassifier()
    ]

    # Initialize lists to store accuracy and log loss scores
    model_names = []
    train_accuracy_scores = []
    test_accuracy_scores = []
    train_log_losses = []
    test_log_losses = []

    # Loop through each model
    for model in models:
        model_name = type(model).__name__
        print(f"Training and evaluating {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the train data
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)
        
        # Calculate train accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracy_scores.append(train_accuracy)
        
        # Calculate train log loss
        train_logloss = log_loss(y_train, y_train_pred_proba)
        train_log_losses.append(train_logloss)
        
        # Make predictions on the test data
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)
        
        # Calculate test accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracy_scores.append(test_accuracy)
        
        # Calculate test log loss
        test_logloss = log_loss(y_test, y_test_pred_proba)
        test_log_losses.append(test_logloss)
        
        model_names.append(model_name)

    # Create a DataFrame
    results_df = pd.DataFrame({
        'Model': model_names,
        'Train Accuracy': train_accuracy_scores,
        'Test Accuracy': test_accuracy_scores,
        'Train Log Loss': train_log_losses,
        'Test Log Loss': test_log_losses,
        'Error': [train - test for train, test in zip(train_accuracy_scores, test_accuracy_scores)]
    })

    # Display the DataFrame
    print("\nModel Performance Summary:")
    print(results_df)

    # Find the top 2 models with the lowest test log loss
    top_models = results_df.nsmallest(2, 'Test Log Loss')['Model'].tolist()
    print(f"\nTop 2 models with the lowest Test Log Loss: {top_models}")

    # Define hyperparameter grids for the top 2 models
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        },
        'RandomForestClassifier': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        },
        'DecisionTreeClassifier': {
            'max_depth': [None, 10, 20]
        },
        'LGBMClassifier': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }

    # Get current date and time for the filename
    current_datetime = datetime.now().strftime('%Y%m%d')

    # Hyperparameter tuning and saving best models
    for model_name in top_models:
        model = [m for m in models if type(m).__name__ == model_name][0]
        param_grid = param_grids[model_name]
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_log_loss', n_jobs=-1)
        
        # Fit the GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        print(f"\nBest Parameters for {model_name}:")
        print(grid_search.best_params_)
        
        # Evaluate the best model
        y_test_pred_proba = best_model.predict_proba(X_test)
        best_test_logloss = log_loss(y_test, y_test_pred_proba)
        print(f"Best Test Log Loss for {model_name}: {best_test_logloss}")

        # Save the best model with date and time
        filename_with_datetime = f'/opt/airflow/data/models/best_model_{current_datetime}_{model_name}.pkl'
        filename_latest = f'/opt/airflow/data/models/best_model_{model_name}_latest.pkl'
        
        # Save both models
        joblib.dump(best_model, filename_with_datetime)
        joblib.dump(best_model, filename_latest)
        
        print(f"Best model '{model_name}' saved as '{filename_with_datetime}' and '{filename_latest}'")


# load_task = PythonOperator(
#     task_id='load_csv',
#     python_callable=load_csv,
#     provide_context=True,
#     dag=dag,
# )

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)
transform_task >> train_task
