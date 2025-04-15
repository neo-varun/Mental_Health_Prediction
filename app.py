from flask import Flask, render_template, request, jsonify
import torch
import pickle
import pandas as pd
import os
from neural_network import build_and_train_model, NeuralNetwork
from student_data_preprocessing import DataTransformation as StudentDataPreprocessing
from professional_data_preprocessing import DataTransformation as ProfessionalDataPreprocessing

app = Flask(__name__)

@app.route('/')
def index():
    student_model_exists = os.path.exists('artifacts/student_model.pth')
    professional_model_exists = os.path.exists('artifacts/professional_model.pth')
    return render_template('index.html', 
                         student_model_exists=student_model_exists, 
                         professional_model_exists=professional_model_exists)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        user_type = request.form['user_type']
        model_path = f'artifacts/{user_type}_model.pth'
        metrics_path = f'artifacts/{user_type}_metrics.pkl'
        preprocessor_path = f'artifacts/{user_type}_preprocessor.pkl'
        
        force_train = request.form.get('force_train', 'false').lower() == 'true'
        if not force_train and os.path.exists(model_path) and os.path.exists(preprocessor_path):
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
            return jsonify({
                "success": True,
                "message": "Model already exists and ready to use!",
                "metrics": metrics
            })
        
        train_path = f'data/{user_type}_train_data.csv'
        if not os.path.exists(train_path):
            return jsonify({
                "success": False,
                "message": f"{user_type.capitalize()} training data not found"
            }), 404

        data_transformer = StudentDataPreprocessing() if user_type == 'student' else ProfessionalDataPreprocessing()
        os.makedirs('artifacts', exist_ok=True)

        train_arr, test_arr = data_transformer.initialize_data_transformation(train_path, train_path)
        model, metrics = build_and_train_model(train_arr, test_arr, target_column_index=-1)
        
        torch.save(model.state_dict(), model_path)
        metrics['model_type'] = user_type
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        return jsonify({
            "success": True,
            "message": "Model trained and saved successfully!",
            "metrics": metrics
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_type = request.form.get('user_type', 'student')
        input_data = request.form.to_dict(flat=True)
        input_data.pop('user_type', None)

        preprocessor_path = f'artifacts/{user_type}_preprocessor.pkl'
        model_path = f'artifacts/{user_type}_model.pth'
        
        if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
            return jsonify({
                "error": f"Model or preprocessor not found for {user_type}. Please train the model first.",
                "success": False
            }), 404

        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        feature_order = [
            'Age', 'Work/Study Hours', 'Financial Stress', 'Sleep Duration', 
            'Dietary Habits', 'Have you ever had suicidal thoughts ?',
            'Family History of Mental Illness', 'Gender', 'City', 'Degree'
        ]
        
        if user_type == 'student':
            feature_order[1:1] = ['Academic Pressure', 'CGPA', 'Study Satisfaction']
        else:
            feature_order[1:1] = ['Work Pressure', 'Job Satisfaction']

        missing_fields = [field for field in feature_order if not input_data.get(field)]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "success": False
            }), 400

        input_features = pd.DataFrame([{key: input_data.get(key, '') for key in feature_order}])
        numerical_cols = ['Age', 'Work/Study Hours', 'Financial Stress']
        numerical_cols.extend(['Academic Pressure', 'CGPA', 'Study Satisfaction'] if user_type == 'student' 
                            else ['Work Pressure', 'Job Satisfaction'])
        
        for col in numerical_cols:
            if col in input_features:
                try:
                    input_features[col] = input_features[col].astype(float)
                except ValueError:
                    return jsonify({
                        "error": f"Invalid value for {col}. Please enter a valid number.",
                        "success": False
                    }), 400

        processed_input = preprocessor.transform(input_features)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = NeuralNetwork(input_size=processed_input.shape[1])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            input_tensor = torch.FloatTensor(processed_input).to(device)
            prediction = model(input_tensor)
            predicted_class = (prediction >= 0.5).int().cpu().item()

        return jsonify({
            "prediction": predicted_class,
            "success": True
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/get_metrics', methods=['POST'])
def get_metrics():
    try:
        data = request.get_json()
        if not (user_type := data.get('user_type')):
            return jsonify({'success': False, 'message': 'User type is required'})
            
        metrics_path = os.path.join('artifacts', f'{user_type}_metrics.pkl')
        if not os.path.exists(metrics_path):
            return jsonify({'success': False, 'message': 'Model metrics not found. Please train the model first.'})
        
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)