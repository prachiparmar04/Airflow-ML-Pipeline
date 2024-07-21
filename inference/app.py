from flask import Flask, abort,request,jsonify
from flask_cors import CORS
from flask_request_id_header.middleware import RequestID
import ast
from .inference import model_inference

app = Flask(__name__)
app.config['REQUEST_ID_UNIQUE_VALUE_PREFIX'] = 'FOO-'
RequestID(app)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return "<h1>Start...</h1>"

@app.route('/v1/model/churn', methods=['POST'])
def churn_model():
    error_msg = None
    try:
        # Extract and parse input data
        record = request.form.get('inputs')
        if not record:
            raise ValueError("Input parameter 'inputs' is missing or incorrect")

        record = record.strip()
        entries = ast.literal_eval(record)

        # Perform model inference
        results = model_inference(entries)
        return jsonify(version='1.0', status=200, output=results)

    except ValueError as e:
        error_msg = str(e)
        status_code = 400  # Bad Request
    except SyntaxError:
        error_msg = "Invalid input format. Ensure 'inputs' parameter is a valid list."
        status_code = 400  # Bad Request
    except Exception as e:
        error_msg = "Unexpected error occurred: " + str(e)
        status_code = 500  # Internal Server Error

    return jsonify(version='1.0', status=status_code, error=error_msg), status_code

# Handles 404 page not found error page
@app.errorhandler(404)
def page_not_found(e):
    return jsonify(status=404, result='API NOT Found'),404

# handles 400 bad request error page.
@app.errorhandler(400)
def bad_request(e):
    return jsonify(status=400, result=e.description),400


if __name__ == '__main__':
    app.run(port=8080)
