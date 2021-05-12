from flask import request
from Views import app
from Controllers.ai_controllers import ai_controller


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'forgot to set key to "image"?'

    img = request.files['image']  # behöver konvertera den till image också

    response = ai_controller.predict_image(img)

    return response
