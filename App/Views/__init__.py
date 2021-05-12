from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

import Views.routes.ai_routes