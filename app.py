#Importing the required libraries
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import torch
import torchaudio
import os

#Creating the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

#Loading the trained model