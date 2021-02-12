from flask import Flask
import pickle

app = Flask(__name__)
app.config.from_object("config.DevelopmentConfig")
# model_name = './Colab/best_model_DenseNet_SK.pkl'
# model_name = './Colab/best_model_DenseNet_DK.pkl'
# model_name = '/home/mhbrt/Desktop/Wind/Multiscale/Colab/best_model_DenseNet_DD.pkl'
# model = pickle.load(open(model_name, 'rb'))

# if app.config["ENV"] == "production":
#     app.config.from_object("config.ProductionConfig")
# else:
#     app.config.from_object("config.DevelopmentConfig")

# print(f'ENV is set to: {app.config["ENV"]}')

from Multiscale import views
from Multiscale import admin_views

# print('Initialitation...')
# print(app.config['MARKER'])
