from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import pickle

model_name = '/home/mhbrt/Desktop/Wind/Multiscale/Colab/DenseNet_150_16_DD_FINALMODEL.h5'
#model = pickle.load(open(model_name, 'rb'))
model = load_model(model_name)
print('_LOAD MODEL DONE_')
print(model.summary())
#model.save("keras_model.h5")

#model_json = model.to_json()
#with open("model.json", "w") as json_file:
    #json_file.write(model_json)
    
#model.save_weights("model.h5")
