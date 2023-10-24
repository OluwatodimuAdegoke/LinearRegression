import numpy
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

train_x = numpy.array([25, 50, 75, 100, 125, 150, 175])

train_y = numpy.array([56, 111, 154, 199, 248, 299, 360])



model = Sequential()


model.add(Dense(9, input_shape = [1], activation="relu"))

model.add(Dense(1, activation="linear"))

model.compile(loss = "mean_squared_error", 
              optimizer = "Adam", metrics = ["mse"])

model.fit(train_x, train_y, epochs = 2000) 

test_x = [30]
print(model.predict(test_x))

tensorflow.keras.models.save_model(model,'model.pbtxt')
converter = tensorflow.lite.TFLiteConverter.from_keras_model(model=model)

model_tflite = converter.convert()
open("linearRegressionModel.tflite","wb").write(model_tflite)