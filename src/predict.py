import sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.preprocessing import image
from keras.models import load_model

# get the file name from the command line arguments
file_name = sys.argv[1]

# load the image and invert it
img = Image.open(file_name).convert('L')  # L is for grayscale
img = ImageOps.invert(img)

# resize the image to 28x28
img = img.resize((28, 28))

# apply the EDGE_ENHANCE_MORE filter
img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

# convert the image to an array
img_array = image.img_to_array(img)

# preprocess the image
img_array = img_array.astype('float32')
img_array /= 255

# save the preprocessed image
image.save_img('preprocessed_image.jpg', img_array)

# load the model
model = load_model('model.h5')

# reshape the image for the model
input_shape = model.input_shape[1:]
img_array = img_array.reshape((1, *input_shape))

# make a prediction
prediction = model.predict(img_array)

# print the best predicted digit
print(np.argmax(prediction))
