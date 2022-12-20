# Sample for classifying digits from handwritten images

## Description

This code is a script for preprocessing an image and making a prediction about what digit it represents using a trained machine learning model.

The script first imports several libraries that it needs for image processing and for loading the trained machine learning model. It then gets the file name of the image to be processed from the command line arguments.

Next, the script loads the image, converts it to grayscale, and inverts it. It then resizes the image to 28x28 pixels and applies the EDGE_ENHANCE_MORE filter to the image. This filter is used to enhance the edges in the image, which can be useful for identifying the digits in the image.

After preprocessing the image, the script converts it to an array and preprocesses the array by converting it to a float data type and normalizing it by dividing it by 255. The script then saves the preprocessed image.

Finally, the script loads the trained machine learning model and uses it to make a prediction about what digit is represented in the image. It prints the best predicted digit to the console.

## python libraries

`pip install numpy keras tensorflow`

## run

`python ./src/predict.py HAND_WRITTEN_IMAGE.png`

### Input image

a 500x500 image created by Windows Paint app with Maximum pencil size, white background and black foreground color.
