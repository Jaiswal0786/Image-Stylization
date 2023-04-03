# Neural Style Transfer using Flask

This is a Python script for a Flask web application that performs neural style transfer on two uploaded images. The generated image will have the content of the first image and the style of the second image.

## Web-Link
`AbhiJais0786.pythonanywhere.com`
## Required Libraries

* Flask
* Pillow (PIL)
* NumPy
* OpenCV (cv2)
* Tensorflow

## Usage

1. Run the script.
2. Navigate to localhost in a web browser.
3. Upload a content image and a style image.
4. Click the "Upload" button.
5. Wait for the script to generate the output image.
6. The generated image will be displayed on the web page.

## How it Works
1. The script checks if the uploaded files are images in PNG, JPEG, or JPG format.
2. If the files are valid, the script saves them to the uploads/ folder.
3. The script then uses the load_img() function to load the content and style images as NumPy arrays and preprocesses them.
4. The script defines the content and style layers to extract from a pre-trained VGG19 network, which will be used to compute the loss functions.
5. The script defines functions to compute the content and style loss functions.
6. The script defines a function to generate the target image, which will be the generated image.
7. The script defines the train_step() function, which performs a single step of optimization to minimize the total loss.
8. The script runs a loop to perform the optimization, using the Adam optimizer to minimize the loss.
9. The generated image is saved to the static/ folder with a randomly generated filename.
10. The generated image is displayed on the web page.

## Notes
1. This script uses the pre-trained VGG19 network provided by TensorFlow Keras.
2. The load_img() function resizes the images to a maximum dimension of 512 while maintaining the aspect ratio.
3. The train_step() function performs 10 iterations of optimization per call.
4. The total loss is the sum of the content loss and the style loss multiplied by their respective weights.
