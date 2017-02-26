# Behavioral Cloning for Self Driving Cars

### Overview

This is the third project in the [SDC Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013 "SDC Engineer Nanodegree")  from [Udacity](https://www.udacity.com/ "Udacity"), where we train a deep learning model using RGB images generated by 3 cameras installed on the front hood cover of the car, with the objective of predicting the steering angle. Driving a car is more complex than only how to steer the car: ideally, we would like the model to also deal with throttle, breaking and control speeds at specific segments of the road, but for the purpose of this project, we're only trying to model for the steering angle.

This README is structured as follows: 

* [Data: Sources, Exploration, Enrichment and Preprocessing](#section_0).
* [Generator Functions](#section_1) for the training and validation data.
* [Model Architecture](#section_2) Description of the layers, overfitting methods, etc.
* [Model Training](#section_3)
* [Discussion](#section_4)
* [Results](#section_5)

The model parameters file and a video of a full round in each track are provided.


### [Data: Sources, Exploration, Enrichment and Preprocessing](#section_0)

Two sources of data are used for training the model in this project. The first source is Udacity's own data. Each data point provides 3 images from three cameras at the front hood cover of the car (left, center and right), along with speed, throttle, break and steering angle.

This dataset is imbalanced since the majority of datapoints are within a small range from straight driving (within the range of -0.15 to 0.15). See figure \ref{udacity_data_hist} for the histogram of steering angle values.Training the model using this source provided very unreliable results in the simulator.

![{udacity_data_hist}](figs/udacity_data.png)

Some ways to enrich the data with more left/right turn points is to record using manual driving in the simulator, and add a recovery angle to the images from the left/right camera and add them to the dataset as their own points.

For sharp turn data, I used [this small data set](https://github.com/cssomnath/udacity-sdc/blob/master/carnd-projects/CarND-Behavioral-Cloning/sharp_turn.zip "Sharp Turn data") recorded by another Udacity SDC student (Kudos to [Somnath Banerjee](https://github.com/cssomnath Somnath Banerjee)) and combined it with the Udacity data set, as recording data using keyboard/mouse does not capture angles smoothly. The histogram for the sharp-turn data is \ref{sharp_turn_hist}.



```python
header_row = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
udacity_data = pd.read_csv('driving_log.csv', skiprows=[0], names=header_row)
sharp_turn_data = pd.read_csv('sharp_turn.csv')

# Combine the two data sources ...
data = pd.concat([udacity_data, sharp_turn_data])
```

![{sharp_turn_hist}](figs/sharp_turn_data.png)

After some trial-and-error, limiting the number of data points corresponding to steering-angle ranges, I ended up using a dataset containing 3871 images with the following histogram:

![{equalized_data_hist}](figs/equalized_data.png)

Preprocessing happens in the following steps:

* Normalize the pixel values to be within the range of [-0.5 - 0.5], performed by a [Lambda Layer](https://keras.io/layers/core/#lambda "Lambda Layer") in the network.

* Crop the image to the area of interest, by removing the sky and the lower part of the image, then resize it to 64x64x3.

```python
def crop_resize(img):
    """
    :param img: original image
    :return: cropped: img - without the sky part, resized to fit the input
                        size requirement for the CNN
    """
    cropped = cv2.resize(img[60:140, :], (64, 64))
    return cropped

```

The two figures below shows an image from a center camera before and after cropping and resizing.

![{center_img}](figs/center_img.png)
![{cropped_resized}](figs/cropped_resized.png)



* Flipping images: Randomly mirror the image L->R or R->L and multiply the angle by -1, based on a coin flip. This is performed on the training data only by the generator, to enrich the dataset.

```python
def flip(img, steer_angle):
    """
    :param img: camera input
    :param steer_angle: steering angle
    :return: new_img: Flipped, along the y axis.
             new_angle: steer_angle multiplied by -1
    """
    new_img = cv2.flip(img, 1)
    new_angle = steer_angle*(-1)
    return new_img, new_angle
```

* Random Brightness changes: Used by the generator for training data only, to convert the image to HSV space and multiply the V channel with a random number from a uniform distribution in the range of [0.25 - 1.0] to make the model generalize to environments with different brightness levels.

```python

def random_brightness(img):
    """
    Convert the image to HSV, and multiply the brightness channel [:,:,2]
    by a random number in the range of [0.25 to 1.0] to get different levels of
    brightness.
    :param img: normalized image in RGB color space
    :return: new_img: Image in RGB Color space
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.25,1.0)
    hsv_img[:, :, 2] = rand*hsv_img[:, :, 2]
    new_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return new_img
```


### [Generator Functions](#section_1)

### [Model Architecture](#section_2)

### [Model Training](#section_3)

### [Discussion](#section_4)

### [Results](#section_5)




