import numpy as np
import pandas as pd
import cv2
import os
import random
from sklearn.model_selection import train_test_split


def load_data(data_path):

    # Load driving log
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(os.path.join(data_path, 'driving_log.csv'), names=columns)

    # Use only center images and steering
    image_paths = data['center'].values
    steering_angles = data['steering'].values

    return image_paths, steering_angles

def split_data(image_paths, steering_angles):

    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths,
        steering_angles,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_valid, y_train, y_valid

def preprocess_image(image):

    # Crop image (remove sky and hood)
    image = image[60:135, :, :]

    # Convert to YUV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Resize to Nvidia model input size
    image = cv2.resize(image, (200, 66))

    # Normalize
    image = image / 255.0

    return image

def load_image(data_path, image_path):

    # Clean spaces and extract filename safely
    filename = os.path.basename(image_path.strip())

    # Construct full path
    full_path = os.path.join(data_path, 'IMG', filename)

    # Read image
    image = cv2.imread(full_path)

    return image

def zoom_image(image):
    zoom_factor = random.uniform(1.0, 1.3)
    height, width = image.shape[:2]

    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    x1 = random.randint(0, width - new_width)
    y1 = random.randint(0, height - new_height)

    image = image[y1:y1 + new_height, x1:x1 + new_width]
    image = cv2.resize(image, (width, height))

    return image


def pan_image(image, steering):
    height, width = image.shape[:2]

    x_shift = random.randint(-40, 40)
    y_shift = random.randint(-10, 10)

    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    image = cv2.warpAffine(image, matrix, (width, height))

    steering += x_shift * 0.002

    return image, steering


def adjust_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_scale = random.uniform(0.5, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_scale, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image


def rotate_image(image, steering):
    height, width = image.shape[:2]
    angle = random.uniform(-10, 10)

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    image = cv2.warpAffine(image, matrix, (width, height))

    steering += angle * 0.002

    return image, steering


def flip_image(image, steering):
    image = cv2.flip(image, 1)
    steering = -steering

    return image, steering


def random_augment(image, steering):

    if random.random() < 0.5:
        image, steering = pan_image(image, steering)

    if random.random() < 0.5:
        image = zoom_image(image)

    if random.random() < 0.5:
        image = adjust_brightness(image)

    if random.random() < 0.5:
        image, steering = rotate_image(image, steering)

    if random.random() < 0.5:
        image, steering = flip_image(image, steering)

    return image, steering

def batch_generator(data_path, image_paths, steering_angles, batch_size, is_training):

    while True:
        images = []
        steers = []

        while len(images) < batch_size:
            index = np.random.randint(0, len(image_paths))

            image = load_image(data_path, image_paths[index])
            steering = float(steering_angles[index])

            if image is None:
                continue

            if is_training:
                image, steering = random_augment(image, steering)

            image = preprocess_image(image)

            images.append(image)
            steers.append(steering)

        yield np.asarray(images), np.asarray(steers)

