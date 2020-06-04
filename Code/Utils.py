'''
This file contains all required methods for implementation
'''
import math
import numpy as np
import cv2.cv2 as cv2
from main import HoughFunction

original_subset_path = r"C:\Users\AliIhsan\Desktop\Assignment_1\Dataset\Original_Subset\\"
detection_subset_path = r"C:\Users\AliIhsan\Desktop\Assignment_1\Dataset\Detection_Subset\\"


# Loads original and detection images then returns the required image
def prepare_im(img_num):
    global original_im_path
    global detection_im_path
    # Paths for Images
    original_im_path = original_subset_path + str(img_num) + ".png"
    detection_im_path = detection_subset_path + str(img_num) + ".png"

    # Read Images
    color_im = cv2.imread(original_im_path)
    main_im = cv2.imread(original_im_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(detection_im_path, cv2.IMREAD_GRAYSCALE)

    # Mask the barcode part of the image
    main_im = cv2.bitwise_and(main_im, mask, mask)

    # Get the region of interest and crop rest of it
    _, thresh = cv2.threshold(main_im, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_im = color_im[y:y + h, x:x + w]

    return cropped_im

# Creates sinusoidals using x, y and theta values
def create_hough_function(x_coordinate_value, y_coordinate_value, theta_values):
    rho_values = np.round(
        x_coordinate_value * np.cos((theta_values / 180) * np.pi) + y_coordinate_value * np.sin(
            (theta_values / 180) * np.pi)).astype(
        int)
    hough_function = HoughFunction()
    hough_function.theta_values = theta_values
    hough_function.rho_values = rho_values
    return hough_function

# Finds max and min points in a hough function list
def find_hough_rho_value_extremes(hough_functions):
    min_rho_val = float('inf')
    max_rho_val = float('-inf')
    for hough_function in hough_functions:
        temp_min_y_val = min(hough_function.rho_values)
        temp_max_y_val = max(hough_function.rho_values)
        if (temp_min_y_val < min_rho_val):
            min_rho_val = temp_min_y_val
        if (temp_max_y_val > max_rho_val):
            max_rho_val = temp_max_y_val
    return (max_rho_val, min_rho_val)


# Plots lines on a given image. Calculates points using inverse transformation from polar to cartesian coordinates
# Uses just two points-namely "start_point" and "end_point"- to draw a line to make algorithm run faster
def plot_found_lines(accumulator, img, min_rho_val):
    theta_values = accumulator[1]
    rho_values = accumulator[0] + min_rho_val
    x_values = np.arange(0, img.shape[1])
    y_values = np.arange(0, img.shape[0])

    for index in range(len(theta_values)):

        theta_val = theta_values[index]
        rho_val = rho_values[index]

        start_point = None
        end_point = None

        if theta_val == 0 or theta_val == 180:
            y_value = 0
            x_value = (int)(np.abs(rho_val))
            start_point = (x_value, y_value)
            y_value = max(y_values)
            end_point = (x_value, y_value)

        else:
            x_value = 0
            y_value = np.floor(
                (-1 / math.tan((theta_val / 180) * math.pi)) * x_value + rho_val / math.sin(
                    (theta_val / 180) * math.pi)).astype(
                int)

            if y_value >= 0 and y_value <= max(y_values):
                start_point = (x_value, y_value)

            x_value = max(x_values)
            y_value = np.floor(
                (-1 / math.tan((theta_val / 180) * math.pi)) * x_value + rho_val / math.sin(
                    (theta_val / 180) * math.pi)).astype(
                int)
            if y_value >= 0 and y_value <= max(y_values):
                if start_point == None:
                    start_point = (x_value, y_value)
                else:
                    end_point = (x_value, y_value)
            if start_point == None or end_point == None:
                y_value = 0
                x_value = np.floor(((rho_val / math.sin((theta_val / 180) * math.pi)) - y_value) * math.tan(
                    (theta_val / 180) * math.pi)).astype(int)

                if x_value >= 0 and x_value <= max(x_values):
                    if start_point == None:
                        start_point = (x_value, y_value)
                    else:
                        end_point = (x_value, y_value)
                if end_point == None:
                    y_value = max(y_values)
                    x_value = np.floor(((rho_val / math.sin((theta_val / 180) * math.pi)) - y_value) * math.tan(
                        (theta_val / 180) * math.pi)).astype(int)
                    end_point = (x_value, y_value)

        cv2.line(img, start_point, end_point, (0, 0, 255),thickness=2)


# Votes for every theta and rho value
def calculate_vote_values(hough_functions, accumulator, min_rho_val):
    for hough_function in hough_functions:
        for point_index in range(len(hough_function.theta_values)):
            rho_value = hough_function.rho_values[point_index] - min_rho_val
            theta_value = hough_function.theta_values[point_index]
            accumulator[rho_value][theta_value] += 1

# Zero outs every neighbor of a peak point which is closer than suppression_distance
def apply_neighborhood_suppression(accumulator, suppression_distance):
    if len(accumulator) <= 2:
        pass
    rho_values = accumulator[0]
    theta_values = accumulator[1]
    value_1_index = 0
    while value_1_index < len(rho_values) - 1:

        value_2_index = value_1_index + 1
        while value_2_index < len(rho_values):

            total_distance = abs(rho_values[value_1_index] - rho_values[value_2_index]) + abs(
                theta_values[value_1_index] - theta_values[value_2_index])

            if total_distance <= suppression_distance:
                rho_values = np.delete(rho_values, value_2_index)
                theta_values = np.delete(theta_values, value_2_index)
            else:
                value_2_index += 1

        value_1_index += 1
    new_accumulator = (rho_values, theta_values)
    return new_accumulator


