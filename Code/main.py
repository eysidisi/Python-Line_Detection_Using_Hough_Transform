import numpy as np
import timeit
import Utils
import cv2.cv2 as cv2


class HoughFunction():
    def __init__(self):
        self.theta_values = []
        self.rho_values = []


if __name__ == "__main__":
    # Images are labeled from 1 to 14
    start_im_num = 1
    end_im_num = 14

    # Values for thresholding boxes
    min_threshold_factor = 0.7
    max_threshold_factor = 0.9
    threshold_increase_val = 0.1

    # Folder path to save the results
    save_folder_path = r"C:\Users\AliIhsan\Desktop\Assignment_1\Results\\"

    for img_num in range(start_im_num, end_im_num + 1, 1):

        # Threshold_factor is used for eliminating Hough Boxes based on their vote values
        # 3 different threshold_factor are used to see the differences
        for threshold_factor in np.arange(min_threshold_factor, max_threshold_factor + (threshold_increase_val / 10),
                                          threshold_increase_val):

            # Path to save the results
            result_path = save_folder_path + "Im_" + str(img_num) + "_Threshold_Factor_" + str(
                threshold_factor) + ".png"

            # Total time that passes for algorithm
            total_time = 0

            # This list will contain all the images in the process and will be saved as result
            result_images_list = []

            # Get image
            start = timeit.default_timer()
            color_img = Utils.prepare_im(img_num)
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it takes to get image ready: " + str(stop - start))

            # Convert image into grey and create a 3 channel grey image to put in result image list
            grey_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            grey_im_3_channel = cv2.cvtColor(grey_img, cv2.COLOR_GRAY2BGR)
            color_img_copy = np.copy(color_img)
            result_images_list.append(grey_im_3_channel)

            diameter = 7
            sigma_color = sigma_space = 100

            # Apply bilateral filter to smooth the image
            start = timeit.default_timer()
            bilateral_filtered_im = cv2.bilateralFilter(grey_img, diameter, sigma_color, sigma_space)
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it takes to apply bilateral filter: " + str(stop - start))

            # Get three channel filtered im and append to result_images_list
            bilateral_filtered_3_channel = cv2.cvtColor(bilateral_filtered_im, cv2.COLOR_GRAY2BGR)
            result_images_list.append(bilateral_filtered_3_channel)

            canny_threshold_val_1 = 100
            canny_threshold_val_2 = 200

            # Apply canny for edge detection
            start = timeit.default_timer()
            canny_im = cv2.Canny(bilateral_filtered_im, canny_threshold_val_1, canny_threshold_val_2)
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it takes to apply canny filter: " + str(stop - start))

            # Get three channel canny im and append to result_images_list
            canny_3_channel_im = cv2.cvtColor(canny_im, cv2.COLOR_GRAY2BGR)
            result_images_list.append(canny_3_channel_im)

            # Get edge coordinates
            edge_y_indices, edge_x_indices = np.nonzero(canny_im)

            # Get theta values for creating Hough Function Sinusoidals
            angle_step = 1
            min_theta_val = 0
            max_theta_val = 180
            theta_values = (np.arange(min_theta_val, max_theta_val + 1, angle_step))

            hough_functions = []

            # Create hough functions and append to list
            start = timeit.default_timer()
            for edge_index in range(len(edge_y_indices)):
                hough_functions.append(
                    Utils.create_hough_function(edge_x_indices[edge_index], edge_y_indices[edge_index], theta_values))
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it takes to create hough functions: " + str(stop - start))

            # Find extreme rho values. These values will be used for creating accumulator array
            start = timeit.default_timer()
            max_rho_val, min_rho_val = Utils.find_hough_rho_value_extremes(hough_functions)
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it takes to find max and min y values hough functions: " + str(stop - start))

            # Create accumulator
            number_of_accumulator_rhos = max_rho_val - min_rho_val + 1
            number_of_accumulator_thetas = max_theta_val + 1
            accumulator = np.zeros((number_of_accumulator_rhos, number_of_accumulator_thetas), dtype=np.uint8)

            # Fill accumulator
            start = timeit.default_timer()
            Utils.calculate_vote_values(hough_functions, accumulator, min_rho_val)
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it calculate hough box vote values using accumulator: " + str(stop - start))

            # Apply threshold to accumulator to eliminate weak points
            start = timeit.default_timer()
            max_val_of_acc = np.amax(accumulator)
            threshold = threshold_factor * max_val_of_acc
            eliminated_accumulator = np.where(accumulator > threshold)
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it takes to eliminate hough boxes : " + str(stop - start))

            # Apply neighborhood suppression to get rid of very close lines
            suppression_distance = 3
            start = timeit.default_timer()
            eliminated_accumulator = Utils.apply_neighborhood_suppression(eliminated_accumulator, suppression_distance)
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it takes to apply neighborhood_suppression: " + str(stop - start))
            max_val_of_acc = np.amax(accumulator)

            # This white image is used to draw the resultant lines
            white_im = np.ones((color_img.shape[0], color_img.shape[1], 3), np.uint8) * 255

            # Plot lines on white and original image
            start = timeit.default_timer()
            Utils.plot_found_lines(eliminated_accumulator, color_img, min_rho_val)
            Utils.plot_found_lines(eliminated_accumulator, white_im, min_rho_val)
            stop = timeit.default_timer()
            total_time += (stop - start)
            print("Time it takes to plot lines : " + str(stop - start))
            result_images_list.append(white_im)
            result_images_list.append(color_img)

            print("Total time it takes " + str(total_time) + " to apply algorithm on image : " + str(
                img_num) + " with thresholding factor: " + str(threshold_factor))

            # Concatenate images for saving the result
            place_holder_im = np.ones((color_img.shape[0], 10, 3), np.uint8) * 100
            horizontal_concat_images = result_images_list[0]
            im_index = 1
            while im_index < len(result_images_list):
                horizontal_concat_images = np.concatenate((horizontal_concat_images, place_holder_im), axis=1)
                horizontal_concat_images = np.concatenate((horizontal_concat_images, result_images_list[im_index]),
                                                          axis=1)
                im_index += 1

            cv2.imwrite(result_path, horizontal_concat_images)
    quit()
