import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    # print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*3/5) # Make the line shorter
    '''
    y = mx+b
    x = (y-b)/m
    '''
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def averaged_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # returns slope and y-intercept(1 is the degree of polynomial)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1] # lines on the left will have negative slope(as x decreases when y increases) and on the right will have positive slope(as x increases when y increases)
        if slope<0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            # print(left_fit)
            # print(right_fit)
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    # print(left_fit_average, 'left')
    # print(right_fit_average, 'right')

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            # print(line.reshape(4))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# #For static image
# image = cv2.imread('Image/Image/test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)

# # cv2.imshow("Canny", canny)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # np.array([]) is array to accumulate votes in bins
# averaged_lines = averaged_slope_intercept(lane_image, lines)
# print(lines)
# # line_image = display_lines(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow("result_line_image", line_image)
# cv2.imshow("result_combo_image", combo_image)

# '''
# This Method of Overlaying is more robust:

# line_image = np.zeros_like(lane_image)
# cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# # Create a mask where line is drawn
# mask = line_image[:, :, 0] > 0  # since line is blue in BGR

# # Place line on top without mixing
# combo_image = lane_image.copy()
# combo_image[mask] = line_image[mask]
# '''

# # plt.imshow(canny)
# # plt.show()
# cv2.waitKey(0)


cap = cv2.VideoCapture("test2.mp4/test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)

    # cv2.imshow("Canny", canny)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # np.array([]) is array to accumulate votes in bins
    averaged_lines = averaged_slope_intercept(frame, lines)
    # print(lines)
    # line_image = display_lines(lane_image, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result_line_image", line_image)
    cv2.imshow("result_combo_image", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()