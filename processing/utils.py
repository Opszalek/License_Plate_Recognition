import cv2
import numpy as np
import os
from processing import font_generator
images = {}
dictionary = {}

def load_letters_from_directory():
    """
    Load letter images from the directory.
    """
    for i in os.listdir('processing/letters'):
        if (i.endswith(".png")):
            dictionary[i] = cv2.imread('processing/letters/' + i, cv2.IMREAD_GRAYSCALE)




def resize_image(img):
    """
    Resize the image while maintaining the aspect ratio.
    :param img: Input image.
    :return: Resized grayscale image.
    """
    scale_percent = 1500 / img.shape[1] * 100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    return img_resized


def average_rect_area(points):
    """
    Calculate the average area of two pairs of length and width rectangle.
    :param points: Four corner points of the rectangle.
    :return: Average area of the rectangle.
    """
    a = abs(points[2][0][0] - points[0][0][0]) * abs(points[2][0][1] - points[0][0][1])
    b = abs(points[1][0][0] - points[3][0][0]) * abs(points[1][0][1] - points[3][0][1])
    return int((a + b) / 2)


def sort_rectangle_corners(contour):
    """
    Sort the four corner points of a rectangle contour in a specific order.
    :param contour: Contour representing a rectangle.
    :return: Sorted array of four corner points.
    """
    x = sorted(contour, key=lambda x: x[0][0])
    left = sorted(x[:2], key=lambda x: x[0][1])
    right = sorted(x[2:], key=lambda x: x[0][1])
    return np.array([left[0], right[0], left[1], right[1]], dtype=np.float32)


def is_rect_inside_rect(rect1, rect2):
    """
    Check if rectangle is inside rectangle.
    :param rect1: First rectangle.
    :param rect2: Second rectangle.
    :return: True if rectangle is inside rectangle, False otherwise.
    """
    return (rect2[0] <= rect1[0] <= rect1[2] <= rect2[2]) and (rect2[1] <= rect1[1] <= rect1[3] <= rect2[3])


def is_rect_intersect_rect(rect1, rect2):
    """
    Check if rect1 intersects with rect2.
    :param rect1: First rectangle.
    :param rect2: Second rectangle.
    :return: True if rect1 intersects with rect2, False otherwise.
    """
    return not (rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or rect1[3] <= rect2[1] or rect1[1] >= rect2[3])


def rectangle_area(rect):
    """
    Calculate the area of a rectangle.
    :param rect: Rectangle coordinates (x1, y1, x2, y2).
    :return: Area of the rectangle.
    """
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def return_biggest_rectangle(rect_list, is_plate=False) -> list:
    """
    Function takes list of rectangles and returns list of biggest rectangles
    :param rect_list: list of rectangles
    :param is_plate: if True function returns only rectangles with area bigger than 9000
    :return: list of biggest rectangles
    """
    rect_list_cleaned = []
    for rect_1 in rect_list:
        is_biggest_rect = True
        for rect_2 in rect_list:
            if rect_1 != rect_2 and (is_rect_inside_rect(rect_2, rect_1) or is_rect_inside_rect(rect_1, rect_2)):
                if rectangle_area(rect_1) < rectangle_area(rect_2):
                    is_biggest_rect = False
                    break
        if is_biggest_rect:
            if is_plate == False:
                rect_list_cleaned.append(rect_1)
            elif is_plate and rectangle_area(rect_1) > 9000:
                rect_list_cleaned.append(rect_1)

    return rect_list_cleaned

def check_retange_ratio(corners)->bool:
    """
    Function checks ratio of rectangle
    :param corners: rectangle corners
    :return: True if ratio is between values defined by user
    """
    ratio_min = 3.5
    ratio_max = 6
    if ratio_min < ((corners[1][0][0] - corners[0][0][0]) / (corners[2][0][1] - corners[0][0][1])) < ratio_max or \
    ratio_min < ((corners[1][0][0] - corners[0][0][0]) / (corners[3][0][1] - corners[1][0][1])) < ratio_max:
        return True
    else:
        return False


def warp_plate_image(image, plate):
    """
    Takes image, plate coorinates and returns plate
    :param image: whole image
    :param plate: image of plate
    """
    if plate is not None:
        corners = sort_rectangle_corners(plate)
        width = 1200  ####
        height = 300  ####
        dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))
        return warped

    else:
        return image

def find_best_plate_coordinates(image, contours, plate_coordinates=None):
    """
    Find the best plate coordinates in the image.
    :param image: Whole image.
    :param contours: Contours found in the image.
    :param plate_coordinates: Existing plate coordinates.
    :return: Best plate coordinates.
    """
    img_wide = image.shape[1]
    for contour in contours:
        approxedRectangle = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approxedRectangle) == 4 and cv2.contourArea(approxedRectangle) > 70000:
            corners = sort_rectangle_corners(approxedRectangle)
            if corners[1][0][0] - corners[0][0][0] > 0.28 * img_wide:
                if check_retange_ratio(corners):
                    if plate_coordinates is None:
                        plate_coordinates = approxedRectangle
                    elif average_rect_area(approxedRectangle) < average_rect_area(plate_coordinates):
                        plate_coordinates = approxedRectangle

    return plate_coordinates

def find_plate(image):
    """
    Find the plate in the whole image.
    :param image: whole image
    :return: image of plate
    """
    thresh = cv2.adaptiveThreshold(image, 214, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate_coordinates=find_best_plate_coordinates(image,contours)
    plate = warp_plate_image(image,plate_coordinates)

    return plate


def find_letters(image, is_plate=False):
    """
    Find letters in the image.
    :param image: image with plate
    :param is_plate: if True - find letters on plate, else - find letters on image
    :return: list of letters
    """
    edges = cv2.Canny(image, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        if image.shape[1] > 100:
            if (cv2.contourArea(contour) > 1000 and cv2.boundingRect(contour)[2] < image.shape[1] / 5):
                x, y, w, h = cv2.boundingRect(contour)
                rectangles.append([x, y, x + w, y + h])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            rectangles.append([x, y, x + w, y + h])

    rectangles = return_biggest_rectangle(rectangles, is_plate)
    letters = []

    rectangles.sort(key=lambda x: x[0])

    for rect in rectangles:
        letters.append(image[rect[1]:rect[3], rect[0]:rect[2]])
    return letters


def compare_images(plate, dictionary):
    """
    Compare the letters in the plate image with the letters in the dictionary.
    :param plate: Image of the plate.
    :param dictionary: Dictionary of letters.
    :return: List of matched letters.
    """
    tablica = ['P','P','P','P','P','P','P']
    i=0
    for letter in plate:
        letter = cv2.GaussianBlur(letter, (5, 5), 0)
        ret, letter = cv2.threshold(letter, 140, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        letter = cv2.morphologyEx(letter, cv2.MORPH_OPEN, kernel, iterations=3)
        best_match = None
        best_match_score = float('inf')

        for key, char in dictionary.items():
            char = cv2.resize(char[0], (letter.shape[1], letter.shape[0]))

            diff = cv2.absdiff(char, letter)
            score = np.sum(diff)

            if score < best_match_score:
                best_match = key
                best_match_score = score
        if i >=7:
            tablica.append(best_match[0])
        else:
            tablica[i]=best_match[0]
        i+=1
    return tablica


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    if len(dictionary) == 0:
        load_letters_from_directory()
        if not dictionary:
            font_generator.Czcionka().generate_letters()
            load_letters_from_directory()
        for key, le in dictionary.items():
            dictionary[key] = find_letters(le)

    image = resize_image(image)
    plate = find_plate(image)
    letters = find_letters(plate,is_plate=True)
    plate_text = compare_images(letters, dictionary)
    plate_text = "".join(plate_text)

    return plate_text
