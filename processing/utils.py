import cv2
import numpy as np
import os
#import images from images folder
path = "/home/opszalek/PycharmProjects/SW_projekt/images"
path = os.path.abspath(path)
images = {}
dictionary = {}

def import_images():
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[filename]=img

def import_letters():
    for i in os.listdir('/home/opszalek/PycharmProjects/SW_projekt/letters'):
        if (i.endswith(".png")):
            dictionary[i]=cv2.imread('/home/opszalek/PycharmProjects/SW_projekt/letters/'+i, cv2.IMREAD_GRAYSCALE)

def resize_images(img):
    scale_percent = 1500 / img.shape[1] *100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img_resized
def area_of_rectangle(points):
    a=abs(points[2][0][0] - points[0][0][0]) * abs(points[2][0][1] - points[0][0][1])
    b=abs(points[1][0][0] - points[3][0][0]) * abs(points[1][0][1] - points[3][0][1])
    return int((a+b)/2)
def sort_corners(contour):
    x=sorted(contour, key=lambda x: x[0][0])
    left=sorted(x[:2], key=lambda x: x[0][1])
    right=sorted(x[2:], key=lambda x: x[0][1])
    return np.array([left[0],right[0],left[1],right[1]], dtype=np.float32)
def is_rect_inside_rect(rect1, rect2):
    return (rect2[0] <= rect1[0] <= rect1[2] <= rect2[2]) and (rect2[1] <= rect1[1] <= rect1[3] <= rect2[3])
def is_rect_intersect_rect(rect1, rect2):
    return not (rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or rect1[3] <= rect2[1] or rect1[1] >= rect2[3])
def rect_area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])
def return_biggest(rect_list,letter=False):
    rect_list_cleaned = []
    for rect1 in rect_list:
        is_max_rect = True
        for rect2 in rect_list:
            if rect1 != rect2 and (is_rect_inside_rect(rect2, rect1) or is_rect_inside_rect(rect1, rect2)):
                if rect_area(rect1) < rect_area(rect2):
                    is_max_rect = False
                    break
        if is_max_rect:
            if letter==True:
                rect_list_cleaned.append(rect1)
            elif letter==False and rect_area(rect1)>9000:
                rect_list_cleaned.append(rect1)

    return rect_list_cleaned

def find_contours(image):
    thresh = cv2.adaptiveThreshold(image, 214, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_wide = image.shape[1]
    tablica = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 70000:
            points = sort_corners(approx)
            if points[1][0][0] - points[0][0][0] > 0.28 * img_wide:
                if 3.5 < ((points[1][0][0] - points[0][0][0]) / (points[2][0][1] - points[0][0][1])) < 6 or \
                        3.5 < ((points[1][0][0] - points[0][0][0]) / (points[3][0][1] - points[1][0][1])) < 6:
                    if tablica is None:
                        tablica = approx
                    elif area_of_rectangle(approx) < area_of_rectangle(tablica):
                        tablica = approx
    if tablica is not None:
        points = sort_corners(tablica)
        width = 1200  ####
        height = 300  ####
        dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(points, dst_points)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))
        plates = warped
    else:
        plates = image
    return plates

def find_letters(image,letter=False):

    edges = cv2.Canny(image, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles=[]
    for cnt in contours:
        if image.shape[1] > 100:
            if (cv2.contourArea(cnt) > 1000 and cv2.boundingRect(cnt)[2] < image.shape[1] / 5):
                x, y, w, h = cv2.boundingRect(cnt)
                rectangles.append([x,y,x+w,y+h])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            rectangles.append([x, y, x + w, y + h])

    rectangles=return_biggest(rectangles,letter)
    letters=[]


    rectangles.sort(key=lambda x: x[0])

    for rect in rectangles:
        letters.append(image[rect[1]:rect[3],rect[0]:rect[2]])
    return letters

def compare_images(plate, dictionary):
    tablica = []

    for letter in plate:
        letter = cv2.GaussianBlur(letter, (5, 5), 0)
        ret, letter = cv2.threshold(letter, 140, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        letter = cv2.morphologyEx(letter, cv2.MORPH_OPEN, kernel, iterations=3)
        best_match = None
        best_match_score = float('inf')
        # cv2.imshow('letter', letter)

        for key,char in dictionary.items():
            char = cv2.resize(char[0], (letter.shape[1], letter.shape[0]))

            diff = cv2.absdiff(char, letter)
            score = np.sum(diff)

            if score < best_match_score:
                best_match = key
                best_match_score = score

                # cv2.imshow('char', char)
                # print(score)

        tablica.append(best_match[0])
        # cv2.waitKey(0)
    return tablica




def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    #TODO: add image processing here
    if len(dictionary) == 0:
        import_letters()
        for key, le in dictionary.items():
            dictionary[key] = find_letters(le,letter=True)

    image=resize_images(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate = find_contours(image)
    letters=find_letters(plate)
    plate_text=compare_images(letters, dictionary)
    plate_text="".join(plate_text)
    # print(plate_text)
    # cv2.waitKey(0)

    # cv2.imshow("img", image)
    # cv2.imshow("img2", plate)
    # cv2.waitKey(0)
    # import_images()
    # import_letters()
    # plates = {}
    # # images = resize_images()
    # # plates=find_contours(images)
    # for key, img in images.items():
    #     images[key] = resize_images(img)
    #
    # for key, img in images.items():
    #     plates[key] = find_contours(img)
    #
    # for key, plate in plates.items():
    #     plates[key] = find_letters(plate)
    #
    # for key, le in dictionary.items():
    #     dictionary[key] = find_letters(le)
    #

    #
    # for key, value in plates.items():
    #     plates[key] = compare_images(value, dictionary)
    # for key, value in plates.items():
    #     print(key)
    #     print(value)
    return plate_text
