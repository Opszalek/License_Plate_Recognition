import cv2
import numpy as np
import os
import math

#import images from images folder
path = "/home/opszalek/PycharmProjects/SW_projekt/images"
path = os.path.abspath(path)
images = []
tablice = []
dictionary = []

def import_images():
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
def import_letters():
    for i in os.listdir('/home/opszalek/PycharmProjects/SW_projekt/letters'):
        if (i.endswith(".png")):
            dictionary.append(cv2.imread('/home/opszalek/PycharmProjects/SW_projekt/letters/'+i, cv2.IMREAD_GRAYSCALE))

def resize_images():
    img_resized = []
    for img in images:
        scale_percent = 1500 / img.shape[1] *100
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_resized.append(img)
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
def return_biggest(rect_list):
    rect_list_cleaned = []
    for rect1 in rect_list:
        if (rect1 not in rect_list_cleaned):
            for rect2 in rect_list:
                if (rect2 not in rect_list_cleaned):
                    if rect1 != rect2 and (is_rect_inside_rect(rect2, rect1) or is_rect_inside_rect(rect2, rect1)) and (is_rect_intersect_rect(rect1, rect2)):
                        if rect_area(rect1) > rect_area(rect2):
                            rect_list_cleaned.append(rect1)
                        elif rect_area(rect1) < rect_area(rect2):
                            rect_list_cleaned.append(rect2)
                            ###
                            ##Tutaj trzeba ogarnąć jak uzunąć z listy rect1 i rect2 jeśli znajde wiekszy od nich
                            ###


    return rect_list_cleaned


def find_contours(images):
    plates = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_wide = image.shape[1]
        tablica = None
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.contourArea(cnt) > 50000:
                points = sort_corners(approx)
                if points[1][0][0]-points [0][0][0]> 0.2 * img_wide:
                    if 3 < ((points[1][0][0]-points [0][0][0]) / (points[2][0][1]-points [0][0][1])) <6 or \
                            3 < ((points[1][0][0]-points [0][0][0]) / (points[3][0][1]-points [1][0][1])) <6:
                        area_of_rectangle(approx)
                        if tablica is None:
                            tablica=approx
                        elif area_of_rectangle(approx) < area_of_rectangle(tablica):
                            tablica=approx

        if tablica is not None:
            points = sort_corners(tablica)
            width = 1200 ####
            height = 300 ####
            dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(points, dst_points)
            warped = cv2.warpPerspective(image, M, (int(width), int(height)))
            plates.append(warped)
            # cv2.imshow("Warped Image", warped)
            # cv2.waitKey(0)

        #cv2.drawContours(image, contours, -1, (0, 255, 255), 3)
        #cv2.drawContours(image, contours_, -1, (0, 255, 0), 3)
        #cv2.imshow("img", image)
        #cv2.waitKey(0)
    return plates

def find_letters(plates):
    for plate in plates:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        letter=[]
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000 and cv2.boundingRect(cnt)[2] < plate.shape[1]/5:
                #cv2.drawContours(plate, [cnt], 0, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(cnt)
                #cv2.rectangle(plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # print(cv2.contourArea(cnt))
                # cv2.imshow("img", plate)
                # cv2.waitKey(0)
                letter.append([x,y,x+w,y+h])
        letter=return_biggest(letter)
        # for let in letter:
        #     cv2.rectangle(plate, (let[0], let[1]), (let[2], let[3]), (0, 255, 0), 2)
        #     cv2.imshow("img", plate[let[1]:let[3],let[0]:let[2]])
        #     cv2.waitKey(0)
        letter=compare_images(letter, dictionary,plate)




        # cv2.imshow("img", plate)
        # cv2.waitKey(0)
def compare_images(letters, dictionary,plate):
    tablica = []
    for letter in letters:
        letter_image = plate[letter[1]:letter[3], letter[0]:letter[2]]
        letter_gray = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)

        letter_gray = cv2.GaussianBlur(letter_gray, (5, 5), 0)

        ret, letter_gray = cv2.threshold(letter_gray, 95, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        letter_gray = cv2.morphologyEx(letter_gray, cv2.MORPH_OPEN, kernel, iterations=2)
        best_match = None
        best_match_score = float('inf')

        for image in dictionary:
            #resize image to letter_gray size
            image = cv2.resize(image, (letter_gray.shape[1], letter_gray.shape[0]))

            score = cv2.matchShapes(image, letter_gray, cv2.CONTOURS_MATCH_I1, 0)

            if score < best_match_score:
                best_match = image
                best_match_score = score
        tablica.append(best_match)
        if best_match is not None:

            cv2.imshow("img", best_match)
            cv2.imshow("img2", letter_image)
            cv2.waitKey(0)

    return tablica






def main():
    import_images()
    import_letters()
    images = resize_images()
    plates=find_contours(images)
    find_letters(plates)



if __name__ == '__main__':
    main()