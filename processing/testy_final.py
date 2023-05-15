import cv2
import numpy as np
import os
import math

#import images from images folder
path = "/home/opszalek/PycharmProjects/SW_projekt/images"
path = os.path.abspath(path)
images = []
tablice = []
def import_images():
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
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
def check_overlap(letter, rect2):
    overlap = False
    if letter is not None:
        #rect2=[x, y, x + w, y + h]
        for rect1 in letter:
            if rect1[2] > rect2[0] and rect1[0] < rect2[2] and rect1[3] > rect2[1] and rect1[1] < rect2[3]:
                overlap = True
                break
    return overlap
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
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 2)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #use canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        #close gaps between edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)


        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        letter=[]
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000 and cv2.boundingRect(cnt)[2] < plate.shape[1]/5:
                cv2.drawContours(plate, [cnt], 0, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # print(cv2.contourArea(cnt))
                # cv2.imshow("img", plate)
                # cv2.waitKey(0)
                if check_overlap(letter, [x,y,x+w,y+h]) == False:
                    letter.append([x,y,x+w,y+h])
        # for let in letter:
        #     cv2.drawContours(plate, [let], 0, (0, 255, 0), 3)


        cv2.imshow("img", plate)
        cv2.waitKey(0)



def main():
    import_images()
    images = resize_images()
    plates=find_contours(images)
    find_letters(plates)



if __name__ == '__main__':
    main()