import cv2
import numpy as np
import os
import math
import json

#import images from images folder
path = "/home/opszalek/PycharmProjects/SW_projekt/images"
path = os.path.abspath(path)
images = []
tablice = []
def import_images():
    for filename in os.listdir(path):
       # print(filename)
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
def resize_images():
    img_resized = []
    scale_percent = 25  # percent of original size
    for img in images:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_resized.append(img)
    return img_resized

#find contours on images
def area_of_rectqngle(points):
    #np.array([left[0],right[0],left[1],right[1]], dtype=np.float32)
    print(points)
    return abs(points[2][0][0] - points[0][0][0]) * abs(points[2][0][1] - points[0][0][1])
def find_contours_test(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_ = [cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)) == 4]
    contours_ = [cnt for cnt in contours_ if cv2.contourArea(cnt) > 30000]
    img_wide = image.shape[1]
    tablice = []
    for cnt in contours_:

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            points = zwracanie_rogow(approx)
            if points[1][0][0]-points [0][0][0]> 0.2* img_wide:
                if 3 < ((points[1][0][0]-points [0][0][0]) / (points[2][0][1]-points [0][0][1])) <6 or \
                        3 < ((points[1][0][0]-points [0][0][0]) / (points[3][0][1]-points [1][0][1])) <6:
                    tablice.append(approx)
    tablica = None
    for temmp_tablica in tablice:
        cv2.drawContours(image, [temmp_tablica], 0, (0, 255, 255), 2)
        if tablica is not None:
            if area_of_rectqngle(temmp_tablica) < area_of_rectqngle(tablica):
                tablica = temmp_tablica
                continue
        tablica = temmp_tablica
    if tablica is not None:
        cv2.drawContours(image, [tablica], 0, (0, 0, 255), 2)
        points = zwracanie_rogow(tablica)
        width = 1200
        height = 300
        dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
        # src_points = np.array([point[0] for point in cnt], dtype=np.float32)
        M = cv2.getPerspectiveTransform(points, dst_points)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))
        cv2.imshow("Warped Image", warped)
       # cv2.waitKey(0)
        # # utwórz maskę i wypełnij kontur białą barwą
        # mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # cv2.fillPoly(mask, [tablica], (255, 255, 255))
        #
        # # zastosuj maskę do oryginalnego obrazu
        # result = cv2.bitwise_and(image, image, mask=mask)
        #
        # # wyświetl wynik
        # cv2.imshow('result', result)
        # cv2.waitKey(0)  #######################genialne wycinania


    #sort contours by area
    # contours_ = sorted(contours_, key=cv2.contourArea, reverse=True)
    # contours_= contours_[-1]

    # for cnt in contours_:
    #     # Approximate the contour to a polygon with 4 vertices
    #     approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    #
    #     # Draw a red dot at each vertex
    #     for vertex in approx:
    #         x, y = vertex[0]
    #         cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    # if len(contours_) == 1:
    #     cnt = contours_[0]
    #     approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    #     points=zwracanie_rogow(approx)
    #
    #     width = 1200
    #     height = 300
    #
    #     dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    #     #src_points = np.array([point[0] for point in cnt], dtype=np.float32)
    #     M = cv2.getPerspectiveTransform(points, dst_points)
    #     warped = cv2.warpPerspective(image, M, (int(width), int(height)))
    #     cv2.imshow("Warped Image", warped)
    #     cv2.waitKey(0)
    #     # utwórz maskę i wypełnij kontur białą barwą
    #     mask = np.zeros(image.shape[:2], dtype=np.uint8)
    #     cv2.fillPoly(mask, [cnt], (255, 255, 255))
    #
    #     # zastosuj maskę do oryginalnego obrazu
    #     result = cv2.bitwise_and(image, image, mask=mask)
    #
    #     # wyświetl wynik
    #     cv2.imshow('result', result)
    #     cv2.waitKey(0)#######################genialne wycinania
    #
    #     # tablice.append(image)
    #     for cnt in contours_:
    #         mask = np.zeros_like(image)
    #         cv2.drawContours(mask, cnt, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    #         result = cv2.bitwise_and(image, mask)
    #         cv2.imshow("img", result)
    #         cv2.waitKey(0)

    #
    # for contour in contours_:
    #     # policz dlugosci bokow
    #     sides = []
    #     for i in range(4):
    #         side = np.sqrt((contour[(i + 1) % 4][0][0] - contour[i][0][0]) ** 2 + (
    #                     contour[(i + 1) % 4][0][1] - contour[i][0][1]) ** 2)
    #         sides.append(side)
    #     # policz ratio bokow
    #     ratio = max(sides) / min(sides)
    #     # sprawdz czy ratio jest w zakresie 2.1-3.1 (w przyblizeniu dla polskich tablic rejestracyjnych)
    #     if 2 <= ratio <= 6:
    #         cv2.drawContours(image, [contour], -1, (0, 255, 255), 3)
    #         cv2.imshow("imag", image)
    #
    # cv2.drawContours(image, contours, -1, (0, 255, 255), 3)
    #cv2.drawContours(image, contours_, -1, (0, 255, 0), 3)
    cv2.imshow("img", image)
    cv2.waitKey(0)
def find_contours(image):
    # konwersja obrazu na skale szarosci
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 2)


    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

    contours_ = [cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)) == 4]
    contours_ = [cnt for cnt in contours_ if cv2.contourArea(cnt) > 30000]

    #cv2.drawContours(image, contours, -1, (0, 255, 255), 3)
    cv2.drawContours(image, contours_, -1, (0, 255, 0), 3)
    cv2.imshow("img", image)
    cv2.waitKey(0)
def zwracanie_rogow(contour):
    # M = cv2.moments(contour)
    # cx = int(M["m10"] / M["m00"])
    # cy = int(M["m01"] / M["m00"])
    # corners = []
    #
    # for point in contour:
    #
    #     distance = ((point[0][0] - cx) ** 2 + (point[0][1] - cy) ** 2) ** 0.5
    #     corners.append((point, distance))
    # corners.sort(key=lambda x: x[1], reverse=True)
    # sorted_corners = [corner[0] for corner in corners[:4]]
    # sorted_corners.sort(key=lambda x: x[0][0])
    # if sorted_corners[0][0][1] > sorted_corners[1][0][1]:
    #     sorted_corners[0], sorted_corners[1] = sorted_corners[1], sorted_corners[0]
    # sorted_corners.sort(key=lambda x: x[0][1])
    # if sorted_corners[2][0][1] > sorted_corners[3][0][1]:
    #     sorted_corners[2], sorted_corners[3] = sorted_corners[3], sorted_corners[2]
    # return np.array(sorted_corners, dtype=np.float32)
    x=sorted(contour, key=lambda x: x[0][0])
    left=x[:2]
    right=x[2:]
    left=sorted(left, key=lambda x: x[0][1])
    right=sorted(right, key=lambda x: x[0][1])

    return np.array([left[0],right[0],left[1],right[1]], dtype=np.float32)


    print(x)
def canny_edge_detection(img):

    # Wczytaj obraz i przekonwertuj na odcienie szarości

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 2)

    # Zastosuj filtr Gaussa, aby zredukować szum
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Wykonaj detekcję krawędzi z użyciem algorytmu Canny
    edges = cv2.Canny(blur, 40, 255)

    # Wykonaj morfologiczną operację otwarcia, aby usunąć mniejsze elementy i zamknąć luki między konturami
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Znajdź kontury i wybierz tylko te, które pasują do wymiarów tablic rejestracyjnych (długość, szerokość, proporcje)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
    #contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
    # countours that are rectangles
    # contours = [cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)) == 4]
    # check that long straight lines are in contours
    # contours_ = [cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)) == 4]
    # contours_ = [cnt for cnt in contours_ if cv2.contourArea(cnt) > 30000]
    for cnt in contours:
        rect =cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        x ,y ,w ,h = cv2.boundingRect(rect)
        aspect_ratio = float(w)/h
        if aspect_ratio > 2.5 and aspect_ratio < 5.5:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
   # contours_ = [cnt for cnt in contours_ if aspect_ratio(cnt) > 2.5 and aspect_ratio(cnt) < 5.5]

    # Wyświetl obraz z zaznaczonymi tablicami rejestracyjnymi
    cv2.drawContours(img, contours, -1, (0, 255, 255), 3)
    #cv2.drawContours(img, contours_, -1, (0, 255, 0), 3)
    cv2.imshow("img", img)
    cv2.waitKey(0)
def check_accuracy():
    file_path = 'corect_results.json'
    file = open(file_path, 'r')
    data = json.load(file)
    file_path = 'tablice.json'
    file = open(file_path, 'r')
    results = json.load(file)
    correct = 0
    all_letters = 0
    for key, value in data.items():
        print(1)
        i = 0
        for char in value:
            if len(results[key])>i:
                if char == results[key][i]:
                    correct += 1
            all_letters += 1
            i+=1
    print(correct/all_letters)
def main():
    # import_images()
    # images = resize_images()
    # for img in images:
    #     #find_contours(img)
    #     find_contours_test(img)
    #     #canny_edge_detection(img)
    check_accuracy()



if __name__ == '__main__':
    main()