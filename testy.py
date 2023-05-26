import cv2
import numpy as np
import os
i=0
#import images from images folder
path = "/home/opszalek/PycharmProjects/SW_projekt/images1"
path = os.path.abspath(path)
images = []
def import_images():
    for filename in os.listdir(path):
       # print(filename)
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
def resize_images(img):
    scale_percent = 1500 / img.shape[1] *100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img_resized
def resize_images1(img):
    scale_percent = 33
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
def return_biggest(rect_list):
    rect_list_cleaned = []
    for rect1 in rect_list:
        is_max_rect = True
        for rect2 in rect_list:
            if rect1 != rect2 and (is_rect_inside_rect(rect2, rect1) or is_rect_inside_rect(rect1, rect2)):
                if rect_area(rect1) < rect_area(rect2):
                    is_max_rect = False
                    break
        if is_max_rect and rect_area(rect1) > 9000:
            print(rect_area(rect1))
            rect_list_cleaned.append(rect1)

    return rect_list_cleaned


def is_rect_inside_rect(rect1, rect2):
    return (rect2[0] <= rect1[0] <= rect1[2] <= rect2[2]) and (rect2[1] <= rect1[1] <= rect1[3] <= rect2[3])
def is_rect_intersect_rect(rect1, rect2):
    return not (rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or rect1[3] <= rect2[1] or rect1[1] >= rect2[3])
def rect_area(rect):
    # print((rect[2] - rect[0]) * (rect[3] - rect[1]))
    return (rect[2] - rect[0]) * (rect[3] - rect[1])
def find_contours(image):
    color= cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    thresh = cv2.adaptiveThreshold(image, 214, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_wide = image.shape[1]
    tablica = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04* cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 70000:
            points = sort_corners(approx)
            if points[1][0][0]-points [0][0][0]> 0.3 * img_wide:
                if 3.5 < ((points[1][0][0]-points [0][0][0]) / (points[2][0][1]-points [0][0][1])) <6 or \
                        3.5 < ((points[1][0][0]-points [0][0][0]) / (points[3][0][1]-points [1][0][1])) <6:
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
        plates=warped
    else:
        plates=image
    return plates
def find_letters(image, x1, x2, x3, x4):
    image1 = image.copy()
    edges = cv2.Canny(image, x1, x2)
    kernel = np.ones((x3, x3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=x4)
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

    # for rect1 in rectangles:
    #     cv2.rectangle(image1, (rect1[0], rect1[1]), (rect1[2], rect1[3]), (0, 255, 0), 2)
    #     print(rect1)
    #     cv2.imshow("image1", image1)
    #     cv2.waitKey(0)
    rectangles=return_biggest(rectangles)
    # for rect1 in rectangles:
    #     cv2.rectangle(image1, (rect1[0], rect1[1]), (rect1[2], rect1[3]), (0, 255, 0), 2)
    #     print(rect1)
    #     cv2.imshow("image1", image1)
    #     cv2.waitKey(0)
    letters=[]


    rectangles.sort(key=lambda x: x[0])

    for rect in rectangles:
        # if rect[3]==300:
        #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        letters.append(image[rect[1]:rect[3],rect[0]:rect[2]])
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print(111111111111111111)
    return letters
def empty_callback(value):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('win', 'image', 3, 255, empty_callback)
cv2.createTrackbar('win1', 'image', 3, 255, empty_callback)
cv2.createTrackbar('win2', 'image', 3, 20, empty_callback)
cv2.createTrackbar('win3', 'image', 1,20, empty_callback)


def main():
    global i
    import_images()
    while True:
        okno = cv2.getTrackbarPos('win', 'image')
        okno1 = cv2.getTrackbarPos('win1', 'image')
        okno2 = cv2.getTrackbarPos('win2', 'image')
        okno3 = cv2.getTrackbarPos('win3', 'image')
        i=0
        for image in images:
            image=resize_images(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plate=find_contours(image)
            letters=find_letters(plate, 100, 200, okno2, okno3)
            image=resize_images1(image)
        #     cv2.imshow("img"+str(i), plate)
        #     i=i+1
        # cv2.waitKey(0)



if __name__ == '__main__':
    main()