import cv2
import numpy as np
import os

#import images from images folder
path = "/home/opszalek/PycharmProjects/SW_projekt/images"
path = os.path.abspath(path)
images = []
def import_images():
    for filename in os.listdir(path):
       # print(filename)
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
#show images
def show_images():
    for img in images:
        cv2.imshow("img", img)
        cv2.waitKey(0)
def resize_images():
    img_resized = []
    scale_percent = 30  # percent of original size
    for img in images:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_resized.append(img)
    return img_resized

#find rectangles on images
def find_max_white_rect(image):
    # konwersja obrazu na skale szarosci
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # progowanie obrazu
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # wyznaczanie konturów
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # lista prostokątów
    rects = []

    # iteracja po konturach
    for cnt in contours:
        # aproksymacja konturu prostokątem
        rect = cv2.minAreaRect(cnt)

        # wyliczenie współrzędnych wierzchołków prostokąta
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # obliczenie sumy wartości pikseli w obszarze prostokąta
        sum_white = np.sum(thresh[box[0][1]:box[2][1], box[0][0]:box[2][0]])

        # dodanie prostokąta do listy razem z jego sumą
        rects.append((box, sum_white))

    # wybranie prostokąta z największą sumą wartości pikseli
    max_rect = max(rects, key=lambda x: x[1])

    # narysowanie wybranego prostokąta na obrazie
    cv2.drawContours(image, [max_rect[0]], 0, (0, 255, 0), 2)

    return image
def find_rectangles(images):
    rectangles=[]
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        ret, thresh = cv2.threshold(gray, 80, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)
            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            if cv2.contourArea(c) > 10000:
                # calculate the minimum area rectangle for the contour
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # draw the rectangle on the image
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        #img=find_max_white_rect(img)

        cv2.imshow("Image", img)
        cv2.waitKey(0)

def find_rectangles2(images):
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 170, 255, 1)
        contours, h = cv2.findContours(thresh, 1, 2)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                #contour area bigger than 10000
                if cv2.contourArea(cnt) > 10000:

                    cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def find_tablice(images):
    for img in images:
        # Wczytaj obraz i przekonwertuj na odcienie szarości

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Zastosuj filtr Gaussa, aby zredukować szum
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Wykonaj detekcję krawędzi z użyciem algorytmu Canny
        edges = cv2.Canny(blur, 50, 150)

        # Wykonaj morfologiczną operację otwarcia, aby usunąć mniejsze elementy i zamknąć luki między konturami
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Znajdź kontury i wybierz tylko te, które pasują do wymiarów tablic rejestracyjnych (długość, szerokość, proporcje)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if w > 80 and h > 10 and aspect_ratio > 2.5 and aspect_ratio < 5:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Wyświetl obraz z zaznaczonymi tablicami rejestracyjnymi
        cv2.imshow('Result', img)
        cv2.waitKey(0)

def find_niebieski(images):
    for img in images:
        # Przefiltrowanie obrazu z wykorzystaniem przestrzeni kolorów HSV, aby zidentyfikować kolory biały i niebieski
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 25, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Kombinacja maski białej i niebieskiej
        mask = cv2.bitwise_or(mask_white, mask_blue)

        # Filtrowanie obrazu z wykorzystaniem maski
        filtered = cv2.bitwise_and(img, img, mask=mask)

        # Konwersja obrazu na odcień szarości i progowanie
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Znalezienie konturów w obrazie
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Przefiltrowanie konturów, aby znaleźć tylko te, które przypominają tablice rejestracyjne
        for c in contours:
            # Obliczenie pola powierzchni konturu
            area = cv2.contourArea(c)

            # Obliczenie obwodu konturu
            perimeter = cv2.arcLength(c, True)

            # Przybliżenie konturu wielokątem, aby sprawdzić, czy ma 4 wierzchołki
            approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)

            # Sprawdzenie, czy kontur ma 4 wierzchołki i odpowiednio dużą powierzchnię
            if len(approx) == 4 and area > 5000:
                # Zaznaczenie konturu na oryginalnym obrazie
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

        # Wyświetlenie wyniku
        cv2.imshow('Tablice', img)
        cv2.waitKey(0)
def find_plates(images):
    plates = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        ret, thresh = cv2.threshold(gray, 80, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for c in contours:
            if cv2.contourArea(c) > 8000:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                rectangles.append(box)

        for rect in rectangles:
            x, y, w, h = cv2.boundingRect(rect)
            if x >= 0 and y >= 0:
                plate = img[y:y+h, x:x+w]
                #plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                #plate_gray = cv2.medianBlur(plate_gray, 5)
                #ret, plate_thresh = cv2.threshold(plate_gray, 110, 255, cv2.THRESH_BINARY)
                blue_mask = cv2.inRange(plate, (100, 0, 0), (255, 80, 80))
                blue_count = cv2.countNonZero(blue_mask)
                if blue_count > 1000:
                    plates.append(plate)
    return plates

def biale_niebieskie(img):

    plates = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        ret, thresh = cv2.threshold(gray, 80, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for c in contours:
            if cv2.contourArea(c) > 1000:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                rectangles.append(box)

        for rect in rectangles:
            x, y, w, h = cv2.boundingRect(rect)
            if x >= 0 and y >= 0:
                plate = img[y:y + h, x:x + w]
                blue_mask = cv2.inRange(plate, (100, 0, 0), (255, 80, 80))
                white_mask = cv2.inRange(plate, (220, 220, 220), (255, 255, 255))
                blue_white_mask = cv2.bitwise_and(blue_mask, white_mask)
                blue_white_count = cv2.countNonZero(blue_white_mask)
                if blue_white_count > 60:
                    plates.append(plate)
    return plates
#find license plates on images using contours

def main():
    import_images()
    #show_images()
    images = resize_images()
    find_rectangles2(images)
    #find_rectangles(images)
    #find_tablice(images)
    #find_niebieski(images)

    #plates=biale_niebieskie(images)
    #plates = find_plates(images)
    #
    # for plate in plates:
    #     cv2.imshow('Tablice', plate)
    #     cv2.waitKey(0)



if __name__ == '__main__':
    main()