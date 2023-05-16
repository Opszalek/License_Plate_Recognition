import cv2
import freetype
import numpy as np


font_path = 'font.ttf'
font_size = 74
font = freetype.Face(font_path)
font.set_char_size(font_size * 64)

image_width = 60
image_height = 60

characters = 'ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789'

for char in characters:
    font.load_char(char)
    bitmap = font.glyph.bitmap
    width, height = bitmap.width, bitmap.rows
    image = np.ones((image_height, image_width), dtype=np.uint8) * 255
    left = (image_width - width) // 2
    top = (image_height - height) // 2
    for x in range(width):
        for y in range(height):
            image[top + y, left + x] = 255 - bitmap.buffer[y * width + x]
    cv2.imwrite(char + '.png', image)
