import cv2
import freetype
import numpy as np
import os
class Czcionka:
    def __init__(self):
        self.font_path = 'processing/letters/font.ttf'
        self.font_size = 74
        self.font = freetype.Face(self.font_path)
        self.font.set_char_size(self.font_size * 64)
        self.image_width = 60
        self.image_height = 60
        self.characters = 'ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789'
        self.output_directory = 'processing/letters'
    def generate_letters(self):
        for char in self.characters:
            self.font.load_char(char)
            bitmap = self.font.glyph.bitmap
            width, height = bitmap.width, bitmap.rows
            image = np.ones((self.image_height, self.image_width), dtype=np.uint8) * 255
            left = (self.image_width - width) // 2
            top = (self.image_height - height) // 2
            for x in range(width):
                for y in range(height):
                    image[top + y, left + x] = 255 - bitmap.buffer[y * width + x]
            output_path = os.path.join(self.output_directory, char + '.png')
            cv2.imwrite(output_path, image)

