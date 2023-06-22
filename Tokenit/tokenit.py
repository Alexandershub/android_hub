import cv2
import numpy as np
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.core.window import Window
from kivy.uix.image import Image


class ImageProcessingApp(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_path = ""  # Path to the captured image
        self.marked_area = []

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.marked_area = [touch.pos]

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            self.marked_area.append(touch.pos)
            self.canvas.clear()
            with self.canvas:
                Color(1, 0, 0)
                Line(points=self.marked_area, width=2)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            self.marked_area.append(touch.pos)
            self.crop_object()

    def crop_object(self):
        image = cv2.imread(self.image_path)
        x_values = [int(point[0]) for point in self.marked_area]
        y_values = [int(point[1]) for point in self.marked_area]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # Create a mask to indicate the object region
        mask = np.zeros(image.shape[:2], np.uint8)
        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Generate a binary mask, separating the object from the background
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply the mask to the original image to obtain the cropped object
        cropped_image = image * mask[:, :, np.newaxis]

        cv2.imshow("Cropped Object", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the cropped image
        cropped_image_path = "path/to/save/cropped_image.jpg"
        cv2.imwrite(cropped_image_path, cropped_image)
        print("Cropped image saved:", cropped_image_path)


class MyApp(App):
    def build(self):
        image_processing_app = ImageProcessingApp()
        return image_processing_app


if __name__ == '__main__':
    Window.size = (500, 500)  # Set the initial window size
    MyApp().run()

