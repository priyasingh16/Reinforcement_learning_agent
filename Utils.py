import numpy as np


class Utils:
    @staticmethod
    def rgb2gray(img):
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def pre_process(img):
        return np.mean(img[::1, ::1], axis=2).astype(np.uint8)
