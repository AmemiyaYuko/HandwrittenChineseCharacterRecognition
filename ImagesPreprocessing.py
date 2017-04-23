import os
import cv2
import numpy as np


def read_gnt(file):
    while True:
        header = np.fromfile(file, dtype="uint8", count=10)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tag_code = header[4] + (header[5] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        image = np.fromfile(file, dtype='uint8', count=width * height).reshape((height, width))
        yield image, tag_code


def file_list(gnt_dir):
    return [os.path.join(gnt_dir, file_name) for file_name in os.listdir(gnt_dir)]


def gen_id():
    char_id = 0
    while True:
        char_id += 1
        yield char_id


files = [open(file_path, "r") for file_path in file_list("C:\\Users\\AmemiyaYuko\\Downloads\\isolated_data")]
i = 0;
for file in files:
    for (image, tag_code) in (read_gnt(file)):
        i += 1
        cv2.imwrite(str(tag_code) + "_" + str(i) + ".jpg", img=image)
