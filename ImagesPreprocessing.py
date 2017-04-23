import os
import cv2
import numpy as np
import argparse


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


def arg_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=str,help="Absolute path of gnt dateset",default=os.path.abspath("E:\HWDB1.1trn_gnt"))
    parser.add_argument("--output-dir",type=str,help="Absolute path for storing pictures",default=os.path.abspath("E:\HWDB_Dataset"))
    return parser.parse_args()


def main(args):
    files = [open(file_path, "r") for file_path in
             file_list(args.data_dir)]
    i = 0
    for file in files:
        for (image, tag_code) in (read_gnt(file)):
            i += 1
            if os.path.isdir(os.path.join(args.output_dir, str(tag_code))):
                pass
            else:
                os.mkdir(os.path.join(args.output_dir, str(tag_code)))
            cv2.imwrite(os.path.join(args.output_dir, str(tag_code)) + "\\" + str(i) + ".jpg", img=image)


if __name__=='__main__':
    main(arg_parser())

