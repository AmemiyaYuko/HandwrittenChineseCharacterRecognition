import os
import cv2
import numpy as np
import argparse
import random
import sys
import pickle
import struct


def read_gnt(file):
    while True:
        header = np.fromfile(file, dtype="uint8", count=10)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tag_code = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        image = np.fromfile(file, dtype='uint8', count=width * height).reshape((height, width))
        yield image, tag_code


def file_list(gnt_dir):
    return [os.path.join(gnt_dir, file_name) for file_name in os.listdir(gnt_dir)]


def getDict(dir,dict_name):
    files = file_list(dir)
    char_set = set()
    for file in files:
        f = open(file, 'r')
        for _, tag_code in read_gnt(f):
            uni = struct.pack('>H', tag_code).decode('gb2312')
            char_set.add(uni)
    char_list = list(char_set)
    cdict = dict(zip(sorted(char_list), range(len(char_list))))
    pickle.dump(cdict, open(dict_name, 'wb'))
    return cdict


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnt-dir", type=str, help="Absolute path of gnt dateset",
                        default=os.path.abspath("C:/Datasets/HWDB/gnt"))
    parser.add_argument("--output-dir", type=str, help="Absolute path for storing pictures",
                        default=os.path.abspath("C:/Datasets/HWDB/"))
    parser.add_argument("--dict", type=str, default="char_dict")
    return parser.parse_args()


def extractor(in_dir, out_dir, char_dict):
    i = 0
    for file_name in file_list(in_dir):
        f = open(file_name, "r")
        for image, tag_code in read_gnt(f):
            i += 1
            tag_code_uni = struct.pack('>H', tag_code).decode('gb2312')
            tag_str = out_dir + "/" + '%0.5d' % char_dict[tag_code_uni]
            if os.path.isdir(tag_str):
                pass
            else:
                os.mkdir(tag_str)
            cv2.imwrite(tag_str + '/' + str(i) + ".png", image)
        f.close()
    return i

def main(args):
    trn_path = os.path.join(args.gnt_dir, "trn")
    tst_path = os.path.join(args.gnt_dir, "tst")
    trn_out_path = os.path.join(args.output_dir, "trn")
    tst_out_path = os.path.join(args.output_dir, "tst")
    char_dict = getDict(trn_path,args.dict)
    print("Total " + str(len(char_dict)) + " characters.")
    trn_imgs = extractor(trn_path, trn_out_path, char_dict)
    print("Total " + str(trn_imgs) + " images in training set.")
    tst_imgs = extractor(tst_path, tst_out_path, char_dict)
    print("Total " + str(tst_imgs) + " images in test set.")


if __name__ == '__main__':
    main(arg_parser())
