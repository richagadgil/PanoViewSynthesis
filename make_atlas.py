from imageio import imread, imwrite
import glob
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input',required=True,help='input directory')
parser.add_argument('--output',required=True,help='output directory')
args = parser.parse_args()

H = 1024
W = 2048
rows = 8
cols = 4
#rgb = np.zeros((H*rows,W*cols,3),dtype='uint8')
#alpha = np.zeros((H*rows,W*cols),dtype='uint8')
rgba = np.zeros((H*rows,W*cols,4),dtype='uint8')
n = 0
for r in range(rows):
    for c in range(cols):
        image_in = imread(os.path.join(args.input,'layer_%d.png'%n))
        #rgb[H*r:H*(r+1),W*c:W*(c+1)] = image_in[...,:3]
        #alpha[H*r:H*(r+1),W*c:W*(c+1)] = image_in[...,3]
        myr = (rows-1)-r
        rgba[H*myr:H*(myr+1),W*c:W*(c+1)] = image_in[:,::-1] # flip horizontally for rendering on backside of cylinder
        n = n + 1
#rgba = rgba[::16,::16]
print(f'writing to {args.output} with size {rgba.shape[0]} x {rgba.shape[1]}')
imwrite(os.path.join(args.output,'atlas.png'),rgba)
#imwrite(os.path.join(args.output,'rgb.png'),rgb)
#imwrite(os.path.join(args.output,'alpha.png'),alpha)

