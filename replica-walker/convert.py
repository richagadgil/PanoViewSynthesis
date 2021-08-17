import numpy as np
import os
import sys
from imageio import imread, imsave
from tqdm import trange
from skimage.transform import warp
import tensorflow as tf
from tensorflow_addons.image import interpolate_bilinear


class Converter:
    def __init__(self, output_height, output_width):
        self.output_height = output_height
        self.output_width = output_width

        # make arrays of equally spaced angle and height values
        theta_range = np.linspace(-np.pi, np.pi,
                                  num=args.output_width, endpoint=False)
        height_range = np.linspace(
            args.bottom, args.top, num=args.output_height, endpoint=True)

        height, theta = np.meshgrid(height_range, theta_range, indexing='ij')

        X = np.sin(theta)
        Y = height
        Z = np.cos(theta)

        # get vertical angle
        phi = np.arcsin(Y/np.sqrt(X*X+Y*Y+Z*Z))

        # project to spherical panorama image (normalized coordinates)
        self.lutx = (0.5*theta/np.pi+0.5)
        self.luty = (phi/np.pi+0.5)

    # @tf.function
    def convert(self, im):
        input_height, input_width = im.shape[:2]
        coords = np.stack(
            [input_width*self.lutx, input_height*self.luty], axis=-1)
        coords_flat = np.reshape(coords, (1, -1, 2))
        pano_flat = interpolate_bilinear(
            tf.cast(im[None, ...], 'float32'), coords_flat, indexing='xy')
        pano = tf.reshape(
            pano_flat[0], (self.output_height, self.output_width) + im.shape[2:])
        #coords = np.stack([self.luty*input_height,self.lutx*input_width],axis=0)
        # print(coords.shape)
        #pano = warp(im,coords,preserve_range=True).astype('uint8')
        return pano


if __name__ == '__main__':
    from configargparse import ArgumentParser
    import glob
    import os

    parser = ArgumentParser(
        description='Convert spherical/equirectangular panoramas to cylindrical format'
    )

    parser.add_argument('--config',
                        is_config_file=True,
                        help='path to configuration file')
    parser.add_argument('--input',
                        required=True,
                        help='image or directory containing spherical input images')
    parser.add_argument('--output', '-o',
                        required=True,
                        help='directory for cylindrical output')
    parser.add_argument('--output_width',
                        default=2048,
                        type=int,
                        help='width of output (cylindrical) panorama image')
    parser.add_argument('--output_height',
                        default=1024,
                        type=int,
                        help='height of output (cylindrical) panorama image')
    parser.add_argument('--bottom',
                        default=-1,
                        type=float,
                        help='bottom height value of cylinder')
    parser.add_argument('--top',
                        default=1,
                        type=float,
                        help='top height value of cylinder')

    args = parser.parse_args()

    # verify/create the output directory

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    # unwarp the images
    if not os.path.isdir(args.input):
        # unwarp a single image
        infiles = [args.input]
    else:
        # unwarp all png files in the given directory
        infiles = sorted(glob.glob(os.path.join(args.input, '*.png')))

    converter = Converter(output_height=args.output_height,
                          output_width=args.output_width)

    # unwarp all files
    for infile in infiles:
        # set the outfile
        filename = os.path.basename(infile)
        outfile = os.path.join(args.output, filename)

        # run the unwarp
        im = imread(infile)
        res = converter.convert(im)
        #res = res.numpy().astype('uint8')

        # save
        imsave(outfile, res)
