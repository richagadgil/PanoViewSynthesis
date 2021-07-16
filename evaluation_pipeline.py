from OpenGL.GL import *
from OpenGL.GLUT import *
import time

import numpy as np
import matplotlib.pyplot as plt
import os
import re

from rendertools import *

from imageio import imwrite

import json
import tensorflow as tf

import argparse


"""

pd = tf.math.log(square_pd)
p_depths = tf.where(~tf.math.is_finite(pd), tf.zeros_like(pd), pd)

ad = tf.math.log(tf.linalg.inv(square_ad))
a_depths = tf.where(~tf.math.is_finite(ad), tf.zeros_like(ad), ad)

s = tf.math.exp(tf.reduce_mean(p_depths + a_depths))
sigmas.append(s)



"""

def compute_sigma(pred_disp,points):
    points = tf.math.log(points)
    pred_disp = tf.math.log(pred_disp)
    points_depths = tf.where(~tf.math.is_finite(points), tf.zeros_like(points), points)
    actual_depths = tf.where(~tf.math.is_finite(pred_disp), tf.zeros_like(pred_disp), pred_disp)
    sigma = tf.math.exp(tf.reduce_mean(points_depths+actual_depths))
    return sigma


if __name__ == '__main__':

    width = 1280
    height = 1280
    fovy = 90

    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_3_2_CORE_PROFILE)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow('window')
    glutHideWindow(window)

    min_depth = 1.
    max_depth = 100.
    depths = 1./np.linspace(1./max_depth, 1./min_depth,
                            32, endpoint=True)

    # regexr = '\/(\w+)\/(\w+).png'
    # folder, image_name = re.findall(regexr, args.test_path)[0]
    # target_json_path = os.path.join(args.test_path.split(
    #     folder)[0], f'{folder}_json', image_name.split('_frame')[0] + '.json')

    # with open(target_json_path) as f:
    #     target_json = json.load(f)

    # Scale Depths 

    locations = [os.path.join("cube_dir", folder) for folder in os.listdir("cube_dir") if len(folder.split(".")) < 2 ]
    for location in locations[0:1]:
        print(f"Working on location {location}...")
        predicted_depth = imread(os.path.join(location, 'predicted_depth.png')).astype('float32')[:,:,0]
        actual_depth = imread(os.path.join(location, 'actual_depth.png')).astype('float32')
        sigma = compute_sigma(predicted_depth, actual_depth) 
        sigma_depths = [i*sigma.numpy() for i in depths]

        test = {"sigma_depths": sigma_depths, "depths": depths}

        # load json
        json_folder_path = os.path.join(location, "00", "cube_images_json")
        img_dir_path = os.path.join(location, "00", "cube_images")

        view_filenames = [file.split(".")[0] for file in os.listdir(json_folder_path)]
        #print(view_filenames)

        MSE = {}

        for filename in view_filenames:
            print(f"Working on frame {filename}...")
            with open(os.path.join(json_folder_path, f"{filename}.json")) as f:
                target_json = json.load(f)
            target_image = imread(os.path.join(img_dir_path, f"{filename}_frame.png")).astype('float32')
            MSE[filename] = {}
            
            for name, d in test.items():
                meshes = [MyCylinder(bottom=-1*depth, top=1*depth, radius=depth,
                                    texturepath=os.path.join(location, 'layers/layer_%d.png' % i)) for i, depth in enumerate(d)]

                renderer = Renderer(meshes, width=width, height=height,
                                    offscreen=True)

                eye = np.array([target_json["eye"]]) * 0.5
                target = np.array(target_json["target"])
                up = np.array(target_json["up"])

                view_matrix = lookAt(eye, target, up)
                proj_matrix = perspective(fovy, width/height, 0.1, 1000.0)
                mvp_matrix = proj_matrix@view_matrix

                produced_image = renderer.render(mvp_matrix)
                # where to write the produced image?
                imwrite(os.path.join(img_dir_path, f"{name}_{filename}_produced.png"), produced_image)
                # Calculate MSE HERE
                mse = ((produced_image - target_image)**2).mean(axis=None)

                MSE[filename][name] = str(mse)

        with open(os.path.join(location,'mse.json'), 'w') as json_file:
            json.dump(MSE, json_file)