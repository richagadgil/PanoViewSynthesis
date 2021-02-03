from OpenGL.GL import *
from OpenGL.GLUT import *
import time

#import pygame as pg
#from pygame.locals import *

import numpy as np
import matplotlib.pyplot as plt
import os

from rendertools import *

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--mode',default='cylindrical')
    args = parser.parse_args()
    
    width = 640
    height = 480
    fovy = 90

    glutInit()
    glutInitDisplayMode(GLUT_RGBA|GLUT_3_2_CORE_PROFILE)
    glutInitWindowSize(width,height)
    glutInitWindowPosition(0,0)
    window = glutCreateWindow('window')
    #glutHideWindow()
    #pg.init()
    #pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
    #pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 2)
    #pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                #pg.GL_CONTEXT_PROFILE_CORE)
    #pg.display.set_mode((width,height), DOUBLEBUF|OPENGL)

    
    min_depth = 1.
    max_depth = 100.
    depths = 1./np.linspace(1./max_depth,1./min_depth,32,endpoint=True)
     
    if args.mode == 'cylindrical':
        meshes = [Cylinder(bottom=-1*depth,top=1*depth,radius=depth,texturepath=os.path.join(args.path,'courtyard_%d.png'%i)) for i,depth in enumerate(depths)]
    else:
        meshes = [Plane(depth=depth,texturepath=os.path.join(args.path,'image1_%d.png'%i)) for i,depth in enumerate(depths)]
    #meshes = meshes[0:1]
    
    renderer = Renderer(meshes,width=width,height=height)

    time0 = time.time()
    def do_render():
        #print(time.time()-time0)
        t = (time.time() - time0)/10
        eye = np.array([np.sin(t*10)/3,0,0])
        #eye = np.array([0,0,0])
        up = np.array([0,1,0])
        #target = eye+np.array([0,0,-1])
        target = eye+np.array([np.sin(t),0,-np.cos(t)])

        view_matrix = lookAt(eye,target,up)
        proj_matrix = perspective(fovy, width/height, 0.1, 1000.0)
        mvp_matrix = proj_matrix@view_matrix

        renderer.render(mvp_matrix)

        glutSwapBuffers()
    
    def keyboard(key,x,y):
        print(key)
        ch = key.decode("utf-8")
        print(ch)
        if key == GLUT_KEY_LEFT:
            eye[0] -= .1
        elif key == GLUT_KEY_RIGHT:
            eye[0] += .1
        print(eye)

    glutDisplayFunc(do_render)
    glutIdleFunc(do_render)
    glutKeyboardFunc(keyboard)
    glutMainLoop()

    #while True:
        #for event in pg.event.get():
            #if event.type == pg.QUIT:
                #pg.quit()
                #quit()
    #
    #renderer.render(mvp_matrix)
    #
    #pg.display.flip()
    #pg.time.wait(10)
