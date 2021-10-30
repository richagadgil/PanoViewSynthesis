from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from imageio import imread, imwrite
from scipy.special import cotdg
import cv2

mpi_vert = """
#version 330

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;

out vec2 vTexCoord;

uniform mat4 mvp;

void main()
{
    gl_Position = mvp * vec4(iPosition,1);
    vTexCoord = iTexCoord;
}
"""

mpi_frag = """
#version 330

uniform sampler2D tColor;

in vec2 vTexCoord;

out vec4 color;

void main()
{
    color = texture(tColor,vTexCoord);
    //color = vec4(vTexCoord,0,1);
}
"""

depth_vert = """
#version 330

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;

out vec2 vTexCoord;
out float vDisparity;

uniform sampler2D tDisparity;
uniform mat4 mvp;

void main()
{
    float disparity = texture(tDisparity,iTexCoord).r;
    gl_Position = mvp * vec4(iPosition/disparity,1);
    vTexCoord = iTexCoord;
    vDisparity = disparity;
}
"""

depth_frag = """
#version 330

uniform sampler2D tColor;
uniform sampler2D tDisparity;

in vec2 vTexCoord;
in float vDisparity;

out vec3 color;

void main()
{
    color = texture(tColor,vTexCoord).rgb;
    //color = vec3(vTexCoord,0);
    //color = texture(tDisparity,vTexCoord).rgb;
}
"""

class Mesh:
    def __init__(self):
        self.vertices = []
        self.texCoords = []
        self.faces = []

    def initializeVertexBuffer(self):
        """
        Assign the triangular mesh data and the triplets of vertex indices that form the triangles (index data) to VBOs
        """
        self.vertexBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        
        self.texCoordBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.texCoordBufferObject)
        glBufferData(GL_ARRAY_BUFFER, self.texCoords, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.indexBufferObject = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBufferObject)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def initializeTexture(self):
        self.textureID = glGenTextures(1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        if self.texture.shape[2] == 4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.texture.shape[1], self.texture.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, self.texture)
        else:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.texture.shape[1], self.texture.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, self.texture)

        self.disparityID = None
        if self.disparity is not None:
            self.disparityID = glGenTextures(1)
            glActiveTexture(GL_TEXTURE1)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glBindTexture(GL_TEXTURE_2D, self.disparityID)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, self.texture.shape[1], self.texture.shape[0], 0, GL_RED, GL_UNSIGNED_BYTE, self.disparity)
            glActiveTexture(GL_TEXTURE0)

    def initializeVertexArray(self):
        """
        Creates the VAO to store the VBOs for the mesh data and the index data, 
        """
        self.vertexArrayObject = glGenVertexArrays(1)
        glBindVertexArray(self.vertexArrayObject)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBufferObject)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.texCoordBufferObject)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBufferObject)
        
        glBindVertexArray(0)

    def render(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        if self.disparityID is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.disparityID)
        glBindVertexArray(self.vertexArrayObject)
        glDrawElements(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_INT, None)

class Cylinder(Mesh):
    def __init__(self,bottom,top,radius,texturepath,disparitypath=None,nsegments=360,nvertsegments=1):
        urange = np.linspace(0,1,nsegments+1)
        vrange = np.linspace(0,1,nvertsegments+1)
        self.vertices = []
        self.texCoords = []
        self.indices = []
        n = 0
        for i in range(nsegments):
            for j in range(nvertsegments):
                theta1 = urange[i]*2*np.pi
                theta2 = urange[i+1]*2*np.pi
                y1 = bottom + vrange[j]*(top-bottom)
                y2 = bottom + vrange[j+1]*(top-bottom)

                x1 = radius*np.cos(theta1)
                z1 = radius*np.sin(theta1)
                x2 = radius*np.cos(theta2)
                z2 = radius*np.sin(theta2)
                self.vertices.append(np.array([x1,y1,z1]))
                self.vertices.append(np.array([x1,y2,z1]))
                self.vertices.append(np.array([x2,y1,z2]))
                self.vertices.append(np.array([x2,y2,z2]))
                self.texCoords.append(np.array([urange[i],vrange[j]]))
                self.texCoords.append(np.array([urange[i],vrange[j+1]]))
                self.texCoords.append(np.array([urange[i+1],vrange[j]]))
                self.texCoords.append(np.array([urange[i+1],vrange[j+1]]))

                self.indices.append(np.array([n,n+1,n+2]))
                self.indices.append(np.array([n+1,n+2,n+3]))

                n += 4
        self.vertices = np.stack(self.vertices,axis=0).astype(np.float32)
        self.texCoords = np.stack(self.texCoords,axis=0).astype(np.float32)
        self.indices = np.stack(self.indices,axis=0).astype(np.uint32)

        self.texture = imread(texturepath)
        if len(self.texture.shape) == 2:
            self.texture = np.stack([self.texture]*3,axis=-1)
        self.texture = np.flipud(self.texture)

        self.disparity = None
        if disparitypath is not None:
            self.disparity = imread(disparitypath)
            print('disparity shape',self.disparity.shape)
            self.disparity = np.flipud(self.disparity)

class DepthCylinder(Mesh):
    def __init__(self,height,radius,texturepath,disparitypath,nsegments=None,nvertsegments=None):
        self.texture = imread(texturepath)
        self.texture = np.flipud(self.texture)

        disparity = imread(disparitypath).astype('float32')/255.
        disparity = np.flipud(disparity)
        H,W = disparity.shape

        if nsegments is not None:
            W = nsegments
        if nvertsegments is not None:
            H = nvertsegments
        disparity = cv2.resize(disparity,(W,H),interpolation=cv2.INTER_LINEAR)
        
        bottom = -height/2
        top = height/2

        self.disparity = None
        self.vertices = []
        self.texCoords = []
        self.indices = []
        n = 0
        for y in range(H-1):
            v1 = y/(H-1)
            v2 = (y+1)/(H-1)
            for x in range(W):
                u1 = x/(W-1)
                u2 = ((x+1)%W)/(W-1)
                
                depth11 = 1./disparity[y,x]
                depth12 = 1./disparity[y+1,x]
                depth21 = 1./disparity[y,(x+1)%W]
                depth22 = 1./disparity[y+1,(x+1)%W]
                
                theta1 = u1*2*np.pi
                theta2 = u2*2*np.pi
                y1 = bottom + v1*(top-bottom)
                y2 = bottom + v2*(top-bottom)

                x1 = radius*np.cos(theta1)
                z1 = radius*np.sin(theta1)
                x2 = radius*np.cos(theta2)
                z2 = radius*np.sin(theta2)
                
                self.vertices.append(np.array([x1,y1,z1])*depth11)
                self.texCoords.append(np.array([u1,v1]))

                self.vertices.append(np.array([x1,y2,z1])*depth12)
                self.texCoords.append(np.array([u1,v2]))

                self.vertices.append(np.array([x2,y1,z2])*depth21)
                self.texCoords.append(np.array([u2,v1]))

                self.vertices.append(np.array([x2,y2,z2])*depth22)
                self.texCoords.append(np.array([u2,v2]))

                self.indices.append(np.array([n,n+1,n+2]))
                self.indices.append(np.array([n+1,n+2,n+3]))

                n += 4
        self.vertices = np.stack(self.vertices,axis=0).astype(np.float32)
        self.texCoords = np.stack(self.texCoords,axis=0).astype(np.float32)
        self.indices = np.stack(self.indices,axis=0).astype(np.uint32)

class Plane(Mesh):
    def __init__(self,depth,texturepath):
        self.vertices = [[-1,-1, -1],
                         [ 1,-1, -1],
                         [ 1, 1, -1],
                         [-1, 1, -1]]
        self.texCoords = [[0,0],
                          [1,0],
                          [1,1],
                          [0,1]]
        self.indices = [[0,1,2],[2,3,0]]
        self.vertices = np.array(self.vertices).astype(np.float32)
        self.vertices *= depth
        self.texCoords = np.array(self.texCoords).astype(np.float32)
        self.indices = np.array(self.indices).astype(np.uint32)

        self.texture = imread(texturepath)
        self.texture = np.flipud(self.texture)

class Renderer:
    def __init__(self,meshes,width,height,disparity=False,offscreen=False):
        self.meshes = meshes
        self.width = width
        self.height = height
        self.offscreen = offscreen
    
        if not disparity :
            shaderDict = {GL_VERTEX_SHADER: mpi_vert, GL_FRAGMENT_SHADER: mpi_frag}
        else:
            shaderDict = {GL_VERTEX_SHADER: depth_vert, GL_FRAGMENT_SHADER: depth_frag}

        self.initializeShaders(shaderDict)
        
        # Set the dimensions of the viewport
        glViewport(0, 0, width, height)
        
        # Performs z-buffer testing
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LEQUAL)
        glDepthRange(0.0, 1.0)
        
        # With all of our data defined, we can initialize our VBOs, FBO, and VAO to the OpenGL context to prepare for rendering
        if self.offscreen:
            self.initializeFramebufferObject()

        for mesh in self.meshes:
            mesh.initializeVertexBuffer()
            mesh.initializeVertexArray()
            mesh.initializeTexture()

    def initializeShaders(self,shaderDict):
        """
        Compiles each shader defined in shaderDict, attaches them to a program object, and links them (i.e., creates executables that will be run on the vertex, geometry, and fragment processors on the GPU). This is more-or-less boilerplate.
        """
        shaderObjects = []
        self.shaderProgram = glCreateProgram()
        
        for shaderType, shaderString in shaderDict.items():
            shaderObjects.append(glCreateShader(shaderType))
            glShaderSource(shaderObjects[-1], shaderString)
            
            glCompileShader(shaderObjects[-1])
            status = glGetShaderiv(shaderObjects[-1], GL_COMPILE_STATUS)
            if status == GL_FALSE:
                if shaderType is GL_VERTEX_SHADER:
                    strShaderType = "vertex"
                elif shaderType is GL_GEOMETRY_SHADER:
                    strShaderType = "geometry"
                elif shaderType is GL_FRAGMENT_SHADER:
                    strShaderType = "fragment"
                raise RuntimeError("Compilation failure (" + strShaderType + " shader):\n" + glGetShaderInfoLog(shaderObjects[-1]).decode('utf-8'))
            
            glAttachShader(self.shaderProgram, shaderObjects[-1])
        
        glLinkProgram(self.shaderProgram)
        status = glGetProgramiv(self.shaderProgram, GL_LINK_STATUS)
        
        if status == GL_FALSE:
            raise RuntimeError("Link failure:\n" + glGetProgramInfoLog(self.shaderProgram).decode('utf-8'))
            
        for shader in shaderObjects:
            glDetachShader(self.shaderProgram, shader)
            glDeleteShader(shader)

    def configureShaders(self,mvp):
        mvpUnif = glGetUniformLocation(self.shaderProgram, "mvp")

        glUseProgram(self.shaderProgram)
        glUniformMatrix4fv(mvpUnif, 1, GL_TRUE, mvp)

        tColorUnif = glGetUniformLocation(self.shaderProgram, "tColor")
        glUniform1i(tColorUnif, 0)

        tDisparityUnif = glGetUniformLocation(self.shaderProgram, "tDisparity")
        glUniform1i(tDisparityUnif, 1)

        glUseProgram(0)

    def initializeFramebufferObject(self):
        """
        Create an FBO and assign a texture buffer to it for the purpose of offscreen rendering to the texture buffer
        """
        self.renderedTexture = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.renderedTexture)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_FLOAT, None)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        self.depthRenderbuffer = glGenRenderbuffers(1)
        
        glBindRenderbuffer(GL_RENDERBUFFER, self.depthRenderbuffer)
        
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        
        self.framebufferObject = glGenFramebuffers(1)
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.renderedTexture, 0)
        
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depthRenderbuffer)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer binding failed, probably because your GPU does not support this FBO configuration.')
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def render(self,mvp):
        if self.offscreen:
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)

        self.configureShaders(mvp)

        if len(self.meshes)>1:
            glEnable(GL_BLEND)
            #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA) # pre-multiplied
        else:
            glDisable(GL_BLEND)

        glUseProgram(self.shaderProgram)

        #glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for mesh in self.meshes:
            mesh.render()

        glUseProgram(0)

        if self.offscreen:
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            rendering = np.frombuffer(data, dtype = np.uint8).reshape(self.height, self.width, 3)

            return np.flipud(rendering)

def normalize(vec):
    return vec / np.linalg.norm(vec)

def lookAt(eye,center,up):
    F = center - eye
    f = normalize(F)
    s = np.cross(f,normalize(up))
    u = np.cross(normalize(s),f)

    M = np.eye(4)
    M[0,0:3] = s
    M[1,0:3] = u
    M[2,0:3] = -f
    
    T = np.eye(4)
    T[0:3,3] = -eye

    return M@T

def perspective(fovy,aspect,zNear,zFar):
    f = cotdg(fovy/2)
    M = np.zeros((4,4))
    M[0,0] = f/aspect
    M[1,1] = f
    M[2,2] = (zFar+zNear)/(zNear-zFar)
    M[2,3] = (2*zFar*zNear)/(zNear-zFar)
    M[3,2] = -1
    return M

