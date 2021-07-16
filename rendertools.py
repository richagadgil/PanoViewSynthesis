from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from imageio import imread, imwrite
from scipy.special import cotdg

vert = """
#version 330

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;

out vec2 texCoord;

uniform mat4 mvp;

void main()
{
    gl_Position = mvp * vec4(iPosition,1);
    texCoord = iTexCoord;
}
"""

frag = """
#version 330

uniform sampler2D sampler;

in vec2 texCoord;

out vec4 color;

void main()
{
    color = texture(sampler,texCoord);
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     self.texture.shape[1], self.texture.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, self.texture)

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
        glBindVertexArray(self.vertexArrayObject)
        glDrawElements(GL_TRIANGLES, self.indices.size,
                       GL_UNSIGNED_SHORT, None)


class Cylinder(Mesh):
    def __init__(self, bottom, top, radius, texturepath, nsegments=1024):
        thetarange = np.linspace(0, 2*np.pi, nsegments)
        self.vertices = []
        self.texCoords = []
        for theta in thetarange:
            x = radius*np.cos(theta)
            z = radius*np.sin(theta)
            self.vertices.append(np.array([x, bottom, z]))
            self.vertices.append(np.array([x, top, z]))
            self.texCoords.append(np.array([theta/(2*np.pi), 0]))
            self.texCoords.append(np.array([theta/(2*np.pi), 1]))
        self.indices = []
        for i in range(len(self.vertices)):
            self.indices.append(
                np.array([i, (i+1) % len(self.vertices), (i+2) % len(self.vertices)]))
        self.vertices = np.stack(self.vertices, axis=0).astype(np.float32)
        self.texCoords = np.stack(self.texCoords, axis=0).astype(np.float32)
        self.indices = np.stack(self.indices, axis=0).astype(np.uint16)

        self.texture = imread(texturepath)
        self.texture = np.flipud(self.texture)


class MyCylinder(Mesh):
    def __init__(self, bottom, top, radius, texturepath, nsegments=1024):
        thetarange = np.linspace(0, 2*np.pi, nsegments)
        self.vertices = []
        self.texCoords = []
        for theta in thetarange:
            x = radius*np.sin(theta)
            y = radius*np.cos(theta)

            self.vertices.append(np.array([x, y, top]))
            self.vertices.append(np.array([x, y, bottom]))
            self.texCoords.append(np.array([theta/(2*np.pi), 0]))
            self.texCoords.append(np.array([theta/(2*np.pi), 1]))
        self.indices = []
        for i in range(len(self.vertices)):
            self.indices.append(
                np.array([i, (i+1) % len(self.vertices), (i+2) % len(self.vertices)]))
        self.vertices = np.stack(self.vertices, axis=0).astype(np.float32)
        self.texCoords = np.stack(self.texCoords, axis=0).astype(np.float32)
        #self.texCoords = np.rot90(self.texCoords, axes=(-2, -1))
        self.indices = np.stack(self.indices, axis=0).astype(np.uint16)

        self.texture = imread(texturepath)

        #self.texture = np.rot90(self.texture, axes=(-2, -1))

class Sphere(Mesh):
    def __init__(self, radius: float, width_segments: int, height_segments: int):
        # points = (height segments, width segments)
        height_range = np.linspace(-np.pi, np.pi, height_segments + 1)
        width_range = np.linspace(0, 2*np.pi, width_segments + 1)
        self.vertices = []
        self.normals = []
        self.texCoords = []
        self.indices = []
        for phi in height_range:
            u_offset = 0.5 / width_segments if phi == -np.pi else -0.5 / width_segments if phi == np.pi else 0
            norm_h = (phi + np.pi)/2*np.pi
            for theta in width_range:
                norm_w = theta/2*np.pi
                x = - radius * np.cos(theta) * np.sin(phi)
                y = radius * np.cos(phi)
                z = radius * np.sin(theta) * np.sin(phi)
                v = np.array([x, y, z])
                self.vertices.append(v)
                # Normals point inward
                self.normals.append(-v / np.linalg.norm(v))
                self.texCoords.append(np.array([norm_w + u_offset, 1 - norm_h]))
        combinations = len(height_range) * len(width_range)
        idx_grid = np.linspace(0, combinations, combinations + 1).reshape(len(height_range), len(width_range))

        for y in range(0, height_segments):
            for x in range(0, width_segments):
                a = idx_grid[y][x + 1]
                b = idx_grid[y][x]
                c = idx_grid[y + 1][x]
                d = idx_grid[y + 1][x + 1]
                if y > 0: self.indices.append(np.array([a,b,d]))
                if y != height_segments - 1: self.indices.append(np.array([b,c,d]))

        

        
                


                

class Plane(Mesh):
    def __init__(self, depth, texturepath):
        self.vertices = [[-1, -1, -1],
                         [1, -1, -1],
                         [1, 1, -1],
                         [-1, 1, -1]]
        self.texCoords = [[0, 0],
                          [1, 0],
                          [1, 1],
                          [0, 1]]
        self.indices = [[0, 1, 2], [2, 3, 0]]
        self.vertices = np.array(self.vertices).astype(np.float32)
        self.vertices *= depth
        self.texCoords = np.array(self.texCoords).astype(np.float32)
        self.indices = np.array(self.indices).astype(np.uint16)

        self.texture = imread(texturepath)
        self.texture = np.flipud(self.texture)


class Renderer:
    def __init__(self, meshes, width, height, offscreen=False):
        self.meshes = meshes
        self.width = width
        self.height = height
        self.offscreen = offscreen

        shaderDict = {GL_VERTEX_SHADER: vert, GL_FRAGMENT_SHADER: frag}

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

    def initializeShaders(self, shaderDict):
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
                raise RuntimeError("Compilation failure (" + strShaderType + " shader):\n" +
                                   glGetShaderInfoLog(shaderObjects[-1]).decode('utf-8'))

            glAttachShader(self.shaderProgram, shaderObjects[-1])

        glLinkProgram(self.shaderProgram)
        status = glGetProgramiv(self.shaderProgram, GL_LINK_STATUS)

        if status == GL_FALSE:
            raise RuntimeError(
                "Link failure:\n" + glGetProgramInfoLog(self.shaderProgram).decode('utf-8'))

        for shader in shaderObjects:
            glDetachShader(self.shaderProgram, shader)
            glDeleteShader(shader)

    def configureShaders(self, mvp):
        mvpUnif = glGetUniformLocation(self.shaderProgram, "mvp")

        glUseProgram(self.shaderProgram)
        glUniformMatrix4fv(mvpUnif, 1, GL_TRUE, mvp)

        samplerUnif = glGetUniformLocation(self.shaderProgram, "sampler")
        glUniform1i(samplerUnif, 0)

        glUseProgram(0)

    def initializeFramebufferObject(self):
        """
        Create an FBO and assign a texture buffer to it for the purpose of offscreen rendering to the texture buffer
        """
        self.renderedTexture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.renderedTexture)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width,
                     self.height, 0, GL_RGB, GL_FLOAT, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glBindTexture(GL_TEXTURE_2D, 0)

        self.depthRenderbuffer = glGenRenderbuffers(1)

        glBindRenderbuffer(GL_RENDERBUFFER, self.depthRenderbuffer)

        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)

        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        self.framebufferObject = glGenFramebuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)

        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.renderedTexture, 0)

        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depthRenderbuffer)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                'Framebuffer binding failed, probably because your GPU does not support this FBO configuration.')

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render(self, mvp):
        if self.offscreen:
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebufferObject)

        self.configureShaders(mvp)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
            data = glReadPixels(0, 0, self.width, self.height,
                                GL_RGB, GL_UNSIGNED_BYTE)
            rendering = np.frombuffer(data, dtype=np.uint8).reshape(
                self.height, self.width, 3)

            return np.flipud(rendering)


def normalize(vec):
    return vec / np.linalg.norm(vec)


def lookAt(eye, center, up):
    F = center - eye
    f = normalize(F)
    s = np.cross(f, normalize(up))
    u = np.cross(normalize(s), f)
    M = np.eye(4)
    M[0, 0:3] = s
    M[1, 0:3] = u
    M[2, 0:3] = -f

    T = np.eye(4)
    T[0:3, 3] = -eye
    return M@T


def perspective(fovy, aspect, zNear, zFar):
    f = cotdg(fovy/2)
    M = np.zeros((4, 4))
    M[0, 0] = f/aspect
    M[1, 1] = f
    M[2, 2] = (zFar+zNear)/(zNear-zFar)
    M[2, 3] = (2*zFar*zNear)/(zNear-zFar)
    M[3, 2] = -1
    return M
