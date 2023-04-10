#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below
import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image 

class Color:
    def __init__(self, R, G, B):
        self.color=np.array([R,G,B]).astype(np.float64)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)
    
class Shader:
    def __init__(self, name, diffuseColor, specularColor=None, exponent=None):
        self.name = name
        self.diffuseColor = diffuseColor
        self.specularColor = specularColor
        self.exponent = exponent

class Sphere:
    def __init__(self, radius, center, shader):
        self.radius = radius
        self.center = center
        self.shader = shader

def tracing(mray, viewPoint, sf_List, index, l_list):
    surface = sf_List[index]
    n = np.array([0,0,0])

    if type(surface).__name__ == "Sphere":
        n = viewPoint + mray - surface.center
        n = n / np.linalg.norm(n)

    r,g,b=0,0,0

    for light in l_list:
        lightvec = - mray + light[0] - viewPoint
        lightvec = lightvec / np.sqrt(np.sum(lightvec * lightvec))

        index2 = -1
        smax = float('inf')

        for i,surface_ in enumerate(sf_List):
            x = np.sum((-lightvec) * (-lightvec))
            y = np.sum((light[0] - surface_.center) * (-lightvec))
            z = np.sum((light[0] - surface_.center)**2) - surface_.radius**2
            a = y**2-x*z

            if a >= 0:
                w=np.sqrt(a)
                s=-y+w
                t=-y-w
                if (s >= 0) :
                    if (smax >= s / x) :
                        smax = s / x
                        index2 = i
                    if (smax >= t / x) :
                        smax = t / x
                        index2 = i
        alight=light[1]
        if index2 == index:
            red, green, blue = surface.shader.diffuseColor
            diffuse_ = max(0, np.dot(lightvec, n))
            r += red * alight[0] * diffuse_
            g += green * alight[1] * diffuse_
            b += blue * alight[2] * diffuse_

            if surface.shader.specularColor is not None:
                nvector = (- mray) / np.linalg.norm((- mray))
                h = (nvector + lightvec) / np.linalg.norm(nvector + lightvec)
                specular_ = np.power(np.maximum(0, np.dot(n, h)), surface.shader.exponent)
                r += surface.shader.specularColor[0] * alight[0] * specular_
                g += surface.shader.specularColor[1] * alight[1] * specular_
                b += surface.shader.specularColor[2] * alight[2] * specular_

    res = Color(r,g,b)
    res.gammaCorrect(2.2)
    return res.toUINT8()

def main():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # set default values
    viewDir = np.array([0,0,-1]).astype(np.float64)
    viewUp = np.array([0,1,0]).astype(np.float64)
    viewProjNormal = -1*viewDir  # you can safely assume this. (no examples will use shifted perspective camera)
    viewWidth = 1.0
    viewHeight = 1.0
    projDistance = 1.0
    intensity = np.array([1,1,1]).astype(np.float64)  # how bright the light is.

    imgSize=np.array(root.findtext('image').split()).astype(np.int64)

    for c in root.findall('camera'):
        viewPoint = np.array(c.findtext('viewPoint').split()).astype(np.float64)
        if(c.findtext('viewDir')):
            viewDir = np.array(c.findtext('viewDir').split()).astype(np.float64)
        if(c.findtext('projNormal')):
            viewProjNormal = np.array(c.findtext('projNormal').split()).astype(np.float64)
        if(c.findtext('viewUp')):
            viewUp = np.array(c.findtext('viewUp').split()).astype(np.float64)
        if(c.findtext('projDistance')):
            projDistance = np.array(c.findtext('projDistance').split()).astype(np.float64)
        if(c.findtext('viewWidth')):
            viewWidth=np.array(c.findtext('viewWidth').split()).astype(np.float64)
        if(c.findtext('viewHeight')):
            viewHeight = np.array(c.findtext('viewHeight').split()).astype(np.float64)

    s_List = []
    for c in root.findall('shader'):
        diffuseColor = np.array(c.findtext('diffuseColor').split()).astype(np.float64)
        shadername = c.get('name')
        s_object = Shader(shadername, diffuseColor)
        
        specularColor = c.findtext('specularColor')
        exponent = c.findtext('exponent')

        if specularColor is not None :
            if exponent is not None:
                specularColor = np.array(specularColor.split()).astype(np.float64)
                exponent = np.array(exponent.split()).astype(np.float64)[0]
                s_object.specularColor = specularColor
                s_object.exponent = exponent
        
        s_List.append(s_object)

    sf_List = []
    for c in root.findall('surface'):
        sf_type = c.get('type')    
        sf_shader = c.find('shader')
        sf_ref = sf_shader.get('ref')
        shaders = None
        for shadings in s_List:
            if shadings.name == sf_ref:
                shaders = shadings
                break

        if sf_type == 'Sphere':
            radius = np.array(c.findtext('radius')).astype(np.float64)
            center = np.array(c.findtext('center').split()).astype(np.float64)
            sf_List.append(Sphere(radius, center, shaders))

    l_list = []
    for c in root.findall('light'):
        position = np.array(c.findtext('position').split()).astype(np.float64)
        intensity = np.array(c.findtext('intensity').split()).astype(np.float64)
        lights = (position, intensity)
        l_list.append(lights)
    #code.interact(local=dict(globals(), **locals()))  

    # Create an empty image
    channels=3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:,:]=0

    # replace the code block below!
    v1=viewDir
    v11=v1/np.sqrt(np.sum(v1 * v1))
    v2=np.cross(v11,viewUp)
    v22=v2/np.sqrt(np.sum(v2 * v2))
    v3=np.cross(v11,v22)
    v33=v3/np.sqrt(np.sum(v3 * v3))
    v44 = v11 * projDistance - v22 * (viewWidth / imgSize[0]) * (imgSize[0] / 2 + 1 / 2) - v33 * (viewHeight / imgSize[1]) * (imgSize[1] / 2 + 1 / 2)

    img = np.zeros((imgSize[1], imgSize[0], 3), dtype=np.uint8)
    img[10, :, :] = (255, 255, 255)
    for i in range(imgSize[1]):
        img[i, i, :] = (255, 0, 0)
    for i in range(imgSize[0]):
        img[i, 0, :] = (0, 0, 255)
    img[5, :, :] = (255, 255, 255)

    for x in np.arange(imgSize[0]):
        for y in np.arange(imgSize[1]):
            tracer = v44 + v22 * x * (viewWidth / imgSize[0]) + v33 * y * (viewHeight / imgSize[1])

            index = -1
            smax = float('inf')

            for j,surface in enumerate(sf_List):
                d1 = np.sum(tracer * tracer)
                d2 = np.sum((viewPoint - surface.center) * tracer)
                d3 = np.sum((viewPoint - surface.center)**2) - surface.radius**2
                a = d2**2-d1*d3

                if a >= 0:
                    w=np.sqrt(a)
                    s=-d2+w
                    t=-d2-w
                    if (s >= 0) :
                        if (smax >= s / d1) :
                            smax = s / d1
                            index = j
                        if (smax >= t / d1) :
                            smax = t / d1
                            index = j

            
            if(index != -1):
                img[y][x] = tracing(smax*tracer, viewPoint, sf_List, index, l_list)
            else:
                img[y][x] = np.array([0, 0, 0])

    rawimg = Image.fromarray(img, 'RGB')
    #rawimg.save('out.png')
    rawimg.save(sys.argv[1]+'.png')

if __name__=="__main__":
    main()