#install face_recognition, PIL, and cv2#

#import necessary modules

import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import face_recognition
from PIL import Image, ImageDraw



#draws a filter over an image frame using an alpha channel

def draw_filter(frame, filterimage, x_offset, y_offset):
    (h, w) = (filterimage.shape[0], filterimage.shape[1])
    (imageh, imagew) = (frame.shape[0], frame.shape[1])

    if y_offset + h >= imageh:
        filterimage = filterimage[0: imageh - y_offset, :, :]
    if x_offset + w >= imagew:
        filterimage = filterimage[0: imagew - x_offset, :, :]
    if x_offset < 0:
        filterimage = filterimage[:, abs(x_offset)::, :]
        w = filterimage.shape[1]
        x_offset = 0

    for c in range(3):
        frame[y_offset: y_offset + h, x_offset: x_offset + w, c] = filterimage[:, :, c] * (filterimage[:, :, 3]//255) + frame[y_offset:y_offset + h, x_offset:x_offset + w, c] * (1 - filterimage[:, :, 3]//255)

    return frame


#applies the filter over the frame
def apply_filter(image, path2filter, x, y, w, h):
    filterimage = cv2.imread(path2filter, -1)
    filterimage = cv2.resize(filterimage, (w, h))
    image = draw_filter(image, filterimage, x, y)

#appiles the filter above a certain position(in this case, above the head)
def apply_head_filter(image, path2filter, width, xpos, ypos):
    filterimage = cv2.imread(path2filter, -1)
    (h_filter, w_filter) = (filterimage.shape[0], filterimage.shape[1])
    factor = 1 * width / w_filter
    filterimage = cv2.resize(filterimage, (0, 0), fx = factor, fy = factor)
    (h_filter, w_filter) = (filterimage.shape[0], filterimage.shape[1])

    yorig = ypos - h_filter

    if yorig <0:
        filterimage = filterimage[abs(yorig)::,:,:]
        yorig = 0

    image = draw_filter(image, filterimage, xpos, yorig)


#########################################

#global variables to be used for making animations
lightsindex = 0
snowindex = 0

#creates a class for Frames with methods used to add animations on a frame
class Frame:

    #the image itself is the attribute of the class
    def __init__(self, image):
        self.image = image

    #iterates over a folder containing frames of an animation
    #applies one animation frame over the existing frame and then replaces it with the next animation frame over the next frame
    def putLights(self):
        global lightsindex
        lightsdir = "./lights/"
        lights = [l for l in listdir(lightsdir) if isfile(join(lightsdir, l))]

        i = lightsindex
        apply_filter(self.image, lightsdir + lights[i], 0, 0, 640, 480)
        i += 1
        i = 0 if i >= len(lights) else i
        lightsindex = i

    def putSnow(self):
        global snowindex
        snowdir = "./snow/"
        snow = [s for s in listdir(snowdir) if isfile(join(snowdir, s))]

        s = snowindex
        apply_filter(self.image, snowdir + snow[s], 0, 0, 640, 480)
        s += 1
        s = 0 if s >= len(snow) else s
        snowindex = s


#creates a subclass of Frame for Face objects
class Face(Frame):

    def __init__(self, image):
        #inherits the attribute of the Frame class
        Frame.__init__(self, image)

        #contains the list of locations of all faces
        self.faces = face_recognition.face_locations(self.image)

        #contains the facial landmarks of all faces
        self.landmarks = face_recognition.face_landmarks(self.image)


    #blurs a face
    def diumano(self):
        for (top, right, bottom, left) in self.faces:
            y = top
            x = left
            w = right - left
            h = bottom - top
            face_image = self.image[y:y+h, x:x+w]
            face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

            self.image[y:y+h, x:x+w] = face_image

    #draws a beard, colors the eyebrows and moth, and adds a mustache

    def Santa(self):
        for landmark in self.landmarks:

            #converts image to PIL form
            self.image = Image.fromarray(self.image)
            d = ImageDraw.Draw(self.image, "RGBA")

            #draws a colored polygon over the eyebrows
            d.polygon(landmark['left_eyebrow'], fill = (255, 255, 255, 128))
            d.polygon(landmark['right_eyebrow'], fill = (255, 255, 255, 128))
            d.line(landmark['left_eyebrow'], fill = (255, 255, 255, 150), width = 5)
            d.line(landmark['right_eyebrow'], fill = (255, 255, 255, 150), width = 5)


            #gets the 3 middle points of the chin and adds 30(15) to the y coordinate(to simulate a beard)
            chin = landmark['chin']

            mid = len(chin)//2
            (x, y) = chin[mid]
            chin[mid] = (x, y+30)

            mid1 = len(chin)//2 - 1
            (x, y) = chin[mid1]
            chin[mid1] = (x, y+15)

            mid2 = len(chin)//2 + 1
            (x, y) = chin[mid2]
            chin[mid2] = (x, y+15)



            #draws a colored polygon of the chin with the simulated beard
            d.polygon(chin[5:-5], fill = (255, 255, 255, 255))
            d.line(chin[5:-5], fill = (255, 255, 255, 150), width = 5)

            #draws another colored polygon for the sides of the chin
            d.polygon(landmark['chin'][-9:-3], fill = (255, 255, 255, 255))
            d.polygon(landmark['chin'][3:9], fill = (255, 255, 255, 255))


            #colors the top and bottom lip
            d.polygon(landmark['top_lip'], fill = (255, 255, 255, 128), outline = (255, 255, 255, 128))
            d.polygon(landmark['bottom_lip'], fill = (255, 255, 255, 128), outline = (255, 255, 255, 128))

            #converts image back to a numpy array
            self.image = np.array(self.image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)



            #adds a mustache between the nose and top_lip
            mid_lip = landmark['top_lip'][len(landmark['top_lip'])//2]
            mid_nose = landmark['nose_tip'][len(landmark['nose_tip'])//2]
            
            #gets the average distance between the nose and the mouth
            nose_to_lip_height = mid_lip[1] - mid_nose[1]
            
            #gets the width of the mouth
            #rightmost x coordinate of the top lip minus leftmos x coordinate of the top lip
            mouth_width = landmark['top_lip'][6][0] - landmark['top_lip'][0][0]


            #gets the leftmost point of the top_lip to serve as starting point for mustache
            x1, y1 = landmark['top_lip'][0][0], landmark['top_lip'][0][1]


            #applies a mustache filter with the same width as the mouth and located 1/3 of the way from the lips to the nose
            apply_head_filter(self.image, "whitestache.png", mouth_width, x1, y1- nose_to_lip_height//3)


        return self.image

    #draws a hat filter
    def putHat(self):
        for (top, right, bottom, left) in self.faces:
            #first x and y coordinate is the left and top respectively
            #width is the difference of the right x coordinate and the left x coordinate
            #height is the difference between the bottom y and the top y coordinate

            y = top
            x = left
            w = right - left
            h = bottom - top


            #applies a filter above the face with a width of 1.05 times the width of the face
            apply_head_filter(self.image, "christmashat.png", w*1.05, x, y)
