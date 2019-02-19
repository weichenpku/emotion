#! /usr/bin/python3
# -*-coding:utf-8
import os
import numpy as np
import cv2
import pupil_detect

WORK_DIR = "/home/soar/avis/"
FPS = 30
TYPES = ["ne", "jo", "sa", "fe"]
FEATURE_FILE = 'haarcascade_eye.xml'

eyeCascadeClassifier = cv2.CascadeClassifier(FEATURE_FILE)

def detect_objects(image, objectClassifier, divider=1):
    #increase divider for more speed, less accuracy
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #small_image = cv2.resize(gray_image, (int(gray_image.shape[1]/divider),int(gray_image.shape[0]/divider)))
    min_object_size = (30,30)
    haar_scale = 1.3
    min_neighbors = 4
    haar_flags = cv2.CASCADE_SCALE_IMAGE
    rects = objectClassifier.detectMultiScale(gray_image, haar_scale, min_neighbors, haar_flags, min_object_size, maxSize=(60,60))
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

class Eye():
    def __init__(self,x, y, x2, y2):
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.width = x2 - x
        self.height = y2 - y
        self.topcorner = (x, y)
        self.bottomcorner = (x2, y2)
    def __str__(self):
        return str(self.x) + " " + str(self.y) + " " + str(self.x2) + " " + str(self.y2)

def draw(photo):
    image = photo.copy()
    image_to_show = image
    height = image.shape[0]
    width = image.shape[1]
    eyes = detect_objects(image, eyeCascadeClassifier)
    #################### Select the rightmost eye ##################
    rightmost_eye = None
    if len(eyes) == 0:
        #print("No eye candidates detected")
        pass
    else:
        #print("Found " + str(len(eyes)) + " candidates")
        for eye in eyes:
            if eye[1] > height * 3 / 4 or eye[3] > height * 3 / 4:
                continue
            cv2.rectangle(image_to_show, (eye[0], eye[1]), (eye[2], eye[3]), (0, 255, 0), 2)
            (x,y) = (eye[0], eye[1])
            (x2,y2) = (eye[2], eye[3])
            if (rightmost_eye is None) or (x > rightmost_eye.x):
                rightmost_eye = Eye(x, y, x2, y2)

    ###############################################################
    ####### Extract the image of the rightmost eye ################
    eye_image_init = None
    if rightmost_eye is not None:
        eye_image_init = image[rightmost_eye.y:rightmost_eye.y2, rightmost_eye.x:rightmost_eye.x2]
    if eye_image_init is not None:
        #cv2.imshow('Face', image_to_show)
        pass
        #cv2.imshow('Image', eye_image_init)

    eye_image = None
    if eye_image_init is not None:
        eye_image = cv2.resize(eye_image_init, (eye_image_init.shape[1]*4, eye_image_init.shape[0]*4))
        to_show = eye_image
        #to_show = 255 - to_show
        #cv2.imshow('Resized', to_show)

    ###############################################################
    ### To assist in edge detection, try to "black out" sclera ####
    binary_eye_image = None
    if eye_image is not None:
        eye_histogram = [0]*256
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_RGB2GRAY)
        #eye_image = 255 - eye_image
        for i in xrange(256):
            value_count = (eye_image == i).sum()
            eye_histogram[i] = value_count
        count = 0
        index = 255
        while count < (eye_image.size*3/4):
            count += eye_histogram[index]
            index -= 1
        quarter_threshold = index
        #Multiply all parts of eye above bottom 1/4 brightness by 0
        #Might not work on people with light irises. Have not tried.
        #binary_eye_image = cv2.equalizeHist((eye_image < quarter_threshold) * eye_image)
        binary_eye_image = cv2.equalizeHist(eye_image)
        #cv2.imshow('Equalized', binary_eye_image)
        #binary_eye_image
        #binary_eye_image = eye_image
    ###############################################################
    ####### Look for circle (the iris). Save coordinates. ########
    relative_iris_coordinates = None
    iris_R = None
    if binary_eye_image is not None:
        eye_circles = cv2.HoughCircles(binary_eye_image, cv2.cv.CV_HOUGH_GRADIENT, 3, 350, maxRadius=50)
        if eye_circles is not None:
            #Usually gets the job done. Messy.
            #print("Iris find")
            #print(eye_circles)
            circle = eye_circles[0][0]
            relative_iris_coordinates = (circle[0], circle[1])
            cv2.circle(to_show, relative_iris_coordinates, circle[2], (0,0,255), thickness=2)
            iris_R = circle[2]
            #cv2.imshow('Found', to_show)
    ###############################################################
    ######### Put blue dot on eye in big picture ##################
    #absolute_iris_coordinates = None
    #if relative_iris_coordinates is not None and rightmost_eye is not None:
    #    absolute_iris_coordinates = (int(relative_iris_coordinates[0]+rightmost_eye.x), int(relative_iris_coordinates[1]+rightmost_eye.y))
        #cv2.circle(binary_eye_image, absolute_iris_coordinates, 5, (255,0,0), thickness=10)
    ###############################################################
    ####### Extract image of iris (and some surrounding) ##########
    iris_image = None
    if relative_iris_coordinates is not None and rightmost_eye is not None and iris_R is not None:
        #x = absolute_iris_coordinates[0]
        #y = absolute_iris_coordinates[1]
        x = relative_iris_coordinates[0]
        y = relative_iris_coordinates[1]
        #Should find a way to make these numbers adaptable.
        #It really messes things up when these arbitrary numbers don't work with
        #whatever image size or eye distance is actually being used.
        #YOU, DEAR READER, MUST CHANGE THESE FOR YOUR CAMERA
        #These numbers describe the guestimated shape of a captured image of an iris.
        #This program makes no effort to automatically find the size of the iris in
        #the captured image of the iris

        tmp_image = cv2.resize(eye_image_init, (eye_image_init.shape[1] * 4, eye_image_init.shape[0] * 4))
        #print(tmp_image.shape)
        if x - iris_R < 0 or y - iris_R < 0 or x + iris_R > tmp_image.shape[1] or y + iris_R > tmp_image.shape[0]:
            pass
        else:
            iris_image = tmp_image[int(y - iris_R): int(y + iris_R),int(x - iris_R):int(x + iris_R)]
        #iris_image_to_show = cv2.resize(iris_image, (iris_image.shape[1]*4, iris_image.shape[0]*4))

        #image_to_show[0:iris_image_to_show.shape[0], 00:00+iris_image_to_show.shape[1]] = iris_image_to_show
    ###############################################################
    ### Draw blue circle around iris. Draw green around pupil #####
    #Also, if the pupil and iris seem to be sensible shapes/sizes,
    #return the current iris_image (picture of the iris+surroundings)
    iris_picture = None
    pupil_R = None
    if iris_image is not None:
        iris_gray = cv2.cvtColor(iris_image, cv2.COLOR_RGB2GRAY)
        iris_circles_image = iris_image.copy()
        #print(iris_image.shape)
        iris_circles = cv2.HoughCircles(iris_gray, cv2.cv.CV_HOUGH_GRADIENT, 2, 100, maxRadius=30)
        if iris_circles is not None:
            circle=iris_circles[0][0]
            cv2.circle(iris_circles_image, (circle[0], circle[1]), circle[2], (255,0,0), thickness=2)
        pupil_coords = pupil_detect.find_pupil(iris_gray)
        if pupil_coords is not None:
            #print("pupil detected ", pupil_coords[2])
            pupil_R = pupil_coords[2]
            cv2.circle(iris_circles_image, pupil_coords[:2], pupil_coords[2], (0,255,0),4)
            #cv2.imshow('hehe', iris_circles_image)
        if iris_circles is not None and pupil_coords is not None:
            ic = iris_circles[0][0]
            pc = pupil_coords
            #Check if pupil is within iris
            if abs(ic[0]-pc[0])<ic[2] and abs(ic[1]-pc[1])<ic[2] and pc[2]<ic[2]:
                iris_picture = iris_circles_image
        #iris_circles_to_show = cv2.resize(iris_circles_image, (iris_circles_image.shape[1]*2,iris_circles_image.shape[0]*2))
        #image_to_show[0:iris_circles_to_show.shape[0], 200:200+iris_circles_to_show.shape[1]] = iris_circles_to_show

    #cv2.imshow('Image', image_to_show)
    #if cv2.waitKey(1000) > 0: #If we got a key press in less than 10ms
        #Also, cv2 has some weirdness where the image won't update without a waitKey()
        #return True, iris_picture
    if pupil_R is not None and iris_R is not None:
        return True, (float(pupil_R) / iris_R)
        #return False, None
    #return False, iris_picture
    return False, None

if __name__ == "__main__":
    #cnt = 0
    #total = 0
    data = {}
    cnt = 0
    for emotion in TYPES:
        print("Detecting pupils in " + emotion)
        data[emotion] = {}
        frames = os.listdir(WORK_DIR + emotion)
        for frame in frames:
            cnt += 1
            if cnt % 100 == 0:
                print(cnt)
            parts = frame.split(".")
            idx = parts[0][0:3]
            seq = parts[0].split("_")[1]
            if idx not in data[emotion]:
                data[emotion][idx] = []
            abPath = os.path.join(WORK_DIR + emotion,frame)
            #print(abPath)
            img = cv2.imread(abPath)
            if img is not None:
                a, b = draw(img)
            if b is not None:
                #cnt += 1
                data[emotion][idx].append((int(seq), b))

            #total += 1
            #print(b)

    features = {} # 4 type: ne jo sa fe, each emotion 5 features: max, min, median, mean, std, meanAD
    for emo in TYPES:
        print("Processing " + emo)
        features[emo] = {}
        for people in data[emo]:
            if people not in features[emo]:
                features[emo][people] = []
            data[emo][people] = sorted(data[emo][people], key=lambda x:x[0])
            tmp_data = data[emo][people]
            n = len(tmp_data)
            if n < 2:
                print("Too few data points for " + emo + ", id: " + people)
                continue
            idx = -1
            i = 1
            while i < len(tmp_data):
                print(tmp_data[i][0])
                if tmp_data[i][0] - tmp_data[i - 1][0] > 900:
                    idx = i
                    break
                i += 1
                idx = i
            print(len(tmp_data))
            print(idx)
            part1 = []
            part2 = []
            j = 0
            while j < idx:
                part1.append(tmp_data[j][1])
                j += 1
            while j < len(tmp_data):
                part2.append(tmp_data[j][1])
                j += 1

            part1 = np.array(part1)
            mean1 = np.mean(part1)
            sum1 = 0.0
            for k in range(len(part1)):
                sum1 += np.abs(part1[k] - mean1)
            sum1 = sum1 / float(len(part1))
            tmp_feature1 = [np.max(part1), np.min(part1), np.median(part1), np.mean(part1), np.std(part1), sum1]

            features[emo][people].append(tmp_feature1)
            if len(part2) > 2:
                part2 = np.array(part2)
                mean2 = np.mean(part2)
                sum2 = 0.0
                for k in range(len(part2)):
                    sum2 += np.abs(part2[k] - mean2)
                sum2 = sum2 / float(len(part2))
                tmp_feature2 = [np.max(part2), np.min(part2), np.median(part2), np.mean(part2), np.std(part2), sum2]
                features[emo][people].append(tmp_feature2)

    print("writing to files\n")
    with open("pupil.data", "wb") as f:
        for emo in TYPES:
            f.write(emo + "\n")
            for people in features[emo]:
                for items in features[emo][people]:
                    to_write = people
                    for item in items:
                        to_write = to_write + "\t" + str(item)
                    to_write += "\n"
                    f.write(to_write)


    #print(cnt)
    #print(total)
            #cv2.imshow('Image', img)
            #cv2.waitKey(1)
