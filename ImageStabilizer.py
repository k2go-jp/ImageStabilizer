'''
How to Use:
    Many images : python ImageStabilizer.py --target ./images/path/to/folder --result result.mp4
    Videos      : python ImageStabilizer.py --target source.mp4 --result result.mp4
'''
import cv2
import numpy as np
import os
import sys
import time
import glob
import json
from absl import app, flags
from collections import deque
from typing import NamedTuple

FLAGS = flags.FLAGS
flags.DEFINE_string('target', 'sample.mp4', 'to stabilize images or movie if you select directory or file')
flags.DEFINE_string('result', 'tmp.mp4', 'filepath for new stabilized movie')
flags.DEFINE_string('masks', 'masks.json', 'filepath for mask data to select feature points')
flags.DEFINE_enum('useMask', 'none', ['none', 'file', 'gui'], 'to select whether you use mask data')
flags.DEFINE_bool('saveMask', False, 'to save mask data')

class BBox(NamedTuple):
    x: int
    y: int
    width: int
    height: int

class Stabilizer():
    def __init__(self, maxlen=5, threshold=1e-3):
        self.image_buffer = deque(maxlen=maxlen)
        self.akaze = cv2.AKAZE_create(threshold=threshold) # default threshold=1e-3 -> 1e-4
        self.masks = []

    def createVideoWriter(self, dst_movie, frame_rate, size):
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(dst_movie, fmt, frame_rate, size)

    def setMaskFromFile(self, cfg):
        with open(cfg) as f:
            tmp = json.load(f)
            lst = tmp.get('masks')
            for rect in lst:
                self.masks.append(BBox(rect[0], rect[1], rect[2], rect[3]))

    def setMaskFromGUI(self, frame0):
        rect = BBox(0, 0, 1920, 1080)
        param = {'drawing' : False, 'moving' : False, 'ix' : -1, 'iy' : -1, 'rect' : rect, 'finish' : False}
        # drawing is true if mouse is pressed
        # moving  is true if mouse is moved
        def draw_rectangle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                param['drawing'] = True
                param['finish']  = False
                param['ix'], param['iy'] = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if param['drawing'] == True:
                    param['moving'] = True
                    top    = max(min(param['iy'], y), 0)
                    left   = max(min(param['ix'], x), 0)
                    width  = abs(param['ix'] - max(min(x, 1920), 0))
                    height = abs(param['iy'] - max(min(y, 1080), 0))
                    param['rect'] = BBox(left, top, width, height)
            elif event == cv2.EVENT_LBUTTONUP:
                param['drawing'] = False
                param['moving']  = False
                param['finish']  = True
                top    = max(min(param['iy'], y), 0)
                left   = max(min(param['ix'], x), 0)
                width  = abs(param['ix'] - max(min(x, 1920), 0))
                height = abs(param['iy'] - max(min(y, 1080), 0))
                param['rect'] = BBox(left, top, width, height)
                print(param['rect'])

        cv2.namedWindow("test")
        cv2.setMouseCallback("test", draw_rectangle, param)
        while True:
            img = frame0.copy()
            if self.masks:
                for rect in self.masks:
                    upper_left = (rect.x, rect.y)
                    lower_right = (rect.x + rect.width, rect.y + rect.height)
                    cv2.rectangle(img, upper_left, lower_right, (0, 255, 255), 1)
            if param['moving']:
                upper_left = (param['rect'].x, param['rect'].y)
                lower_right = (param['rect'].x + param['rect'].width, param['rect'].y + param['rect'].height)
                cv2.rectangle(img, upper_left, lower_right, (0, 255, 0), 2)
            cv2.imshow("test", img)

            if param['finish']:
                self.masks.append(param['rect'])
                param['finish'] = False

            # Abort if pushing q-key
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    def writeMaskToFile(self, cfg):
        with open(cfg, 'w') as f:
            json.dump({'masks' : self.masks}, f, indent=4)

    def setMask(self, img):
        if FLAGS.useMask == 'file':
            self.setMaskFromFile(FLAGS.masks)
        elif FLAGS.useMask == 'gui':
            self.setMaskFromGUI(img)
            if FLAGS.saveMask:
                self.writeMaskToFile(FLAGS.masks)
        else:
            self.masks.append(BBox(0, 0, 1920, 1080))

    def writeNewFrame(self, img, msg=''):
        frame = img.copy()
        '''
        frame[:, 0:1080] = 0
        frame[:, 1320:1920] = 0
        frame[0:100, :] = 0
        frame[800:980, :] = 0
        '''
        cv2.putText(frame, msg, (10, 150),
            cv2.FONT_HERSHEY_PLAIN, 4.0,
            (255, 255, 255), 4, cv2.LINE_AA)
        self.writer.write(frame)

    def extractFeatures(self, img1):
        img1_trim = img1[0:980, 0:1920]
        gray1 = img1_trim.copy()

        mask = np.zeros(gray1.shape[:2], np.uint8)
        for rect in self.masks:
            mask[rect.y:rect.y+rect.height, rect.x:rect.x+rect.width] = 1

        #gray1 = cv2.cvtColor(img1_trim, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.akaze.detectAndCompute(gray1, mask)
        return (kp1, des1)

    def createHomographyMatrix(self, feat1, feat2, skew_th=0.05):
        (kp1, des1) = feat1
        (kp2, des2) = feat2

        size1 = [k.size for k in kp1 if k.size > 10]
        '''
        if size1:
            resp1 = [k.response for k in kp1 if k.size > 10]
            print(' Max : {0:5.2f}, Min : {1:5.2f}, Mean : {2:6.3f}, StDev : {3:6.3f}'.format(max(size1), min(size1), np.mean(size1), np.std(size1)))
            print(' Max : {0:5.2f}, Min : {1:5.2f}, Mean : {2:6.3f}, StDev : {3:6.3f}'.format(max(resp1), min(resp1), np.mean(resp1), np.std(resp1)))
        '''
        if des1 is None or des2 is None:
            #print('No keypoints in original images')
            return None, 'No keypoints'

        usingBF = True
        if usingBF:
            match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = match.knnMatch(des2, des1, k=2)
            good = []
            for m, n in matches:
                #print(m.distance,n.distance)
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        else:
            match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = match.match(des2, des1)
            matches = sorted(matches, key = lambda x:x.distance)
            good = matches

        MIN_MATCH_COUNT = 10
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ])
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ])
            H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
            if H is None or len(H.shape) < 2:
                #print('Not create homography matrix')
                return None, 'Not creation'
            #print(H, np.linalg.det(H))
            if np.linalg.det(H) < 0:
                #print('Not suitable homography matrix (Miss matching)')
                return None, 'Miss matching'
            elif np.linalg.det(H) < 1.0 - skew_th or np.linalg.det(H) > 1.0 + skew_th:
                #print('Not suitable homography matrix (Too skew)')
                return None, 'Too skew'
        else:
            #print('Not enough matches are found - {}/{}'.format(len(good), MIN_MATCH_COUNT))
            return None, 'Not enough matches'

        return H, 'Succeed!'

    def createStitchedImage(self, H, img1, img2):
        patch_warped = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[0]))
        img_stitched = patch_warped.copy()
        return img_stitched[:img1.shape[0], :img1.shape[1]]

    def stabilize(self, images):
        self.setMask(images[0])
        print(self.masks)

        old_features = None

        elapsed_times = [0.0, 0.0, 0.0]
        start = time.perf_counter()
        for i, img in enumerate(images):
            t1 = time.perf_counter()
            new_features = self.extractFeatures(img)
            t2 = time.perf_counter()
            elapsed_times[0] += (t2 - t1)
            '''
            while True:
                cv2.imshow('Keypoints', cv2.drawKeypoints(img, new_features[0], None, flags=4))
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            '''
            if new_features[1] is None:
                self.writeNewFrame(img, 'No keypoints')
            elif old_features is None:
                self.writeNewFrame(img, 'Base image')
                old_image    = img
                old_features = new_features
                self.image_buffer.append( (old_image, old_features) )
            else:
                t1 = time.perf_counter()
                tmp_list = list(reversed(self.image_buffer))
                min_j = None
                min_H = 100.0*np.eye(3)
                for j, item in enumerate(tmp_list):
                    H, msg = self.createHomographyMatrix(item[1], new_features)
                    if not H is None:
                        #print(i, j, np.linalg.det(H))
                        if abs(1.0 - np.linalg.det(min_H)) > abs(1.0 - np.linalg.det(H)):
                            min_H = H
                            min_j = j
                t2 = time.perf_counter()
                elapsed_times[1] += (t2 - t1)

                if min_j is None:
                    self.writeNewFrame(img, msg)
                else:
                    t1 = time.perf_counter()
                    res = self.createStitchedImage(min_H, tmp_list[min_j][0], img)
                    t2 = time.perf_counter()
                    elapsed_times[2] += (t2 - t1)

                    old_image    = res

                    t1 = time.perf_counter()
                    old_features = self.extractFeatures(res)
                    t2 = time.perf_counter()
                    elapsed_times[0] += (t2 - t1)
                    self.image_buffer.append( (old_image, old_features) )
                    self.writeNewFrame(res, 'Image stabilization')
            #'''
        end = time.perf_counter()
        print(f'Stabilizing Images: {end - start} sec.')
        print(f'- Extract Features: {elapsed_times[0]} sec.')
        print(f'- Select Images   : {elapsed_times[1]} sec.')
        print(f'- Transform Images: {elapsed_times[2]} sec.')

    def stabilize_images(self, src_target, dst_movie):
        #list = [f.name for f in os.scandir(folder) if f.name.endswith(".jpg")]
        #list = glob.glob(src_target + '/**/*.jpg', recursive=True)
        file_list = glob.glob(src_target + '/*.jpg', recursive=False)
        file_list.sort()
        '''
        for i, f in enumerate(file_list):
            print(i, f)
        '''
        start = time.perf_counter()
        images = []
        for file in file_list:
            img = cv2.imread(file)
            images.append(img)
        end = time.perf_counter()
        print(f'Loading Image: {end - start} sec.')

        frame_rate = 3
        size = (1920, 1080)
        self.createVideoWriter(dst_movie, frame_rate, size)

        '''
        for img in images:
            self.writeNewFrame(img, 'No image stabilization')
        '''
        self.stabilize(images)
        self.writer.release()

    def stabilize_movies(self, src_movie, dst_movie):
        #src_movie = '../SnowDetection/data/Amenomiya/20210107/Amenomiya_20210107123000.mp4'
        cap = cv2.VideoCapture(src_movie)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate  = int(cap.get(cv2.CAP_PROP_FPS))

        start = time.perf_counter()
        ret = True
        images = []
        while True and ret:
            ret, bgr = cap.read()
            if ret:
                images.append(bgr)
        end = time.perf_counter()
        print(f'Loading Image: {end - start} sec.')

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)
        #fname, ext = os.path.splitext(os.path.basename(src_movie))
        self.createVideoWriter(dst_movie, frame_rate, size)

        '''
        for img in images:
            self.writeNewFrame(img, 'No image stabilization')
        '''
        self.stabilize(images)
        self.writer.release()

def main(argv):
    #----
    # https://qiita.com/itoshogo3/items/7a3279668b24008a3761
    #----
    sys.setrecursionlimit(30000)
    s = Stabilizer()

    print('target is {}'.format(FLAGS.target))
    print('result is {}'.format(FLAGS.result))
    if os.path.isdir(FLAGS.target):
        s.stabilize_images(FLAGS.target, FLAGS.result)
    elif os.path.isfile(FLAGS.target):
        s.stabilize_movies(FLAGS.target, FLAGS.result)
    else:
        print('Error! target is not found')

if __name__ == '__main__':
  app.run(main)
