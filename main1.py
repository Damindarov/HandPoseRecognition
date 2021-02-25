# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time

import pyrealsense2 as rs
import numpy as np
import cv2
# from yolo import YOLO


import copy
from pip._vendor.msgpack.fallback import xrange

def distance(x1, y1, x2, y2):
    c = np.math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return c

def centroid(max_contour):
    if max_contour is not None:
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return 0, 0


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def calculateFingers(res, drawing,xc,yc):
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)

    if len(hull) > 20:
        defects = None
        try:
            defects = cv2.convexityDefects(res, hull)
        except Exception as ex:
            pass

        if defects is not None:
            cnt = 0
            # print(res[0])
            max = 0
            ind = ()
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = np.math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem

                if np.math.pi >= angle >= np.math.pi*0.6:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    dist = distance(far[0],far[1],xc,yc)

                    if dist >= max:
                        ind = far
                        max = dist
            for i in range(len(far)):
                cv2.circle(drawing, ind, 4, [211, 84, 0], -1)


def print_hi(name):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    pc = rs.pointcloud()
    tr = rs.threshold_filter(min_dist=0.15, max_dist=1.5)

    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    # threshold_filter = rs.threshold_filter()
    # threshold_filter.set_option(rs.option.max_distance, 1)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    size_p = 4
    point_list = np.zeros(size_p)

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # frames = tr.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # distances = depth_frame.get_distance()
            width = depth_frame.get_width()
            height = depth_frame.get_height()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_frame_copy = tr.process(depth_frame)
            color_image_copy = tr.process(color_frame)

            depth_image = np.asanyarray(depth_frame_copy.get_data())
            color_image = np.asanyarray(color_image_copy.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

            color_image = cv2.bilateralFilter(color_image, 10, 50, 100)  # Smoothing


            # for i in range(vts.shape[0]):
            #     print(vts[i])
            # vts = vts[np.where(vts[:, 2] < 0.35)]



            # color_image = cv2.bilateralFilter(color_image, 5, 50, 100)
            # color_image = color_image[np.where(vts[:, 2] < 0.35)]

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

            cv2.imshow('depthColorMap',depth_colormap)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape


            # color_image = depth_colormap



            # dist_to_center = depth_frame.get_distance(int(width / 2), int(height / 2))
            # print(dist_to_center)

            # bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
            # fgmask = bgModel.apply(color_image)
            # kernel = np.ones((3, 3), np.uint8)
            # fgmask = cv2.erode(fgmask, kernel, iterations=1)
            # img = cv2.bitwise_and(color_image, color_image, mask=fgmask)

            bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
            fgmask = bgModel.apply(color_image)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            img = cv2.bitwise_and(color_image, color_image, mask=fgmask)

            # Skin detect and thresholding
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 48, 80], dtype="uint8")
            upper = np.array([20, 255, 255], dtype="uint8")
            skinMask = cv2.inRange(hsv, lower, upper)
            cv2.imshow('Threshold Hands', skinMask)


            images_p = None
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images_p = np.hstack((resized_color_image, depth_colormap))
            else:
                images_p = np.hstack((color_image, depth_colormap))
            cv2.imshow('depth', images_p)

            skinMask1 = copy.deepcopy(skinMask)
            contours, hierarchy = cv2.findContours(skinMask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in xrange(length):
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i
                        res = contours[ci]

                hull = cv2.convexHull(res)

                drawing = np.zeros(img.shape, np.uint8)
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
                drawing1 = drawing.copy()
                cx, cy = centroid(res)#center of mass

                zDepth = depth_frame.get_distance(int(cx), int(cy))



                point_list = np.append(point_list,zDepth)
                point_list = point_list[1:(size_p+1)]
                max_p = max(point_list)
                min_p = min(point_list)
                delta = max_p - min_p
                #Добавить окружность для точек xy
                if delta > 0.12 and min_p > 0 and list(point_list).index(min_p) < list(point_list).index(max_p):
                    point_list = np.zeros(size_p)
                    print('toutch')

                calculateFingers(res, drawing1,cx,cy)

                # chechcountr distacnes

                if zDepth < 3.9:
                    draw_circles(drawing1, [(cx, cy)])
                else:
                    drawing1 = np.zeros(img.shape, np.uint8)
                cv2.imshow('output', drawing1)

            # #detectfaces and remove it
            # gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # # Detect faces
            # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # # Draw rectangle around the faces
            #
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
            #     cv2.rectangle(skinMask, (x, y), (x + w, y + h), (0, 0, 0), -1)
            #
            # #detect skin
            # hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            # lower = np.array([0, 48, 80], dtype="uint8")
            # upper = np.array([20, 255, 255], dtype="uint8")
            # skinMask = cv2.inRange(hsv, lower, upper)#colorFilter
            #
            #
            #
            # cv2.imshow('d',skinMask)
            # cv2.imshow('d', color_image)
            #

            #
            # bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
            # fgmask = bgModel.apply(color_image)
            # kernel = np.ones((3, 3), np.uint8)
            # fgmask = cv2.erode(fgmask, kernel, iterations=1)
            # img = cv2.bitwise_and(color_image, color_image, mask=fgmask)
            #
            # #detectfaces and remove it
            # gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # # Detect faces
            # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # # Draw rectangle around the faces
            #
            # #detect skin
            # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # lower = np.array([0, 48, 80], dtype="uint8")
            # upper = np.array([20, 255, 255], dtype="uint8")
            # skinMask = cv2.inRange(hsv, lower, upper)
            #
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
            #     cv2.rectangle(skinMask, (x, y), (x + w, y + h), (0, 0, 0), -1)
            #
            # # Display the output
            # cv2.imshow('img', color_image)
            # cv2.imshow('Threshold Hands', skinMask)
            #
            # #depth_frame = frames.get_depth_frame()
            # # zDepth = depth_frame.get_distance(int(1), int(1))
            # # print(zDepth)
            #
            #
            #
            # # If depth and color resolutions are different, resize color image to match depth image for display
            # if depth_colormap_dim != color_colormap_dim:
            #     resized_color_image = cv2.resize(img, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
            #                                      interpolation=cv2.INTER_AREA)
            #     images = np.hstack((resized_color_image, depth_colormap))
            # else:
            #     images = np.hstack((color_image, depth_colormap))
            #
            # # Show images

            cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
