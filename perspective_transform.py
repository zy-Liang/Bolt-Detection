# 本文件进行相机标定，去除畸变和透视变换
# 支持批量读取识别用照片
# 请创建相应的文件夹，并修改下方的文件路径SOURCE_CALIBRATION，SOURCE和RESULT

import glob
import numpy as np
import cv2

# 标定用照片的文件路径：
SOURCE_CALIBRATION = r'E:\project_vis\code\optimize_recog\source\new_image\calibration\*.jpg'
# 测量用照片的文件路径：
SOURCE = r'E:\project_vis\code\optimize_recog\source\new_image\bolt\6.jpg'
# 结果文件的保存路径：
RESULT = r'E:\project_vis\code\optimize_recog\result\new_image\perspective\6.jpg'

CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
CHESSBOARD_SIZE = (11, 8)

# 相机标定
def perspective_recover(image, criteria, w, h, size=4000, ratio=0.03):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 识别棋盘格角点
    ret_1, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    objp2 = np.zeros((11 * 8, 2), np.float32)
    cap = max(h, w)
    k = np.mgrid[-5:6:1, -4:4:1].T.reshape(-1, 2)
    objp2[:, :2] = (k * ratio + 0.5) * cap
    pts1 = np.float32([corners[0][0], corners[10][0], corners[-1][0], corners[-11][0]])
    vec = corners[10][0] - corners[0][0]
    tan = vec[1] / vec[0]
    if tan <= 1 and tan >= -1:
        if vec[0] >= 0:
            pts2 = np.float32([objp2[0], objp2[10], objp2[-1], objp2[-11]])
        else:
            pts2 = np.float32([objp2[-1], objp2[-11], objp2[0], objp2[10]])
    else:
        if vec[1] >= 0:
            pts2 = np.float32([objp2[10], objp2[-1], objp2[-11], objp2[0]])
        else:
            pts2 = np.float32([objp2[-11], objp2[0], objp2[10], objp2[-1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    pts_ = cv2.perspectiveTransform(pts, M)
    [x_min, y_min] = np.int32(pts_.min(axis=0).ravel()-0.5)
    [x_max, y_max] = np.int32(pts_.max(axis=0).ravel()+0.5)
    diff = [-x_min, -y_min]
    H_diff = np.array([[1, 0, diff[0]], [0, 1, diff[1]], [0, 0, 1]])
    H = H_diff.dot(M)
    dst = cv2.warpPerspective(image, H, (x_max-x_min, y_max-y_min))
    return dst

if __name__ == '__main__':
    # 以灰度格式读取标定用照片
    calibration_img = [cv2.imread(file, 0) for file in glob.glob(SOURCE_CALIBRATION)]

    objp= np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    img_points = []
    obj_points = []
    for gray_image in calibration_img:
        # 识别并优化棋盘格角点坐标
        retval, corners = cv2.findChessboardCorners(gray_image, CHESSBOARD_SIZE, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                      cv2.CALIB_CB_FAST_CHECK +
                                                                      cv2.CALIB_CB_NORMALIZE_IMAGE)
        corners2 = cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), CRITERIA)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        obj_points.append(objp)

        # 调用相机标定
        size_all = calibration_img[0].shape[::-1]
        retval, mtx, dist, rvec, tvec = cv2.calibrateCamera(obj_points, img_points, size_all, None, None)
        
        # 相机参数优化
        h, w = calibration_img[0].shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 读取识别用照片
    images = [cv2.imread(file) for file in glob.glob(SOURCE)]
    undistorted_images = []
    for image in images:
    # 去除畸变
        undistorted = cv2.undistort(image, mtx, dist, None, new_camera_mtx)
        undistorted_images.append(undistorted)

    # 透视变换
    for count, undistorted_image in enumerate(undistorted_images):
        perspective = perspective_recover(undistorted_image, CRITERIA, w, h)
        cv2.imwrite(RESULT, perspective)

    print('Finished.')

