# coding=utf-8
import cv2
import sys
import numpy as np
import pandas as pd
import pyrealsense2 as rs

class Check_Calibration_Data:

    def __init__(self):

        self.pc = rs.pointcloud()  # 点云
        self.points = rs.points()  # 点
        self.pipeline = rs.pipeline()  # 创建管道
        self.config = rs.config()  # 创建流式传输
        # 配置传输管道
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipe_profile = self.pipeline.start(self.config)  # 开启传输流
        self.align_to = rs.stream.color  # 对齐流
        self.align = rs.align(self.align_to)  # 设置为其他类型,允许和深度流对齐

        self.pattern_type = "symmetric_circles"
        self.pattern_rows = 3
        self.pattern_columns = 3
        self.distance_in_world_units = 0.1
        self.figsize = (8, 8)
        self.debug_dir = None
        self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        self.subpixel_refinement = True

        # 默认检查标定小夹爪 1: 小夹爪 2: 大夹爪
        self.calbration_type = 1
        try:
            self.calbration_type = int(sys.argv[1])
            if self.calbration_type == 2:
                print("-----开始检查大夹爪标定结果-----")
            elif self.calbration_type == 1:
                print("------开始检查小夹爪标定结果-----")
            else:
                self.calbration_type = 1
                print("!!!无效参数,使用默认参数检查小夹爪标定结果!!!")

        except:
            print("!!!使用默认夹爪检查标定结果!!!")

        if self.pattern_type in ["asymmetric_circles", "symmetric_circles"]:
            self.subpixel_refinement = False
            self.use_clustering = True
            # Setup Default SimpleBlobDetector parameters.
            self.blobParams = cv2.SimpleBlobDetector_Params()
            # Change thresholds
            self.blobParams.minThreshold = 8
            self.blobParams.maxThreshold = 255
            # Filter by Area.
            self.blobParams.filterByArea = True
            self.blobParams.minArea = 50  # minArea may be adjusted to suit for your experiment
            self.blobParams.maxArea = 10e5  # maxArea may be adjusted to suit for your experiment
            # Filter by Circularity
            self.blobParams.filterByCircularity = True
            self.blobParams.minCircularity = 0.8
            # Filter by Convexity
            self.blobParams.filterByConvexity = True
            self.blobParams.minConvexity = 0.87
            # Filter by Inertia
            self.blobParams.filterByInertia = True
            self.blobParams.minInertiaRatio = 0.01
        if self.pattern_type == "asymmetric_circles":
            self.double_count_in_column = True

    def _symmetric_world_points(self):
        x, y = np.meshgrid(range(self.pattern_columns), range(self.pattern_rows))
        prod = self.pattern_rows * self.pattern_columns
        pattern_points = np.hstack((x.reshape(prod, 1), y.reshape(prod, 1), np.zeros((prod, 1)))).astype(np.float32)
        # print(pattern_points)
        return (pattern_points)

    def _asymmetric_world_points(self):
        pattern_points = []
        if self.double_count_in_column:
            for i in range(self.pattern_rows):
                for j in range(self.pattern_columns):
                    x = j / 2
                    if j % 2 == 0:
                        y = i
                    else:
                        y = i + 0.5
                    pattern_points.append((x, y))
        else:
            for i in range(self.pattern_rows):
                for j in range(self.pattern_columns):
                    y = i / 2
                    if i % 2 == 0:
                        x = j
                    else:
                        x = j + 0.5

                    pattern_points.append((x, y))

        pattern_points = np.hstack(
            (pattern_points, np.zeros((self.pattern_rows * self.pattern_columns, 1)))).astype(
            np.float32)
        return (pattern_points)

    def _chessboard_image_points(self, img):
        found, corners = cv2.findChessboardCorners(img, (self.pattern_columns, self.pattern_rows))
        return (found, corners)

    def _circulargrid_image_points(self, img, flags, blobDetector):
        found, corners = cv2.findCirclesGrid(img, (self.pattern_columns, self.pattern_rows),
                                             flags=flags,
                                             blobDetector=blobDetector
                                             )
        return (found, corners)

    def _calc_reprojection_error(self, figure_size=(8, 8), save_dir=None):
        """
        Util function to Plot reprojection error
        """
        reprojection_error = []
        for i in range(len(self.calibration_df)):
            imgpoints2, _ = cv2.projectPoints(self.calibration_df.obj_points[i], self.calibration_df.rvecs[i],
                                              self.calibration_df.tvecs[i], self.camera_matrix, self.dist_coefs)
            temp_error = cv2.norm(self.calibration_df.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            reprojection_error.append(temp_error)
        self.calibration_df['reprojection_error'] = pd.Series(reprojection_error)
        avg_error = np.sum(np.array(reprojection_error)) / len(self.calibration_df.obj_points)
        print("The Mean Reprojection Error in pixels is:  {}".format(avg_error))

    def calibrate_camera(self,
                         img,
                         threads=1,
                         custom_world_points_function=None,
                         custom_image_points_function=None,
                         ):

        img_points = []
        obj_points = []
        working_images = []

        if self.pattern_type == "chessboard":
            pattern_points = self._symmetric_world_points() * self.distance_in_world_units

        elif self.pattern_type == "symmetric_circles":
            pattern_points = self._symmetric_world_points() * self.distance_in_world_units
            blobDetector = cv2.SimpleBlobDetector_create(self.blobParams)
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            if self.use_clustering:
                flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

        elif self.pattern_type == "asymmetric_circles":
            pattern_points = self._asymmetric_world_points() * self.distance_in_world_units
            blobDetector = cv2.SimpleBlobDetector_create(self.blobParams)
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            if self.use_clustering:
                flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

        elif self.pattern_type == "custom":
            pattern_points = custom_world_points_function(self.pattern_rows, self.pattern_columns)

        h, w = img.shape[:2]

        def process_single_image(img, size_r, size_c):
            assert w == img.shape[1] and h == img.shape[0], "All the images must have same shape"

            if self.pattern_type == "chessboard":
                found, corners = self._chessboard_image_points(img)
            elif self.pattern_type == "asymmetric_circles" or self.pattern_type == "symmetric_circles":
                found, corners = self._circulargrid_image_points(img, flags, blobDetector)

            elif self.pattern_type == "custom":
                found, corners = custom_image_points_function(img, self.pattern_rows, self.pattern_columns)
                assert corners[0] == pattern_points[
                    0], "custom_image_points_function should return a numpy array of length matching the number of control points in the image"

            if found:
                # self.working_images.append(img_path)
                if self.subpixel_refinement:
                    corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), self.term_criteria)
                else:
                    corners2 = corners.copy()
                # print(corners2)

                cv2.drawChessboardCorners(img, (size_c, size_r), corners2, found)

                cv2.imshow("dte", img)
                cv2.waitKey(1)
                return corners2
            else:
                print("Calibration board NOT FOUND")
                return None

        calibrationBoards = process_single_image(img, self.pattern_rows, self.pattern_columns)
        # print(calibrationBoards)

        working_images.append(img)
        img_points.append(calibrationBoards)
        print(pattern_points, type(pattern_points))
        obj_points.append(pattern_points)

        # combine it to a dataframe
        self.calibration_df = pd.DataFrame({"image_names": working_images,
                                            "img_points": img_points,
                                            "obj_points": obj_points,
                                            })
        self.calibration_df.sort_values("image_names")
        self.calibration_df = self.calibration_df.reset_index(drop=True)

        # calibrate the camera
        self.rms, self.camera_matrix, self.dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
            self.calibration_df.obj_points, self.calibration_df.img_points, (w, h), None, None)

        self.calibration_df['rvecs'] = pd.Series(rvecs)
        self.calibration_df['tvecs'] = pd.Series(tvecs)

        print("\nRMS:", self.rms)
        # print("camera matrix:\n", self.camera_matrix)
        # print("distortion coefficients: ", self.dist_coefs.ravel())
        # plot the reprojection error graph
        self._calc_reprojection_error(figure_size=self.figsize, save_dir=self.debug_dir)

        return calibrationBoards

    def check_calbration_data(self):
        while True:
            self.frames = self.pipeline.wait_for_frames()
            self.aligned_frames = self.align.process(self.frames)  # 深度图和彩色图对齐
            self.color_frame = self.aligned_frames.get_color_frame()  # 获取对齐后的彩色图
            self.depth_frame = self.aligned_frames.get_depth_frame()  # 获取对齐后的深度图

            # 获取彩色帧内参
            self.color_profile = self.color_frame.get_profile()
            self.cvs_profile = rs.video_stream_profile(self.color_profile)
            self.color_intrin = self.cvs_profile.get_intrinsics()
            self.color_intrin_part = [self.color_intrin.ppx, self.color_intrin.ppy, self.color_intrin.fx,
                                      self.color_intrin.fy]

            self.ppx = self.color_intrin_part[0]
            self.ppy = self.color_intrin_part[1]
            self.fx = self.color_intrin_part[2]
            self.fy = self.color_intrin_part[3]

            # 　图像转数组
            self.img_color = np.asanyarray(self.color_frame.get_data())
            self.img_depth = np.asanyarray(self.depth_frame.get_data())

            # 获取深度标尺
            self.depth_sensor = self.pipe_profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()

            # 　深度彩色外参,对齐
            self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
            self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
            self.depth_to_color_extrin = self.depth_frame.profile.get_extrinsics_to(self.color_frame.profile)

            cv2.imshow("frame", self.img_color)

            circle_point = []

            k = cv2.waitKey(1)
            if k == ord(' '):
                circle_data = self.calibrate_camera(self.img_color)

                if (circle_data[0][0] != None).all():
                    try:
                        if self.calbration_type == 1:
                            check_path = "calibration.txt"
                        else:
                            check_path = "bjj_calibration.txt"

                        calbration_data = np.loadtxt(check_path)
                        average_depth = calbration_data[6]
                    except:
                        print("Get Calibration depth data error!!!!!!!!!")
                        average_depth = 0.0
                    if average_depth != 0.0:
                        for i in range(len(circle_data)-1, -1, -1):
                            target_xy_true = [1000*((circle_data[i][0][0] - self.ppx) * average_depth / self.fx),
                                              1000*((circle_data[i][0][1] - self.ppy) * average_depth / self.fy)]
                            # target_xy_true = [1.0*(circle_data[i][0][0]), 1.0*(circle_data[i][0][1])]
                            circle_point.append(target_xy_true)
                            cv2.putText(self.img_color, str(9 - i), (int(circle_data[i][0][0])-5, int(circle_data[i][0][1])-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 0, 255])

                        # cam_data = [[-1.15915093e-02, 9.96677835e-01, 4.12389780e+02],
                        #             [1.00216761e+00, 9.23502553e-03, -2.19945838e+02]]

                        for i in range(len(circle_point)):
                            x_length = (calbration_data[0] * circle_point[i][0]) + (
                                    calbration_data[1] * circle_point[i][1] + calbration_data[2])
                            y_length = (calbration_data[3] * circle_point[i][0]) + (
                                    calbration_data[4] * circle_point[i][1] + calbration_data[5])
                            print("circle_data: ", x_length, y_length)
			print("--------------")

			for i in range(len(circle_point)):
			    x_length = circle_point[i][0]*calbration_data[7]
			    y_length = circle_point[i][1]*calbration_data[7]
			    print("circle_point: ", x_length, y_length)

            if k == ord("q") or k == 27:
                cv2.destroyAllWindows()
                self.pipeline.stop()
                break


if __name__ == "__main__":
    calibration_object = Check_Calibration_Data()
    results = calibration_object.check_calbration_data()
