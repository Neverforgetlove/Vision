# coding=utf-8
import cv2
import sys
import numpy as np
import pandas as pd
import math
import pyrealsense2 as rs

class Calibration_Robot:
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

        print(rs.quaternion)

        self.pattern_type = "symmetric_circles"  # 标定板类型-对称圆
        self.pattern_rows = 3  # 一行的圆的个数
        self.pattern_columns = 3  # 一列的圆的个数
        self.distance_in_world_units = 0.1  # 相机高度
        self.figsize = (8, 8)
        self.debug_dir = None
        self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        self.subpixel_refinement = True

        # 默认标定小夹爪 1: 小夹爪 2: 大夹爪
        self.calbration_type = 1
        try:
            self.calbration_type = int(sys.argv[1])
            if self.calbration_type == 2:
                print("-----开始标定大夹爪-----")
            elif self.calbration_type == 1:
                print("------开始标定小夹爪-----")
            else:
                self.calbration_type = 1
                print("!!!无效参数,使用默认参数标定小夹爪!!!")
        except:
            print("!!!使用默认夹爪标定!!!")

        # 斑点检测参数
        if self.pattern_type in ["asymmetric_circles", "symmetric_circles"]:
            self.subpixel_refinement = False
            self.use_clustering = True
            #  设置斑点检测
            self.blobParams = cv2.SimpleBlobDetector_Params()
            # 变换阈值
            self.blobParams.minThreshold = 8
            self.blobParams.maxThreshold = 255
            # 面积阈值
            self.blobParams.filterByArea = True
            self.blobParams.minArea = 50
            self.blobParams.maxArea = 10e5
            # 圆度过滤
            self.blobParams.filterByCircularity = True
            self.blobParams.minCircularity = 0.8
            # 凹凸过滤
            self.blobParams.filterByConvexity = True
            self.blobParams.minConvexity = 0.87
            # 惯性过滤
            self.blobParams.filterByInertia = True
            self.blobParams.minInertiaRatio = 0.01
        if self.pattern_type == "asymmetric_circles":
            self.double_count_in_column = True

    # 世界坐标
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

    # 检测标定板斑点
    def _chessboard_image_points(self, img):
        found, corners = cv2.findChessboardCorners(img, (self.pattern_columns, self.pattern_rows))
        return (found, corners)

    def _circulargrid_image_points(self, img, flags, blobDetector):
        found, corners = cv2.findCirclesGrid(img, (self.pattern_columns, self.pattern_rows),
                                             flags=flags,
                                             blobDetector=blobDetector
                                             )
        return (found, corners)

    # 计算误差
    def _calc_reprojection_error(self, figure_size=(8, 8), save_dir=None):
        reprojection_error = []
        for i in range(len(self.calibration_df)):
            imgpoints2, _ = cv2.projectPoints(self.calibration_df.obj_points[i], self.calibration_df.rvecs[i],
                                              self.calibration_df.tvecs[i], self.camera_matrix, self.dist_coefs)
            temp_error = cv2.norm(self.calibration_df.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            reprojection_error.append(temp_error)
        self.calibration_df['reprojection_error'] = pd.Series(reprojection_error)
        avg_error = np.sum(np.array(reprojection_error)) / len(self.calibration_df.obj_points)
        print("The Mean Reprojection Error in pixels is:  {}".format(avg_error))

    # 九点标定
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

        # 组成数据帧
        self.calibration_df = pd.DataFrame({"image_names": working_images,
                                            "img_points": img_points,
                                            "obj_points": obj_points,
                                            })
        self.calibration_df.sort_values("image_names")
        self.calibration_df = self.calibration_df.reset_index(drop=True)

        # 校准相机
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

    # 计算仿射变换矩阵
    def compute_calibration_data(self, circle_point, robot_point):
        cam_data = cv2.estimateRigidTransform(circle_point, robot_point, True)
        print(cam_data)
        print("-------------")
        return cam_data

    # 计算标定误差
    def compute_calibration_error(self, calibration_data, circle_point, robot_point):
        x_dis = 0.0
        y_dis = 0.0
        all_dis = 0.0
        calibration_again_circle = []
        calibration_again_robot = []
        for i in range(len(circle_point)):
            t_rx = (calibration_data[0][0] * circle_point[i][0]) + (
                    calibration_data[0][1] * circle_point[i][1] + calibration_data[0][2])
            t_ry = (calibration_data[1][0] * circle_point[i][0]) + (
                    calibration_data[1][1] * circle_point[i][1] + calibration_data[1][2])

            x_dis += abs(t_rx - robot_point[i][0])
            y_dis += abs(t_ry - robot_point[i][1])
            if abs(t_rx - robot_point[i][0]) < 0.1 and abs(abs(t_ry - robot_point[i][1])) < 0.15:
                calibration_again_circle.append([circle_point[i][0], circle_point[i][1]])
                calibration_again_robot.append([robot_point[i][0], robot_point[i][1]])
            all_dis += (abs(x_dis) + abs(y_dis))
            print("Camlibration dis: ", t_rx - robot_point[i][0], t_ry - robot_point[i][1])

        print("x_dis: ", x_dis / 9)
        print("y_dis: ", y_dis / 9)
        print("all_dis: ", all_dis / 18)
        return calibration_again_circle, calibration_again_robot

    # 计算像素比
    def compute_pixel_ratio(self, real_point, img_point):
        all_pixel_ratio = 0.0
        pixel_ratio_dis = 0.0
        for i in range(len(real_point) - 1):
            real_length = math.sqrt(
                (real_point[i][0] - real_point[i + 1][0]) ** 2 + (real_point[i][1] - real_point[i + 1][1]) ** 2
            )

            img_length = math.sqrt(
                (img_point[i][0] - img_point[i + 1][0]) ** 2 + (img_point[i][1] - img_point[i + 1][1]) ** 2
            )
            all_pixel_ratio += img_length / real_length
        all_pixel_ratio /= 9
        print("all_pixel_ratio: ", all_pixel_ratio)

        # 验证像素比
        for i in range(len(img_point) - 1):
            current_img_length = math.sqrt(
                (img_point[i][0] - img_point[i + 1][0]) ** 2 + (img_point[i][1] - img_point[i + 1][1]) ** 2
            )
            current_real_length = current_img_length / all_pixel_ratio
            if i == 2 or i == 5:
               # pixel_ratio_dis += abs(math.sqrt(200) - current_real_length)
                pass
            else:
                print(current_real_length)
                pixel_ratio_dis += abs(20 - current_real_length)
                pixel_ratio_dis /= (20*6)
                all_pixel_ratio += pixel_ratio_dis
        return all_pixel_ratio

    def calibration_camera(self):

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

            k = cv2.waitKey(1)
            if k == ord(' '):
                circle_center_data = self.calibrate_camera(self.img_color)
                circle_point = []
                average_depth = 0.0
                count = 9
                if(circle_center_data[0][0] != None).all():
                    for i in range(len(circle_center_data)-1, -1, -1):
                        depth_data = self.depth_frame.get_distance(
                            circle_center_data[i][0][0], circle_center_data[i][0][1])
                        print(depth_data)
                        if depth_data != 0.0:
                            average_depth += depth_data
                        else:
                            count -= 1
                    average_depth /= count
                    print("average_depth:", average_depth)
                    for i in range(len(circle_center_data)-1, -1, -1):
                        target_real_point = [1000*((circle_center_data[i][0][0] - self.ppx) * average_depth / self.fx),
                                          1000*((circle_center_data[i][0][1] - self.ppy) * average_depth / self.fy)]

                        circle_point.append(target_real_point)

                        cv2.putText(self.img_color, str(9 - i),
                                    (int(circle_center_data[i][0][0])-5, int(circle_center_data[i][0][1] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 0, 255])

                    cv2.imshow("draw", self.img_color)

                    # 读取机械臂坐标点
                    try:
                        if self.calbration_type == 1:
                            robot_point_file = "robot_point.txt"
                        else:
                            robot_point_file = "bjj_robot_point.txt"
                        robot_point = np.loadtxt(robot_point_file)
                    except:
                        print("!!!!!未读取到对应数据,请检查并重新启动标定!!!!!")
                        cv2.destroyAllWindows()
                        self.pipeline.stop()
                        sys.exit(1)

                    # 圆心坐标转数组
                    circle_point_narray = np.array(circle_point)

                    # 标定像素比
                    pixel_ratio = self.compute_pixel_ratio(circle_point_narray,robot_point)
                    print("pixel_ration is :", pixel_ratio)

                    # 初次标定
                    cal_data = self.compute_calibration_data(circle_point_narray, robot_point)
                    if self.calbration_type == 1:
                        save_path = "calibration.txt"
                    else:
                        save_path = "bjj_calibration.txt"

                    with open(save_path, "w+") as f:
                        for i in range(len(cal_data)):
                            for j in range(len(cal_data[i])):
                                f.write(str(cal_data[i][j]) + "\n")
                        f.write(str(average_depth+0.005)+ "\n")
                        f.write(str(pixel_ratio))
                    print("------------first time calibration success!!!!")
                    try:
                        if (cal_data != None).all():
                            # 计算误差, 挑选好的点位再次标定
                            new_circle_point, new_robot_point = self.compute_calibration_error(cal_data, circle_point_narray, robot_point)
                        print("---------calibration again---------")
                        # 再次标定
                        print("calibration again data :", new_circle_point, new_robot_point)
                        new_cal_data = self.compute_calibration_data(np.array(new_circle_point), np.array(new_robot_point))
                        if (new_cal_data != None).all():
                            # 计算误差
                            self.compute_calibration_error(new_cal_data, circle_point_narray, robot_point)

                            if self.calbration_type == 1:
                                save_path = "calibration.txt"
                            else:
                                save_path = "bjj_calibration.txt"

                            with open(save_path, "w+") as f:
                                for i in range(len(new_cal_data)):
                                    for j in range(len(new_cal_data[i])):
                                        f.write(str(new_cal_data[i][j]) + "\n")
                                f.write(str(average_depth+0.005)+ "\n")
                                f.write(str(pixel_ratio))
                            print("calibration success!!!!")
                    except:
                        print("calibration again error, you can try again or save first time data!")
                else:
                    print("Calibration Robot Error")

            if k == ord("q") or k == 27:
                cv2.destroyAllWindows()
                self.pipeline.stop()
                break


if __name__ == "__main__":
    calibration_object = Calibration_Robot()
    results = calibration_object.calibration_camera()
