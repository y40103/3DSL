# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 05:01:52 2020

@author: chung-chi
"""
import numpy as np
import cv2
import numba as nb
import time
import pickle
import math

imgSets_dir = "./dataset/"

bit = 9


# 二值化
class Binarization(object):
    bit = bit
    dataSet_dir = imgSets_dir
    width = cv2.imread(imgSets_dir + "0/0.jpg", 0).shape[1]
    height = cv2.imread(imgSets_dir + "0/0.jpg", 0).shape[0]
    binarization_imgSet = np.zeros([bit + 1, height, width], dtype="float32")
    binarization_imgSet.fill(np.nan)

    @staticmethod
    # @nb.njit
    def get_bounder(bool_result, index_group):  # find the smallest unit of graycode pattern
        target_pixel = []
        for i in range(index_group.shape[1]):
            left = bool_result[index_group[0][i], index_group[1][i] - 1]
            right = bool_result[index_group[0][i], index_group[1][i] + 1]
            if left == False and right == True:
                target_pixel.append([index_group[0][i], index_group[1][i]])
            elif left == True and right == False:
                target_pixel.append([index_group[0][i], index_group[1][i] + 1])
        return target_pixel.copy()

    def __init__(self, imgFold_order):
        self.imgFold_order = imgFold_order
        self.work_area = None
        self.target_pixel = []

    def get_data_result(self):  # import image and then return binariztion result
        for _ in range(0, 2 * (Binarization.bit + 1), 2):
            font_img = cv2.imread(imgSets_dir + str(self.imgFold_order) + r"/" + str(_) + ".jpg", 0)
            back_img = cv2.imread(imgSets_dir + str(self.imgFold_order) + r"/" + str(_ + 1) + ".jpg", 0)
            result = self._binarization(_, font_img, back_img)
        self.target_pixel = np.array(self.target_pixel).T

    def _binarization(self, order, font_img, back_img):  # 二值化
        result = font_img > back_img + 3
        if order == 0:
            self.work_area = np.argwhere(result == True).T
            Binarization.binarization_imgSet[0][self.work_area[0], self.work_area[1]] = 0
        else:
            work_area = np.argwhere(result == True).T
            result = np.where(result == True, 1, 0).astype("float32")
            result += Binarization.binarization_imgSet[0]  # 利用nan特性 將排除非工作區 防止物件的變緣被非pattern邊界誤判
            self._cal_get_bounder(result, work_area)
            Binarization.binarization_imgSet[order // 2] = result.copy()

    def _cal_get_bounder(self, result, work_area):
        valid_unit = Binarization.get_bounder(result, work_area)
        self.target_pixel.extend(valid_unit)


class Graycode(object):
    _binArray = np.fromiter([2 ** x for x in range(8, -1, -1)], "float32")
    _demical_result = np.zeros([Binarization.height, Binarization.width])

    @staticmethod
    # @nb.njit
    # decoding graycode image to demical
    def _decoding_to_demical(binarization_result, work_area, demical_result, binArray, bin_temp):
        shape = work_area.shape[1]
        for _ in range(shape):  # process the pixel in the pattern bounder
            bin_temp[:] = 0
            for i in range(bit):
                if i == 0:  # 第一項
                    bin_temp[i] = binarization_result[1:, work_area[0, _], work_area[1, _]][i]
                    temp = binarization_result[1:, work_area[0, _], work_area[1, _]][i]
                else:
                    bin_temp[i] = temp ^ binarization_result[1:, work_area[0, _], work_area[1, _]][i]
                    temp = temp ^ binarization_result[1:, work_area[0, _], work_area[1, _]][i]  # 修過
            demical_result[work_area[0, _], work_area[1, _]] = bin_temp.dot(binArray)
        return demical_result

    def __init__(self, binarization_result, work_area):
        self.binarization_result = np.nan_to_num(binarization_result).astype("uint8")
        self.work_area = work_area

    def cal_decoding_demical(self):
        binarization_result = self.binarization_result
        work_area = self.work_area
        demical_result = Graycode._demical_result
        binArray = Graycode._binArray
        bin_temp = np.zeros(9).astype("float32")  # 無法在numba中 轉換numpy dtype

        demical_result = Graycode._decoding_to_demical(binarization_result, work_area, demical_result, binArray,
                                                       bin_temp)
        Graycode.demical_result = demical_result.copy()


class Light_engine(object):
    def __init__(self, light_width, light_height, x_t_camera, z_t_camera, x_PxNum, y_PxNum):
        self.dmd_width = (light_width / light_height) * z_t_camera
        self.dmd_height = (y_PxNum / x_PxNum) * self.dmd_width
        self.unit_width = self.dmd_width / 480
        self.pixel_light_tangent = {x: ((x * light_width / 480) - light_width / 2) / light_height for x in
                                    range(0, 480 + 1)}
        self._x_t_camera = x_t_camera
        self._z_t_camera = z_t_camera

    def base_line(self, which_strip):
        if which_strip != 240:
            return -self._x_t_camera + self.pixel_light_tangent[which_strip] * self._z_t_camera
        else:
            return -self._x_t_camera  # stereo 校正結果 為 光機為原點 與相機的相對位置 本系統以相機為原點 因此x值為負數


class Camera(object):
    @staticmethod
    # @nb.njit
    def _world_xyz(imgColumn, imgRow, _xPixelsize, _yPixelsize, _inverse_to_world_coordination):
        xyz = _inverse_to_world_coordination.dot(np.array([imgColumn * _xPixelsize, imgRow * _yPixelsize, 1]))
        return xyz

    def __init__(self, img_width, img_height, xPixelsize, yPixelsize, x_sensorOffset, y_sensorOffset, focal,
                 rotation_matrix, translation):
        self.sensor_offset = np.array(
            [1, 0, x_sensorOffset * xPixelsize, 0, 1, y_sensorOffset * yPixelsize, 0, 0, 1]).reshape([3, 3])
        self._focal = np.array([focal, 0, 0, 0, focal, 0, 0, 0, 1]).reshape([3, 3])
        self._xPixelsize = xPixelsize
        self._yPixelsize = yPixelsize
        self._rotation = rotation_matrix
        self._translation = translation
        self._width = img_width
        self._height = img_height
        self.inverse_to_world_coordination = np.linalg.inv(self.sensor_offset.dot(self._focal.dot(self._rotation)))
        # inverse 分配率 A^-1 * B^-1 = (BA)^-1
        # dot 順序 A*B*C  A.dot(B.dot(C))

    def img_to_world_xyz(self, imgColumn, imgRow):
        _xPixelsize = self._xPixelsize
        _yPixelsize = self._yPixelsize
        _inverse_to_world_coordination = self.inverse_to_world_coordination
        xyz = Camera._world_xyz(imgColumn, imgRow, _xPixelsize, _yPixelsize, _inverse_to_world_coordination)
        return xyz


class Structure_light(object):
    adress = None

    def __new__(cls, order, camera_instance, light_instance):
        if cls.adress is None:
            cls.adress = super(Structure_light, cls).__new__(cls)
        return cls.adress

    def __init__(self, order, camera_instance, light_instance):
        self.order = order
        self.Bin_temp = Binarization(order)
        self.Bin_temp.get_data_result()
        self.Gray_temp = Graycode(Binarization.binarization_imgSet, self.Bin_temp.target_pixel)
        self.Gray_temp.cal_decoding_demical()
        self.camera_instance = camera_instance
        self.light_instance = light_instance

    def show_result(self):
        g = np.zeros([Binarization.height, Binarization.width]).astype("uint8")
        g[self.Bin_temp.target_pixel[0], self.Bin_temp.target_pixel[1]] = 255
        cv2.namedWindow("window_white" + str(self.order), cv2.WINDOW_NORMAL)
        cv2.imshow("window_white" + str(self.order), g)

    def graycode_result(self):
        return Graycode.demical_result.copy()

    def create_axis_system_rotationMatrix(self, axis_center, p1, p2, rotation_angle):
        mid_point = axis_center.copy()

        # 產生旋轉矩陣
        rotation_axis_matrix = [math.cos(rotation_angle * (math.pi / 180)), -math.sin(rotation_angle * (math.pi / 180)),
                                0, math.sin(rotation_angle * (math.pi / 180)),
                                math.cos(rotation_angle * (math.pi / 180)), 0, 0, 0, 1]
        rotation_axis_matrix = np.reshape(rotation_axis_matrix, (3, 3))

        # 以中心軸為原點 平面上兩向量
        basis1 = np.array(p1) - mid_point
        basis2 = np.array(p2) - mid_point

        # 法向量
        normal_vector = np.cross(basis2, basis1)

        ##orthogonal basis 兩向量 正交化
        v1 = basis1.copy()
        v2 = basis2 - (basis2.dot(v1.T)) / (np.linalg.norm(v1) ** 2) * v1

        # 標準化
        x2 = v1 / np.linalg.norm(v1)
        y2 = v2 / np.linalg.norm(v2)
        z2 = normal_vector / np.linalg.norm(normal_vector)

        new_coordiantion_system = np.vstack((x2, y2, z2)).T  # 新軸系 vector space orthornomal

        return new_coordiantion_system, rotation_axis_matrix

    def cal_tri(self, p1, p2, center):
        new_axis_system, axis_rotation_matrix = self.create_axis_system_rotationMatrix(center, p1, p2, self.order * 0)
        # 旋轉盤軸系
        graycode_result = self.graycode_result().astype("int32")

        graycode_result[1400:, :] = 0
        graycode_result[graycode_result < 150] = 0
        # 過濾轉盤位置點雲產生

        world_coordination = []
        for _ in range(self.Bin_temp.target_pixel.shape[1]):
            img_row = self.Bin_temp.target_pixel[0][_]
            img_column = self.Bin_temp.target_pixel[1][_]
            strip = graycode_result[img_row, img_column]
            if strip != 240 and strip <= 480:
                tan = 1 / self.light_instance.pixel_light_tangent[strip]
            else:
                continue
            c_xyz = self.camera_instance.img_to_world_xyz(img_column, img_row)
            baseline = self.light_instance.base_line(strip)
            zc = (-tan * baseline) / ((-tan * c_xyz[0]) + c_xyz[2])
            if zc <= 120 or zc >= 200:
                continue
            world_coordination.append(zc * c_xyz)
        world_coordination = np.array(world_coordination)

        w_mean = world_coordination.mean(axis=0)
        w_std = world_coordination.std(axis=0) * 1.5
        w_std_distance = np.sum(pow(w_std ** 2, 0.5))
        result = np.zeros(world_coordination.shape[0]).astype(bool)
        for i in range(world_coordination.shape[0]):
            temp_distance = np.sum(pow((world_coordination[i] - w_mean) ** 2, 0.5))
            if temp_distance > w_std_distance:
                result[i] = False
            else:
                result[i] = True
        # 濾掉decoding誤判造成的離散的點雲

        world_coordination = world_coordination[result]

        # 旋轉盤角度
        world_coordination = (world_coordination - center).T
        world_coordination = np.linalg.inv(new_axis_system).dot(world_coordination)
        # 基變換
        world_coordination = (axis_rotation_matrix.dot(world_coordination)).T
        # 旋轉盤 角度變換
        world_coordination[:, 1] = world_coordination[:, 1] * -1

        np.savetxt(f"scan_object{self.order}.xyz", world_coordination)
        return world_coordination


if __name__ == "__main__":

    def dtr(angle):
        return angle * np.pi / 180


    def rtd(rad):
        return rad * 180 / np.pi


    with open(imgSets_dir + "stereo_rotation.pkl", "rb") as f:
        rotation = np.linalg.inv(pickle.load(f))

    with open(imgSets_dir + "stereo_translation.pkl", "rb") as f:
        translation = pickle.load(f)
    # camera R,T between light engine

    with open(imgSets_dir + "axis_center.pkl", "rb") as f:
        center = pickle.load(f)

    with open(imgSets_dir + "p1.pkl", "rb") as f:
        p1 = pickle.load(f)

    with open(imgSets_dir + "p2.pkl", "rb") as f:
        p2 = pickle.load(f)
        # rotation axis coordinate system

    s = time.time()
    print("Start calculating")
    nvm = Light_engine(240, 487.5, translation[0], translation[2], 1920, 1080)
    camera_500 = Camera(2592, 1944, 0.0022, 0.0022, 2592 // 2, 1944 // 2, 12, rotation, translation)
    for i in range(4):
        s_temp = Structure_light(i, camera_500, nvm)
        world_coordination = s_temp.cal_tri(center, p1, p2)
    se = time.time()
    print("Finish")

