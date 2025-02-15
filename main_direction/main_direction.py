import cv2
import numpy as np
import math

def calculate_main_direction(image_path):
    """
    计算输入图像切片的主方向角
    Args:
        image_path (str): 输入图像的路径
    Returns:
        float: 主方向角（单位：度，范围 [0, 180)）
    """
    # Step 1: 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Step 2: 高斯滤波
    img_filtered = cv2.GaussianBlur(img, (5, 5), 0)

    # Step 3: 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 4: 二值化
    # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

    cv2.imshow('binary', binary)
    cv2.waitKey(0)

    # Step 5: 提取非零像素坐标
    coords = np.column_stack(np.where(binary > 0))  # coords: [[y1, x1], [y2, x2], ...]

    if coords.shape[0] < 2:
        raise ValueError("Insufficient non-zero pixels to calculate direction.")

    # Step 6: 中心化像素坐标
    mean_coords = np.mean(coords, axis=0)  # 质心 [mean_y, mean_x]
    centered_coords = coords - mean_coords

    # Step 7: 计算协方差矩阵
    covariance_matrix = np.cov(centered_coords, rowvar=False)

    # Step 8: 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    max_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]  # 最大特征值对应的特征向量

    # Step 9: 计算主方向角
    angle_rad = math.atan2(max_eigenvector[0], max_eigenvector[1])  # atan2(y, x)
    angle_deg = math.degrees(angle_rad)  # 转为角度

    # 调整角度到 [0, 180) 范围
    if angle_deg < 0:
        angle_deg += 180
    elif angle_deg >= 180:
        angle_deg -= 180

    return angle_deg

# def calculate_main_direction(image_path):
#     # Step 1: 读取图像
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("无法读取图像，请检查路径")
    
    
#     cv2.imshow('image', image)
#     cv2.waitKey(0)

#     # Step 2: 高斯滤波去噪
#     blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # 5x5核大小，标准差为0
#     cv2.imshow('blurred_image', blurred_image)
#     cv2.waitKey(0)

#     # Step 3: 转换为灰度图像
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('gray_image', gray_image)
#     cv2.waitKey(0)

#     # Step 4: 二值化处理
#     # _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
#     # 使用自适应阈值
#     binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                          cv2.THRESH_BINARY, 11, 2)
#     cv2.imshow('binary_image', binary_image)
#     cv2.waitKey(0)

#     # Step 5: 获取非零像素的位置坐标
#     coords = np.column_stack(np.where(binary_image > 0))  # (y, x)坐标

#     # Step 6: 计算位置矩阵的协方差矩阵
#     mean_x, mean_y = np.mean(coords, axis=0)  # 均值
#     centered_coords = coords - np.array([mean_x, mean_y])
#     cov_matrix = np.cov(centered_coords, rowvar=False)

#     # Step 7: 计算协方差矩阵的特征值和特征向量
#     eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
#     max_eigen_index = np.argmax(eigenvalues)
#     principal_eigenvector = eigenvectors[:, max_eigen_index]

#     # Step 8: 计算主方向角（以弧度表示）
#     angle_radians = np.arctan2(principal_eigenvector[1], principal_eigenvector[0])
#     angle_degrees = np.degrees(angle_radians)  # 转换为角度

#     # 输出主方向角
#     return angle_degrees

# 示例：使用图像路径
image_path = 'C:\\Users\\14398\\Desktop\\shandong13.tif'  # 替换为实际图像路径
angle = calculate_main_direction(image_path)
print(f"主方向角：{angle:.2f} 度")