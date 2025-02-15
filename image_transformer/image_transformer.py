import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
import cv2
from scipy.fftpack import dct, idct
from scipy.fft import fft2, ifft2, fftshift

# 生成一张示例图像（使用随机噪声生成）

img_path = 'C:\\Users\\14398\\Desktop\\res_pos'
filelist = os.listdir(img_path)
file_path = os.path.join(img_path, filelist[1])

image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)

print(image.shape)

# # 进行小波变换，使用离散小波变换（DWT）
# coeffs2 = pywt.dwt2(image, 'haar')  # 'haar' 是一种常见的小波
# LL, (LH, HL, HH) = coeffs2  # LL是低频部分，LH、HL、HH是高频部分

# # 显示小波变换后的结果
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# axes[0, 0].imshow(LL, cmap='gray')
# axes[0, 0].set_title('LL (Low-Low)')
# axes[0, 1].imshow(LH, cmap='gray')
# axes[0, 1].set_title('LH (Low-High)')
# axes[1, 0].imshow(HL, cmap='gray')
# axes[1, 0].set_title('HL (High-Low)')
# axes[1, 1].imshow(HH, cmap='gray')
# axes[1, 1].set_title('HH (High-High)')
# plt.show()

# # 对图像进行DCT变换
# dct_image = dct(dct(image.T, norm='ortho').T, norm='ortho')

# # 显示变换后的图像
# plt.imshow(np.abs(dct_image), cmap='gray')
# plt.title('DCT of Image')
# plt.show()

# # 进行逆变换（IDCT）
# idct_image = idct(idct(dct_image.T, norm='ortho').T, norm='ortho')
# print(idct_image)

# # 显示逆变换后的图像
# plt.imshow(idct_image, cmap='gray')
# plt.title('Inverse DCT of Image')
# plt.show()

# 进行二维傅里叶变换
f_transform = fft2(image)

# 移动零频率成分到频谱中心
f_transform_shifted = fftshift(f_transform)

# 显示傅里叶变换的幅度谱
plt.imshow(np.log(np.abs(f_transform_shifted) + 1), cmap='gray')
plt.title('Fourier Transform (Magnitude Spectrum)')
plt.show()

# 进行逆傅里叶变换
image_reconstructed = np.abs(ifft2(f_transform))

# 显示重建后的图像
plt.imshow(image_reconstructed, cmap='gray')
plt.title('Inverse Fourier Transform')
plt.show()

