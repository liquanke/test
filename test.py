# 测试暂存代码
import os
import shutil
import torch
import cv2

def full_image_split(images, split_withd, r, model):
    results = torch.zeros((1, 0, 7), device=next(model.parameters()).device)
    h, w = images.shape[0], images.shape[1]
    row = int((h-r)/(split_withd-r))
    col  = int((w-r)/(split_withd-r))
     # print(h, w, row, col)

    for i in range(row+1):
         for j in range(col+1):
             #-------------------------------------------------#
             #   裁剪的图像不在最后一列和最后一行无需过多考虑
            if i < row and j < col:
                image = images[i*(split_withd-r):i*(split_withd- r)+split_withd, j*(split_withd- r):j*(split_withd- r)+split_withd] 
             #-------------------------------------------------#
             #   裁剪的图像在最后一行
            elif i == row and j < col:
                image = images[h-split_withd: h, j*(split_withd- r):j*(split_withd- r)+split_withd]

             #-------------------------------------------------#
             #   裁剪的图像在最后一列
            elif i < row and j == col:
                image = images[i*(split_withd- r):i*(split_withd- r)+split_withd, w-split_withd:w]

             #-------------------------------------------------#
             #   裁剪的图像在右下角
            elif i == row and j == col:
                image = images[h-split_withd:h, w-split_withd:w]

            with torch.no_grad():
                 # print(image.shape)
                image = torch.from_numpy(image).unsqueeze(dim=0).permute(0, 3, 1, 2).to(next(model.parameters()).device)
                pred = model(image)[0] # [bs, 特征点个数, 预测结果], 预测结果: x y w h ob cls, , 其中 xywh 不是归一化的值
                # print(pred.shape) 
            if i < row and j < col: # 不在最后一行和最后一列
                pred[:, :, 0] += j*(split_withd-r)
                pred[:, :, 1] += i*(split_withd-r)
            elif i == row and j < col: #   裁剪的图像在最后一行
                pred[:, :, 0] += j*(split_withd- r)
                pred[:, :, 1] += h-split_withd
            elif i < row and j == col: #   裁剪的图像在最后一列
                pred[:, :, 0] += w-split_withd

# 将与标签文件同名的源图像文件移到指定文件夹
def label_sanme_img ():
    label_path = 'C:\\Users\\14398\Desktop\\buaa_cfpp\\buaa_addcfpp2.0_gcn\\Annotations_gcn2.0'
    img_path = 'C:\\Users\\14398\\Desktop\\buaa_cfpp\\buaa_addcfpp2.0_gcn\\JPEGImages'
    save_path = 'C:\\Users\\14398\\Desktop\\buaa_cfpp\\buaa_addcfpp2.0_gcn\\JPEGImages_gcn'
    os.makedirs(save_path, exist_ok=True)

    filelist = os.listdir(label_path)
    for file_name in filelist:
        file_name = file_name.replace('.xml', '.jpg')
        img_origin = os.path.join(img_path, file_name)
        img_dist = os.path.join(save_path, file_name)
        shutil.move(img_origin,img_dist)
    pass

# 测试cv2读取图片通道顺序
def cv2_test():
    jpg_path = 'C:\\Users\\14398\Desktop\\0_shandong333.jpg'
    jpg_data = cv2.imread(jpg_path)
    print(jpg_data.shape)
    cv2.imwrite('C:\\Users\\14398\Desktop\\0_shandong3333.jpg', jpg_data)

if __name__ == '__main__':
    # label_sanme_img()
    cv2_test()