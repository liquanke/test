import os
import cv2

# def full_image_split(images, split_withd, r, model):
#     results = torch.zeros((1, 0, 7), device=next(model.parameters()).device)
#     h, w = images.shape[0], images.shape[1]
#     row  = int((h-r)/(split_withd-r))
#     col = int((w-r)/(split_withd-r))
#     # print(h, w, row, col)
#     for i in range(row+1):
#         for j in range(col+1):
#             #-------------------------------------------------#
#             #   裁剪的图像不在最后一列和最后一行无需过多考虑
#             if i < row and j < col:
#                 image = images[i*(split_withd-r):i*(split_withd- r)+split_withd, j*(split_withd- r):j*(split_withd- r)+split_withd] 

#             #-------------------------------------------------#
#             #   裁剪的图像在最后一行
#             elif i == row and j < col:
#                 image = images[h-split_withd: h, j*(split_withd- r):j*(split_withd- r)+split_withd]

#             #-------------------------------------------------#
#             #   裁剪的图像在最后一列
#             elif i < row and j == col:
#                 image = images[i*(split_withd- r):i*(split_withd- r)+split_withd, w-split_withd:w]

#             #-------------------------------------------------#
#             #   裁剪的图像在右下角
#             elif i == row and j == col:
#                 image = images[h-split_withd:h, w-split_withd:w]

#             with torch.no_grad():
#                 # print(image.shape)
#                 image = torch.from_numpy(image).unsqueeze(dim=0).permute(0, 3, 1, 2).to(next(model.parameters()).device)
#                 pred = model(image)[0] # [bs, 特征点个数, 预测结果], 预测结果: x y w h ob cls, , 其中 xywh 不是归一化的值
#                 # print(pred.shape)

#             if i < row and j < col: # 不在最后一行和最后一列
#                 pred[:, :, 0] += j*(split_withd-r)
#                 pred[:, :, 1] += i*(split_withd-r)
#             elif i == row and j < col: #   裁剪的图像在最后一行
#                 pred[:, :, 0] += j*(split_withd- r)
#                 pred[:, :, 1] += h-split_withd
#             elif i < row and j == col: #   裁剪的图像在最后一列
#                 pred[:, :, 0] += w-spli

# 批量裁剪
def im_split():
    # 读取图片
    split_withd = 640
    overlap = 128
    image_dir = "D:\\shandong\\shandong227.tif"
    filelist = os.listdir(image_dir)
    for img_file in filelist:
        img_path = os.path.join(image_dir, img_file)
        filename = os.path.basename(img_path)
        images = cv2.imread(img_path, 1)  # color
        h, w = images.shape[0], images.shape[1]
        row  = int((h-overlap)/(split_withd-overlap))
        col = int((w-overlap)/(split_withd-overlap))
        # 裁剪
        for i in range(row+1):
            for j in range(col+1):
                #-------------------------------------------------#
                #   裁剪的图像不在最后一列和最后一行无需过多考虑
                if i < row and j < col:
                    image = images[i*(split_withd-overlap):i*(split_withd- overlap)+split_withd, j*(split_withd- overlap):j*(split_withd- overlap)+split_withd]

                #-------------------------------------------------#
                #   裁剪的图像在最后一行
                elif i == row and j < col:
                    image = images[h-split_withd: h, j*(split_withd- overlap):j*(split_withd- overlap)+split_withd]

                #-------------------------------------------------#
                #   裁剪的图像在最后一列
                elif i < row and j == col:
                    image = images[i*(split_withd- overlap):i*(split_withd- overlap)+split_withd, w-split_withd:w]

                #-------------------------------------------------#
                #   裁剪的图像在右下角
                elif i == row and j == col:
                    image = images[h-split_withd:h, w-split_withd:w]
                
                # 保存图像至指定文件夹
                cv2.imwrite("C:\\Users\\14398\\Desktop\\shandong227\\"+filename+"_"+str(i)+'_'+str(j)+'.jpg', image)        

def im_split_test():
    # 读取图片
    split_withd = 640
    overlap = 160
    image_path = "D:\\shandong\\shandong25.tif"
    filename = 'shandong25'
    images = cv2.imread(image_path, 1)  # color
    print(images.shape)

    h, w = images.shape[0], images.shape[1]
    row  = int((h-overlap)/(split_withd-overlap))
    col = int((w-overlap)/(split_withd-overlap))

    # 裁剪
    for i in range(row+1):
        for j in range(col+1):
            #-------------------------------------------------#
            #   裁剪的图像不在最后一列和最后一行无需过多考虑
            if i < row and j < col:
                image = images[i*(split_withd-overlap):i*(split_withd- overlap)+split_withd, j*(split_withd- overlap):j*(split_withd- overlap)+split_withd]

            #-------------------------------------------------#
            #   裁剪的图像在最后一行
            elif i == row and j < col:
                image = images[h-split_withd: h, j*(split_withd- overlap):j*(split_withd- overlap)+split_withd]

            #-------------------------------------------------#
            #   裁剪的图像在最后一列
            elif i < row and j == col:
                image = images[i*(split_withd- overlap):i*(split_withd- overlap)+split_withd, w-split_withd:w]

            #-------------------------------------------------#
            #   裁剪的图像在右下角
            elif i == row and j == col:
                image = images[h-split_withd:h, w-split_withd:w]
            
            # 保存图像至指定文件夹
            cv2.imwrite("C:\\Users\\14398\\Desktop\\shandong25\\"+filename+"_"+str(i)+'_'+str(j)+'.jpg', image)


if __name__ == "__main__":
    
    im_split_test()
    # im_split()

