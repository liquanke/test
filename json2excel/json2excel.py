import json
import gdal
import pandas as pd
import numpy as np
import os

# 读取 TIFF 图像的地理坐标系信息
def get_geotransform(tif_path):
    dataset = gdal.Open(tif_path)
    transform = dataset.GetGeoTransform()  # 获取仿射变换矩阵
    crs = dataset.GetProjection()  # 获取坐标参考系（CRS）
    return transform, crs

# 将像素坐标转换为经纬度
def pixel_to_geo(transform, x, y):
    lon = transform[0] + x * transform[1] + y * transform[2]
    lat = transform[3] + x * transform[4] + y * transform[5]
    return lon, lat

# 将目标框的像素坐标转换为经纬度
def convert_bbox_to_geo(transform, bbox):
    x_min, y_min, x_max, y_max = bbox
    lon_min, lat_min = pixel_to_geo(transform, x_min, y_min)
    lon_max, lat_max = pixel_to_geo(transform, x_max, y_max)
    
    # 计算目标框的中心坐标（经纬度）
    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2
    
    return lon_center, lat_center

# 从 JSON 文件中读取检测结果
def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# 保存结果到 csv 文件
def save_to_csv(output_data, output_path):
    # df = pd.DataFrame(output_data, columns=["Filename", "Label", "Longitude", "Latitude", "Score"])
    df = pd.DataFrame(output_data, columns=["Filename", "Label", "Longitude", "Latitude", "x_min", "y_min", "x_max", "y_max", "Score"])
    df.to_csv(output_path, index=False)

# 主函数，整合所有步骤
def process_inference_results(tif_path, json_path, output_excel_path):
    # 读取 TIFF 图像的地理信息
    transform, _ = get_geotransform(tif_path)

    # 读取 JSON 文件中的检测结果
    detection_data = read_json(json_path)

    # 提取文件名
    filename = 'shandong25'

    # 存储输出数据
    output_data = []

    # 遍历 JSON 中的检测结果，转换每个目标框为经纬度
    for result in detection_data:
        label = result['label']
        bbox = result['bbox']
        score = result['score']

        # 转换目标框为经纬度
        lon_center, lat_center = convert_bbox_to_geo(transform, bbox)

        # 汇总数据
        output_data.append([filename, label, lon_center, lat_center, score])

    # 导出为 csv 文件
    save_to_csv(output_data, output_excel_path)
    print(f'Results saved to {output_excel_path}')

# 文件夹内文件批量转换
def all_process_inference_results(tif_path, json_path, output_excel_path):
    
    json_filelist = os.listdir(json_path)
    
    # 存储输出数据
    output_data = []
    # 便利文件夹所有文件进行转换
    for filename in json_filelist:
        json_file = os.path.join(json_path,filename)
        tif_file = os.path.join(tif_path,filename.replace('.json', '.tif'))
        
        # 读取 JSON 文件中的检测结果
        detection_data = read_json(json_file)
        if(len(detection_data)==0):
            continue
        
        # 读取 TIFF 图像的地理信息
        transform, _ = get_geotransform(tif_file)

        # 遍历 JSON 中的检测结果，转换每个目标框为经纬度
        for result in detection_data:
            label = result['label']
            bbox = result['bbox']
            score = result['score']

            # 保存框的位置
            x_min, y_min, x_max, y_max = bbox

            # 转换目标框为经纬度
            lon_center, lat_center = convert_bbox_to_geo(transform, bbox)

            # 汇总数据
            # output_data.append([filename, label, lon_center, lat_center, score])
            output_data.append([filename, label, lon_center, lat_center,  x_min, y_min, x_max, y_max, score])
    # 导出为 csv 文件
    save_to_csv(output_data, output_excel_path)
    print(f'Results saved to {output_excel_path}')

if __name__ == '__main__':
    # 单个文件的转换
    # 调用主函数处理数据
    # tif_path = 'C:\\Users\\14398\\Desktop\\shandong\\shandong\\shandong25.tif'  # TIFF 图像路径
    # json_path = 'C:\\Users\\14398\\Desktop\\shandong25_0.831.json'  # JSON 文件路径
    # output_excel_path = 'C:\\Users\\14398\\Desktop\\shandong25_0.831.csv'  # 输出 csv 文件路径

    # process_inference_results(tif_path, json_path, output_excel_path)

    # 文件夹所有文件批量转换
    tif_path = 'D:\\shandong'  # TIFF 图像路径
    json_path = 'C:\\Users\\14398\\Desktop\\json'  # JSON 文件路径
    output_excel_path = 'C:\\Users\\14398\\Desktop\\shandong.csv'  # 输出 csv 文件路径

    all_process_inference_results(tif_path, json_path, output_excel_path)