import json
# import rasterio
from osgeo import gdal
import pandas as pd
import numpy as np
import os

# 读取 TIFF 图像的地理坐标系信息
def get_geotransform(tif_path):
    dataset = gdal.Open(tif_path)
    transform = dataset.GetGeoTransform()  # 获取仿射变换矩阵
    crs = dataset.GetProjection()  # 获取坐标参考系（CRS）
    return transform, crs

# # 读取 TIFF 图像的地理坐标系信息
# def get_geotransform(tif_path):
#     with rasterio.open(tif_path) as dataset:
#         transform = dataset.transform  # 获取仿射变换矩阵
#         crs = dataset.crs  # 获取坐标参考系（CRS）
#     return transform, crs

# 将像素坐标转换为经纬度
def pixel_to_geo(transform, x, y):
    lon = transform[0] + x * transform[1] + y * transform[2]
    lat = transform[3] + x * transform[4] + y * transform[5]
    return lon, lat

# # 将像素坐标转换为经纬度
# def pixel_to_geo(transform, x, y):
#     lon, lat = transform * (x, y)  # 直接使用 rasterio 的仿射变换
#     return lon, lat

# 将目标框的像素坐标转换为经纬度
def convert_bbox_to_geo(transform, bbox):
    x, y = bbox
    lon, lat = pixel_to_geo(transform, x, y)
    return lon, lat


# 计算两点之间的欧几里得距离
def calc_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def merge_near(df, path):
    
    # 定义一个阈值
    threshold = 256

    # 用于存储合并后的数据
    merged_data = []

    # 标记每一行是否已经合并
    merged_flags = [False] * len(df)

    # 遍历所有行进行合并
    for i in range(len(df)):
        if merged_flags[i]:
            continue  # 如果该点已经被合并过，跳过

        # 当前点
        x1, y1, score1, preds1, lon1, lat1 = df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['probs1'], df.iloc[i]['preds'], df.iloc[i]['lon'], df.iloc[i]['lat']
        
        # 用一个列表存储合并的点
        merge_x = [x1]
        merge_y = [y1]
        merge_score = [score1]
        merge_preds = [preds1]
        merge_lon = [lon1]
        merge_lat = [lat1]

        # 遍历其它点
        for j in range(i + 1, len(df)):
            if merged_flags[j]:
                continue  # 跳过已合并的点

            # 获取第二个点的坐标和得分
            x2, y2, score2, preds2, lon2, lat2 = df.iloc[j]['x'], df.iloc[j]['y'], df.iloc[j]['probs1'], df.iloc[j]['preds'], df.iloc[j]['lon'], df.iloc[j]['lat']

            # 计算两点之间的距离
            distance = calc_distance(x1, y1, x2, y2)
            
            if distance <= threshold:
                # 如果距离小于阈值，进行合并
                merge_x.append(x2)
                merge_y.append(y2)
                merge_score.append(score2)
                merge_preds.append(preds2)
                merge_lon.append(lon2)
                merge_lat.append(lat2)

                # 标记第二个点已合并
                merged_flags[j] = True

        # 计算合并后点的平均坐标和平均得分
        merged_x = np.mean(merge_x)
        merged_y = np.mean(merge_y)
        merged_lon = np.mean(merge_lon)
        merged_lat = np.mean(merge_lat)
        merged_score = np.max(merge_score) 
        merged_preds = np.max(merge_preds) 
        

        # 将合并后的点添加到结果中
        merged_data.append([merged_x, merged_y, merged_lon, merged_lat, merged_score, merged_preds])

    # 将合并后的数据转换为DataFrame
    merged_df = pd.DataFrame(merged_data, columns=['x', 'y', 'lon', 'lat', 'probs1', 'preds'])

    # 保存合并后的数据到新的CSV文件
    merged_df.to_csv(path)


# 文件夹内文件批量转换
def all_process_inference_results(tif_path, csv_path, output_excel_path):

    csv_filelist = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    
    # 存储输出数据
    # output_data = []
    # 便利文件夹所有文件进行转换
    for filename in csv_filelist:
        output_data = []
        csv_file = os.path.join(csv_path,filename)
        tif_file = os.path.join(tif_path,filename.replace('.csv', '.tif'))
        
        # 读取 JSON 文件中的检测结果
        detection_result = pd.read_csv(csv_file)
        cfpp_loc  = detection_result[['x','y']].values
        
        # 读取 TIFF 图像的地理信息
        transform, _ = get_geotransform(tif_file)

        # 遍历 JSON 中的检测结果，转换每个目标框为经纬度
        for result in cfpp_loc:

            # 转换目标框为经纬度
            lon, lat = convert_bbox_to_geo(transform, result)

            # 汇总数据
            output_data.append([lon, lat])
        detection_result[['lon', 'lat']] = np.array(output_data)

        # 阈值先简单过滤一下
        detection_result = detection_result[detection_result['probs1']>0.10]

        if (len(detection_result)==0):continue

        out_csv_file = os.path.join(output_excel_path, os.path.basename(csv_file))

        # 对csv中检测出来的重复目标进行合并
        merge_near(detection_result, out_csv_file)

    print(f'Results saved to {output_excel_path}')

# 合并文件夹内所有csv文件的函数
def merge_csv(path, merge_path):
    
    # 获取文件夹中所有的 CSV 文件
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

    # 初始化一个空的列表，用来存储所有读取的 DataFrame
    data_frames = []

    # 遍历所有 CSV 文件，读取并追加到 data_frames 列表
    for file in csv_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)  # 读取 CSV 文件
        data_frames.append(df)  # 将每个 DataFrame 添加到列表中

    # 将所有 DataFrame 合并成一个大的 DataFrame
    merged_df = pd.concat(data_frames, ignore_index=True)

    # 保存合并后的数据到一个新的 CSV 文件
    merged_df.to_csv(merge_path)

    print("CSV 文件已合并并保存到", merge_path)

if __name__ == '__main__':

    # 文件夹所有文件批量转换
    tif_path = 'D:\\shandong'  # TIFF 图像路径
    csv_path = 'C:\\Users\\14398\\Desktop\\csv'  # JSON 文件路径
    output_excel_path = 'C:\\Users\\14398\\Desktop\\csv\\csv1'  # 输出 csv 文件路径
    merge_path = 'C:\\Users\\14398\\Desktop\\shandong_all1.csv' # 合并所有csv文件的路径
    os.makedirs(output_excel_path, exist_ok=True)

    all_process_inference_results(tif_path, csv_path, output_excel_path)

    merge_csv(output_excel_path, merge_path)