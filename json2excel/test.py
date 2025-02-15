import os
import numpy as np
import pandas as pd


# 计算两点之间的欧几里得距离
def calc_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def merge_near(df, path):
    
    # 定义一个阈值
    threshold = 128

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


csv_path = 'C:\\Users\\14398\\Desktop\\csv\\shandong25.csv'  # CSV 文件路径

csv_pd = pd.read_csv(csv_path)

print(csv_pd)

csv_pd = csv_pd[csv_pd['probs1']>0.10]

print(len(csv_pd))

# merge_near(csv_pd, 'C:\\Users\\14398\\Desktop\\shandong2525.csv')
