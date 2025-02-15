import os
import geopandas as gpd
import numpy as np
import shutil

def get_coordinates(shp_path):
    gdf = gpd.read_file(shp_path)
    # gdf = gdf.to_crs("EPSG:4326")
    coordinates = []

    for geom in gdf['geometry']:
        minx, miny, maxx, maxy = geom.envelope.bounds
        left, up, right, down = minx, maxy, maxx, miny
        coordinate = [left, right, up, down]
        coordinates.append(coordinate)

    return coordinates

# 批量下载矢量文件路径
shp_path = "C:\\Users\\14398\\Desktop\\shandong\\shp\\clip\\shandong_tiles_clip.shp"

# 文件保存路径
save_path = "D:\\geid_download\\shandong_add"
os.makedirs(save_path, exist_ok=True)

# 总的图片保存位置
merge_path = "D:\\geid_download\\shandong_add\\merge_tif"
os.makedirs(merge_path, exist_ok=True)

# 放大级别（从1开始）
zoom_from = 18
zoom_to = 18

# 历史图像时间
historical_date = '2024-11-30'

selcet_coordinates = get_coordinates(shp_path)
print(f"筛选后的面个数为{len(selcet_coordinates)}")

# 读取使用批量下载代码没有的图片
failed_download = np.load('./geid_cl_download/failed_download.npy')
print(failed_download)
selcet_coordinates = np.array(selcet_coordinates)
selcet_coordinates = selcet_coordinates[failed_download]
print('下载的图像数目为：', len(selcet_coordinates))


all_task=[]
for i, coordinates in enumerate(selcet_coordinates):
    # if (i>2):continue
    filename = 'shandong'
    task = {"name": filename+str(failed_download[i])+".geid", "zoom_from": zoom_from, "zoom_to": zoom_to, "left": coordinates[0], "right": coordinates[1], "top": coordinates[2], "bottom": coordinates[3], "save_path": save_path, "date": historical_date, "basename": filename+str(failed_download[i])}
    all_task.append(task)

# 下载工具路径
downloader_path = r"C:\\software\\geid\\downloader.exe"
# 合并工具路径
conbiner_path = r"C:\\software\\geid\\combiner.exe"


# 执行命令
for task in all_task:
    # cmd = f'{downloader_path} "{task["name"]}" {task["zoom_from"]} {task["zoom_to"]} ' \
    #       f'{task["left"]} {task["right"]} {task["top"]} {task["bottom"]} "{task["save_path"]}" "{task["date"]}"'
    cmd = f'{downloader_path} "{task["name"]}" {task["zoom_from"]} {task["zoom_to"]} ' \
          f'{task["left"]} {task["right"]} {task["top"]} {task["bottom"]} "{task["save_path"]}"'
    print(f"正在执行命令: {cmd}")
    os.system(cmd)  # 执行命令行操作

     # 合并影像
    combine_cmd = f'{conbiner_path} "{task["save_path"]}\\{task["name"]}" {task["zoom_to"]} tif'
    print(f"正在合并影像: {combine_cmd}")
    os.system(combine_cmd)  # 执行合并命令

    # 移动合并后的影像到统一目标文件夹
    tif_path = str(task["save_path"])+"\\"+str(task["basename"])+"_combined"
    for file in os.listdir(tif_path):
        if file.endswith((".tif")):  # 检查是否为影像文件
            source_path = os.path.join(tif_path, file)
            target_path = os.path.join(merge_path, str(task["basename"])+".tif")
            shutil.move(source_path, target_path)  # 移动文件

print("所有任务已完成！")
