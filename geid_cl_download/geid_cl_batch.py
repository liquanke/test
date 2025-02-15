import os
import geopandas as gpd

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
shp_path = "C:\\Users\\14398\\Desktop\\Large_factory_detectiom\\code\\cfpp_gnn_model\\result_area\\result_area.shp"

# 文件保存路径
save_path = "D:\\geid_download\\shandong_add"
os.makedirs(save_path, exist_ok=True)

# 放大级别（从1开始）
zoom_from = 18
zoom_to = 18

# 历史图像时间
historical_date = '2024-11-30'

selcet_coordinates = get_coordinates(shp_path)
print(f"筛选后的面个数为{len(selcet_coordinates)}")

all_task=[]
for i, coordinates in enumerate(selcet_coordinates):
    if (i>4):continue
    filename = 'reslut'
    task = {"name": filename+str(i)+".geid", "zoom_from": zoom_from, "zoom_to": zoom_to, "left": coordinates[0], "right": coordinates[1], "top": coordinates[2], "bottom": coordinates[3], "save_path": save_path, "date": historical_date}
    all_task.append(task)

# 下载工具路径
downloader_path = r"C:\\software\\geid\\downloader.exe"

# 执行命令
for task in all_task:
    cmd = f'{downloader_path} "{task["name"]}" {task["zoom_from"]} {task["zoom_to"]} ' \
          f'{task["left"]} {task["right"]} {task["top"]} {task["bottom"]} "{task["save_path"]}" "{task["date"]}"'
    # cmd = f'{downloader_path} "{task["name"]}" {task["zoom_from"]} {task["zoom_to"]} ' \
    #       f'{task["left"]} {task["right"]} {task["top"]} {task["bottom"]} "{task["save_path"]}"'
    print(f"正在执行命令: {cmd}")
    os.system(cmd)  # 执行命令行操作

print("所有任务已完成！")
