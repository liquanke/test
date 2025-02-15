import math
import pandas as pd

# 计算两点之间的距离
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

# 读取文件
GEM_df = pd.read_csv('./evaluate_GEM_WRI/shandong_delete_repeat.csv')
WRI_df = pd.read_csv('./evaluate_GEM_WRI/wri_coal_shandong.csv')
cfpp_potensial_df = pd.read_csv('C:\\Users\\14398\\Desktop\\shandong_all.csv')

GEM_df_location = GEM_df[['Latitude', 'Longitude']].values
WRI_df_location = WRI_df[['latitude', 'longitude']].values
cfpp_potensial_df = cfpp_potensial_df[cfpp_potensial_df['probs1']>0.99]
cfpp_potensial_location = cfpp_potensial_df[['lat', 'lon']].values

# print(GEM_df_location)
# print(WRI_df_location)
# print(cfpp_potensial_location)

# 计算与GEM数据库的重叠率
len_gem = len(GEM_df_location)
num_gem = 0
for location1 in GEM_df_location:
    for location2 in cfpp_potensial_location:
        distances = eucliDist(location1, location2)
        if distances <= 0.003:
            num_gem = num_gem + 1
            break

recall_gem = num_gem / len_gem
print(num_gem)
print(len_gem)
print(recall_gem)


# 计算与WRI数据库的重叠率
len_wri = len(WRI_df_location )
num_wri = 0
for location1 in WRI_df_location:
    for location2 in cfpp_potensial_location:
        distances = eucliDist(location1, location2)
        if distances <= 0.003:
            num_wri = num_wri + 1
            break

recall_wri = num_wri / len_wri
print(num_wri)
print(len_wri)
print(recall_wri)