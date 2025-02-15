# 根据已有的训练集和测试集label文件进行VOC2007数据集的制作，包括生成trainval文件和test文件

import os

dataroot = "C:\\Users\\14398\\Desktop\\voc2007\\ori_data"
saveroot = 'C:\\Users\\14398\\Desktop\\voc2007\\ImageSets\\Main'

trainval = dataroot+'/trainval'
test = dataroot+'/test'

# 生成trainval.txt文件
xml_list  = os.listdir(trainval)
trainval_out_txt = os.path.join(saveroot,'trainval.txt')
print(trainval_out_txt)
with open(trainval_out_txt, "w+" ,encoding='UTF-8') as out_file:
    for xml_file in xml_list:
        out_file.write(xml_file[0:len(xml_file)-4]+'\n')
        print(xml_file[0:len(xml_file)-4])

# 生成test.txt文件
xml_list  = os.listdir(test)
test_out_txt = os.path.join(saveroot,'test.txt')
print(test_out_txt)
with open(test_out_txt, "w+" ,encoding='UTF-8') as out_file:
    for xml_file in xml_list:
        out_file.write(xml_file[0:len(xml_file)-4]+'\n')
        print(xml_file[0:len(xml_file)-4])
