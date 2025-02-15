# 文件名称   ：roxml_to_dota.py
# 功能描述   ：把rolabelimg标注的xml文件转换成dota能识别的xml文件，
#             再转换成dota格式的txt文件
#            把旋转框 cx,cy,w,h,angle，转换成四点坐标x1,y1,x2,y2,x3,y3,x4,y4
import os
import xml.etree.ElementTree as ET
import math


def xml2txt(xmldir, out_path):
    filelist = os.listdir(xmldir)
    for file in filelist:
        abspath = os.path.join(xmldir, file)
        if (os.path.isfile(abspath)==False):continue
        # print(abspath)
        tree = ET.parse(abspath)
        root = tree.getroot()

        name = file.strip('.xml')
        output = out_path + name + '.txt'
        file = open(output, 'w')

        objs = tree.findall('object')
        for obj in objs:
            cls = obj.find('name').text
            box = obj.find('bndbox')
            x0 = int(float(box.find('x0').text))
            y0 = int(float(box.find('y0').text))
            x1 = int(float(box.find('x1').text))
            y1 = int(float(box.find('y1').text))
            x2 = int(float(box.find('x2').text))
            y2 = int(float(box.find('y2').text))
            x3 = int(float(box.find('x3').text))
            y3 = int(float(box.find('y3').text))
            file.write("{} {} {} {} {} {} {} {} {} 0\n".format(x0, y0, x1, y1, x2, y2, x3, y3, cls))
        file.close()
        print(output)




if __name__ == '__main__':
    # -----**** 第一步：把xml文件统一转换成旋转框的xml文件 ****-----
    xmldir = "D:\\software\\labelimage\\test\\annotation"  # 目录下保存的是需要转换的xml文件
    out_path = 'D:\\software\\labelimage\\test\\annotation\\train\\txt\\'# 转换的txt文件
    xml2txt(xmldir, out_path)
