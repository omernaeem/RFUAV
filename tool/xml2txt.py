"""
將labelme打的標簽文件轉換為.txt看bbox的位置
"""
import xml.etree.ElementTree as ET
from os import getcwd
classes = ["signal"]


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_name):
    out_file = open('C:/Users/user/Desktop/' + 'temp' + '.txt', 'w')  # 转换后的txt文件存放路径
    f = open('C:/Users/user/Desktop/DAUTEL.xml')
    xml_text = f.read()
    root = ET.fromstring(xml_text)
    f.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)


    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

if __name__ == '__main__':

    convert_annotation(1)
