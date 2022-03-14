import os
import torch



# 用在批量推理代码自动生成txt文件
def det2txt(output_dir,img_path,pred):
    classes = ['seated','standing','lying','other']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.basename(img_path)
    base_name = os.path.splitext(image_name)[0]
    txt_path = output_dir + base_name + '.txt'
    #####################复制循环进去###########yolov5lite-s
    with open(txt_path, 'a') as f:
        for box in pred:
            if box == []:
                continue
            label, x0, y0, x1, y1, score = box
            f.write("%s %s %s %s %s %s\n" % (classes[label], score, x0, y0, x1, y1))