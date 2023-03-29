搭建环境

python3.6

pip install -r requirements.txt

 训练

1, 修改model_data下的voc_classes中的类别为你的类别

2, 下载预训练模型至model_data

3,  修改train.py中的model_path为预训练模型的地址

即可开始训练

评估 

1, 修改yolo.py中的model_path为你训练好的模型，训练好的模型存放在logs文件夹下

2, 修改yolo.py中的classes_path为model_data下的voc_classes

3, 运行yolo.py

4, 运行get_map.py

热力图CAM

1 将cam.py中的path修改为训练好的模型

2 运行cam.py即可得到热力图
