python sample-apps/radiology/main.py -s /home/yczhao/Desktop/MONAILabel_datasets/myTask01_Breast -t train
python sample-apps/radiology/main.py -s D:\Desktop\MONAILabel_datasets\myTask01_Breast -t train
monailabel start_server --app sample-apps/radiology -s ../MONAILabel_datasets/myTask01_Breast --conf models segmentation_breast
monailabel start_server --app sample-apps/econet -s ../MONAILabel_datasets/myTask01_Breast
ssh -L 8000:127.0.0.1:8000  -N -f yczhao@server.aibdt.cn -p 10088