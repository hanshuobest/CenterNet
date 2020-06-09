#### ctdet检测，结果保存在output_reuslt目录中
python3 demo.py ctdet --dataset coco --demo ../images/19064748793_bb942deea1_k.jpg --load_model ../models/ctdet_coco_dla_2x.pth

> 如果是自定义数据集修改--dataset custom_data，opts.py中init函数修改dataset为custom_data

> 在script目录下有脚本可以将voc的标注数据转为coco格式
---------------------------------------------------------------------------------------

#### 检测3d box
 python3 demo.py ddd --exp_id 3dop  --debug 1 --demo ../images/33887522274_eebd074106_k.jpg --load_model ../models/ddd_3dop.pth  --resume

 #### 训练ctdet
 python3 main.py ctdet --exp_id coco_dla --batch_size 12 --master_batch 1 --lr 1.25e-4 --gpus 0

 #### test
 python3 test.py --exp_id coco_dla --not_prefetch_test ctdet --load_model ../exp/ctdet/coco_dla/model_best.pth 
