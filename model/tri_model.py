import argparse
import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

deficit_weight = './model/bdc_tsc_ood_frcnn_deficit21_20201125.pth'
bolt_weight = './model/bdc_tsc_ood_frcnn_bolt7_20200910.pth'
joint_weight = './model/bdc_tsc_ood_frcnn_tianshui-joint_20201104.pth'

deficit21_class_path = './model/classes_deficit21.txt'
bolt7_class_path = './model/classes_bolt7.txt'

def tri_model(in_path, out_path,csv=False,filter=False):
    ### deficit21 model
    f = open(deficit21_class_path)
    classes = f.read().split()
    class_dict = dict.fromkeys(classes, 0)

    result_table = pd.DataFrame(columns=['file_name', 'bbox', 'label', 'score'])

    if not os.path.exists(out_path + '/deficit21'):
        os.mkdir(out_path + '/deficit21')

    deficit_out_path = out_path + '/deficit21'

    model=fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False,num_classes=len(classes)+1)
    model.eval()
    model.load_state_dict(torch.load(deficit_weight, map_location=torch.device('cpu')))

    imglist = os.listdir(in_path)

    for file in imglist:

        frame= cv2.imdecode(np.fromfile(in_path + '/' + file,dtype = np.uint8),-1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=torch.Tensor(frame)
        img/=255.
        img=img.permute(2,0,1)

        with torch.no_grad():
            res=model([img])

        bboxes = res[0]['boxes'].cpu().numpy()
        labels = res[0]['labels'].cpu().numpy()
        scores = res[0]['scores'].cpu().numpy()
        th = scores > 0.25
        bboxes = bboxes[th]
        labels = labels[th]
        scores = scores[th]
        
        if filter:
            if not labels.any():
                continue

        for bbox, label, score in zip(bboxes, labels, scores):

            class_dict[classes[label-1]] += 1

            bbox_int = bbox.astype(np.int32)

            result_table = result_table.append({'file_name': file, 'bbox': bbox_int, 'label': str(label), 'score': score}, ignore_index=True)

            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(frame, left_top, right_bottom, (0,240,255), thickness=5)
            cv2.putText(frame, classes[label - 1], (left_top[0], left_top[1]), cv2.FONT_HERSHEY_COMPLEX, 1., (0,0,255), 2)

        im = Image.fromarray(frame)
        im.save(deficit_out_path + '/' + file)

    if csv:
        result_table.to_csv(deficit_out_path + '/' + 'deficit_results.csv')
    
    ### joint model
    classes = ['joint']
    class_dict = dict.fromkeys(classes, 0)

    result_table = pd.DataFrame(columns=['file_name', 'bbox', 'label', 'score'])

    if not os.path.exists(out_path + '/joint'):
        os.mkdir(out_path + '/joint')
    joint_out_path = out_path + '/joint'

    model=fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False,num_classes=len(classes)+1)
    model.eval()
    model.load_state_dict(torch.load(joint_weight, map_location=torch.device('cpu')))

    imglist = os.listdir(in_path)

    for file in imglist:

        frame= cv2.imdecode(np.fromfile(in_path + '/' + file,dtype = np.uint8),-1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=torch.Tensor(frame)
        img/=255.
        img=img.permute(2,0,1)

        with torch.no_grad():
            res=model([img])

        bboxes = res[0]['boxes'].cpu().numpy()
        labels = res[0]['labels'].cpu().numpy()
        scores = res[0]['scores'].cpu().numpy()
        th = scores > 0.25
        bboxes = bboxes[th]
        labels = labels[th]
        scores = scores[th]

        count = 1

        for bbox, label, score in zip(bboxes, labels, scores):

            class_dict[classes[label-1]] += 1

            bbox_int = bbox.astype(np.int32)

            result_table = result_table.append({'file_name': file, 'bbox': bbox_int, 'label': str(label), 'score': score}, ignore_index=True)

    #         left_top = (bbox_int[0], bbox_int[1])
    #         right_bottom = (bbox_int[2], bbox_int[3])
    #         cv2.rectangle(frame, left_top, right_bottom, (0,240,255), thickness=5)
    #         cv2.putText(frame, str(label), (left_top[0], left_top[1]), cv2.FONT_HERSHEY_COMPLEX, 1., (0,0,255), 2)
            b = (bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3])

            image = Image.open(in_path + '/' + file, 'r')
            clip = image.crop(b)
            clip = clip.convert('RGB')
            clip.save(joint_out_path + '/' + file[:-4] + '_joint' + str(count) + '.jpg')
            count += 1

    #     im = Image.fromarray(frame)
    #     im.save(joint_out_path + '/' + file[:-4] + '_joint.jpg')

    # result_table.to_csv(joint_out_path + '/' + 'joint_results.csv')


    ### bolts model
    f = open(bolt7_class_path)
    classes = f.read().split()
    class_dict = dict.fromkeys(classes, 0)

    result_table = pd.DataFrame(columns=['file_name', 'bbox', 'label', 'score'])

    if not os.path.exists(out_path + '/bolt'):
        os.mkdir(out_path + '/bolt')
    bolt_out_path = out_path + '/bolt'

    model=fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False,num_classes=len(classes)+1)
    model.eval()
    model.load_state_dict(torch.load(bolt_weight, map_location=torch.device('cpu')))

    imglist = os.listdir(out_path + '/joint')

    for file in imglist:

        frame= cv2.imdecode(np.fromfile(out_path + '/joint/' + file,dtype = np.uint8),-1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=torch.Tensor(frame)
        img/=255.
        img=img.permute(2,0,1)

        with torch.no_grad():
            res=model([img])

        bboxes = res[0]['boxes'].cpu().numpy()
        labels = res[0]['labels'].cpu().numpy()
        scores = res[0]['scores'].cpu().numpy()
        th = scores > 0.25
        bboxes = bboxes[th]
        labels = labels[th]
        scores = scores[th]
        
        if filter:
            if not labels.any():
                continue

        for bbox, label, score in zip(bboxes, labels, scores):

            if label == 1:
                continue

            class_dict[classes[label-1]] += 1

            bbox_int = bbox.astype(np.int32)

            result_table = result_table.append({'file_name': file, 'bbox': bbox_int, 'label': str(label), 'score': score}, ignore_index=True)

            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(frame, left_top, right_bottom, (0,240,255), thickness=5)
            cv2.putText(frame, classes[label - 1], (left_top[0], left_top[1]), cv2.FONT_HERSHEY_COMPLEX, 1., (0,0,255), 2)

        im = Image.fromarray(frame)
        im.save(bolt_out_path + '/' + file)
    
    if csv:
        result_table.to_csv(bolt_out_path + '/' + 'bolt_results.csv')