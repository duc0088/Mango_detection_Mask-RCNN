import cv2
from Mask_RCNN_train.mrcnn.m_rcnn import *
from Mask_RCNN_train.mrcnn import config,model,visualize,utils,m_rcnn
from Mask_RCNN_train.mrcnn.visualize import random_colors, get_mask_contours, draw_mask

CLASS_NAMES = ['Mango']


class Customconfigs(m_rcnn.CustomConfig):

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    NUM_CLASSES = 1 + 1

model = model.MaskRCNN(mode="inference",
                             config=Customconfigs(1),
                             model_dir=os.getcwd())

model.load_weights(filepath="./model/mask_rcnn_object_0004.h5",
                   by_name=True)

image = cv2.imread("./data/dataset/10.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)

r = r[0]

visualize.display_instances(image=image,
                                  boxes=r['rois'],
                                  masks=r['masks'],
                                  class_ids=r['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=r['scores'])