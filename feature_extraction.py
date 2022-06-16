import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from pathlib import Path
import argparse
import numpy

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image",
                required=True,
                help="Path to folder")

args = vars(ap.parse_args())
mypath = args["image"]
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = numpy.empty(len(onlyfiles), dtype=object)
classes = ["bicycle", "car", "motorcycle", "airplane", "bus",
           "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
           "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat",
           "baseball glove", "skateboard", "surfboard", "tennis racket",
           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
           "banana", "apple", "broccoli", "carrot",
           "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
           "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
           "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
           "hair drier", "toothbrush"]

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color1 = np.array([0.0, 0.0, 255.])  # red
    color2 = np.array([0.0, 255.0, 255.0])  # Other yellow
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color1, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color2, 2)


for n in range(0, len(onlyfiles)):
    path = join(mypath, onlyfiles[n])
    images[n] = cv2.imread(join(mypath, onlyfiles[n]),
                           cv2.IMREAD_UNCHANGED)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    dims = img.shape
    print("Image width: {}, height: {}, depth: {}".format(dims[1], dims[0], dims[2]))
    net = cv2.dnn.readNet('./yoloV4/yolov4.weights', './yoloV4/yolov4.cfg')
    scale = 1. / 255
    dims = img.shape
    # blobFromImage(image, scale, (Width,Height), (0,0,0), True, crop=False)
    blob = cv2.dnn.blobFromImage(img, scale, (640,640), (0, 0, 0), True, crop=True)
    # blob = np.random.standard_normal([1, 3, 608, 608]).astype(np.float32)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    Width = dims[1]
    Height = dims[0]
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.25
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.05:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    print("Classes found: " + str(class_ids))
    print("Classes names: " + str([classes[i] for i in class_ids]))
    print("Classes confidence: " + str(confidences))
    print(boxes)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    # retval = cv2.imwrite('output/' + str(n) + '_new.jpg', img)
    # rect = boxes[0]
    rect = (int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3]))
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    img0 = cv2.imread(path, cv2.IMREAD_COLOR)

    cv2.grabCut(img0, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img2 = img * mask2[:, :, np.newaxis]
    retval = cv2.imwrite('feature_extraction/' + 'Mango_' + str(n) + '.jpg', img2)
print("Images Extraction Successfully")
