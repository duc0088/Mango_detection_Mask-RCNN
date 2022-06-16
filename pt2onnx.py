import cv2
from Mask_RCNN_train.mrcnn.m_rcnn import *
from Mask_RCNN_train.mrcnn.visualize import random_colors, get_mask_contours, draw_mask

cap = cv2.VideoCapture(0)
test_model, inference_config = load_inference_model(1, "./model/mask_rcnn_object_0004.h5")
while True:
    ret,frame = cap.read()
    # img = cv2.resize(frame,(512,512))
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect results
    r = test_model.detect([frame])[0]
    colors = random_colors(10)

    # h,w = frame.shape[:2]
    object_count = len(r["class_ids"])
    print(object_count)
    for i in range(object_count):
        # 1. Mask
        mask = r["masks"][:, :, i]
        score = r["scores"]
        bbox = r['rois']
        (startx,starty,endx,endy) = bbox[0]

        print(str(score[0]))
        labels = r['class_ids'][0]
        print(labels)
        # label = classes[labels]
        contours = get_mask_contours(mask)
        if score[0] > 0.5:
            if labels == 1:
                label = 'Mango'
                # cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            else:
                label = 'Unknow'
            for cnt in contours:
                    cv2.polylines(frame, [cnt], True, colors[i], 2)
                    frame = draw_mask(frame, [cnt], colors[i])
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.rectangle(frame, (startx, starty), (startx + endx, starty + endy), (0, 255, 0))
    cv2.imshow('a', frame)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()