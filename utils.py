import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


#eg detections: [0.7512247 , 0.32209778, 0.8467484 , 0.7646923 ]

#  [[0.79898655]
#  [0.54339504]
#  [0.04227827]
#  [0.21582666]]

def bbox_to_kalman(bbox):
    """
    input as x_min,y_min,x_max,y_max
    output as x_centre,y_centre,size,ascpect ratio
    """
    width, height = bbox[2:4] - bbox[0:2]
    center_x, center_y = (bbox[0:2] + bbox[2:4]) / 2
    area = width * height  # scale is just area
    r = width / height
    out = np.array([center_x, center_y, area, r]).astype(np.float64)
    return np.expand_dims(out,axis=1)

def kalman_to_bbox(bbox):
    """
    input as x_centre,y_centre,size,ascpect ratio
    output as x_min,y_min,x_max,y_max
    """
    bbox = bbox[:, 0]
    width = np.sqrt(bbox[2]*bbox[3])
    height = bbox[2]/width
    x_min,y_min,x_max,y_max = bbox[0]-width/2,bbox[1]-height/2,bbox[0]+width/2,bbox[1]+height/2

    return np.array([x_min,y_min,x_max,y_max]).astype(np.float32)

def iou(a: np.ndarray,b: np.ndarray) -> float:
    """
    calculate iou between two boxes
    """
    a_tl, a_br = a[:4].reshape((2, 2))
    b_tl, b_br = b[:4].reshape((2, 2))
    int_tl = np.maximum(a_tl, b_tl)
    int_br = np.minimum(a_br, b_br)
    int_area = np.product(np.maximum(0., int_br - int_tl))
    a_area = np.product(a_br - a_tl)
    b_area = np.product(b_br - b_tl)
    return int_area / (a_area + b_area - int_area)


def compare_boxes(detections,trackers,iou_thresh=0.3):

    iou_matrix = np.zeros(shape=(len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)
    
    # calculate maximum iou for each pair through hungarian algorithm
    row_id, col_id = linear_sum_assignment(-iou_matrix)
    matched_indices = np.transpose(np.array([row_id,col_id]))
    # geting matched ious
    iou_values = np.array([iou_matrix[row_id,col_id] for row_id,col_id in matched_indices])
    best_indices = matched_indices[iou_values > iou_thresh]

    unmatched_detection_indices = np.array([d for d in range(len(detections)) if d not in best_indices[:,0]])  
    unmatched_trackers_indices = np.array([t for t in range(len(trackers)) if t not in best_indices[:,1]])

    return best_indices,unmatched_detection_indices,unmatched_trackers_indices


