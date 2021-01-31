import cv2
import numpy as np
from . utils import bbox_to_kalman, kalman_to_bbox

class  KalmanTracker():
    def __init__(self,bbox,label):

        self.no_of_updates = 0

        self.time_since_last_update = 0
        self.kalman = cv2.KalmanFilter(dynamParams=7,measureParams=4,type=cv2.CV_64F)
        self.id = label
        # ux,uy,s,r,ux',uy',s'
        self.kalman.transitionMatrix = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]],dtype=np.float64)

        self.kalman.measurementMatrix = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]],dtype=np.float64)
        # Q -> process noise
        # we assume noise in ux,uy,s,r is small
        self.kalman.processNoiseCov = np.diag([10,10,10,10,10000,10000,10000]).astype(np.float64)
        # R -> Measurement noise
        # we assume little noise in ux and uy
        self.kalman.measurementNoiseCov = np.diag([1,1,10,10]).astype(np.float64)

        # initial state matrix and process covariance matrix
        # initial state matrix gets set when a new object comes to a frame with detections
        # for eg: [[0.7512247], [0.32209778], [0.8467484], [0.7646923]]
        # we concat zeros because its where output comes

        # Xk-1 -> state matrix 7x1
        self.kalman.statePost = np.vstack((bbox_to_kalman(bbox), [[0], [0], [0]]))
        # Pk-1 -> process covariance matrix
        self.kalman.errorCovPost = np.diag([1, 1, 1, 1, 0.001, 0.001, 0.00001]).astype(np.float64)

    def predict(self):

        if self.time_since_last_update > 0:
            self.no_of_updates = 0
        self.time_since_last_update += 1
        
        out = self.kalman.predict()
        return kalman_to_bbox(out)

    def update(self,bbox):
        # if we have called predict twice in a row , it will set update flag as 0
        self.no_of_updates += 1
        self.time_since_last_update = 0
        self.kalman.correct(bbox_to_kalman(bbox))

    @property
    def current_state(self):
        """
        returns current bounding box
        """
        return kalman_to_bbox(self.kalman.statePost)





"""
Note: Time since last update is that each time we prdict using kalman
it increment by 1. it actually indicates the no of frames/iterations the tracker is not updated.
When the tracker is not updated for long we will remove those tracker which we will implement in
sort.py
So For unmatched trackers we donot update the detections, In that cases we 
only predict those trackers. If this continous for n frames, time_since_last update will 
be equal to n.

noof_updates: no_of_updates increment by 1 whenever we update. This can be used to control
when we need to take output from kalman filter. we have hit_sum. if hit_sum = 3, we will 
take output only when no_of_updates becomes 3.

"""