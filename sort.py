import cv2
import numpy as np
from . kalman import KalmanTracker
from . utils import compare_boxes

class Sort:
    def __init__(self,max_age=10,hit_sum=1):
        #max_age: no of cocecutive frames in which tracker can exist without an associated box
        self.max_age = max_age
        # no of concecutive frames an tracker is associated with a detection so that we consider
        # it as an object
        self.hit_sum = hit_sum
        self.trackers = []
        self.count = 0
    
    def next_id(self):
        self.count += 1
        return self.count

    def update(self, dets):

        #For all active trackers, it will predict new location using .predict() and obtain those
        #using .current_state()
        self.trackers = [tracker for tracker in self.trackers if not 
                                    np.any(np.isnan(tracker.predict()))]

        trks = np.array([tracker.current_state for tracker in self.trackers])
        
        # Then we associate detections with tracker boxes
        matched, unmatched_dets, unmatched_tracks = compare_boxes(dets,trks)

        # Then we will update the kalman filter with measurements
        # for each detection we maintain seperate filter
        for detection_num, tracker_num in matched:
            self.trackers[tracker_num].update(dets[detection_num])

        # For all unmatched detections we will create new tracking in Kalman.
        # it means new object comes to the frame
        for i in unmatched_dets:
            self.trackers.append(KalmanTracker(dets[i, :], self.next_id()))


        # we are taking only those trackers which is updated and predicted atleast self.hit_sum        
        out = np.array([np.concatenate((trk.current_state, [trk.id]))
                for trk in self.trackers
                if trk.time_since_last_update == 0 and trk.no_of_updates >= self.hit_sum])
        
        # we are deleting those trackers which remians in frame for long and not updating
        self.trackers = [tracker for tracker in self.trackers if tracker.time_since_last_update <= self.max_age]
        
        return out
        