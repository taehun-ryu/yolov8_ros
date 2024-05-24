import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState 

import message_filters
from cv_bridge import CvBridge

from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
from ultralytics.engine.results import Boxes

from sensor_msgs.msg import Image
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray

def compute_iou(box1, box2):
    # Calculate intersection areas
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

class TrackingNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("tracking_node")
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.cv_bridge = CvBridge()
        self.track_mappings = {}
        self.id_pool = list(range(1, 4))  # Initialize ID pool with 1, 2, 3

        self.track_mappings = {}  # Maps detected track index to an assigned track ID

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Configuring {self.get_name()}')
        tracker_name = self.get_parameter("tracker").get_parameter_value().string_value
        self.image_reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
        self.tracker = self.create_tracker(tracker_name)
        self._pub = self.create_publisher(DetectionArray, "tracking", 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Activating {self.get_name()}')
        image_qos_profile = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.image_sub = message_filters.Subscriber(self, Image, "image_raw", qos_profile=image_qos_profile)
        self.detections_sub = message_filters.Subscriber(self, DetectionArray, "detections", qos_profile=10)
        self._synchronizer = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.detections_sub], 10, 0.5)
        self._synchronizer.registerCallback(self.detections_cb)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Deactivating {self.get_name()}')
        self.destroy_subscription(self.image_sub)
        self.destroy_subscription(self.detections_sub)
        del self._synchronizer
        self._synchronizer = None
        self.track_mappings.clear()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Cleaning up {self.get_name()}')
        del self.tracker
        self.tracker_ids.clear()
        self.id_pool = list(range(1, 4))
        self.track_mappings.clear()
        return TransitionCallbackReturn.SUCCESS

    def create_tracker(self, tracker_yaml: str) -> BaseTrack:
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")
        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))
        assert cfg.tracker_type in ["bytetrack", "botsort"], f"Unsupported tracker type '{cfg.tracker_type}'"
        return TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)

    def detections_cb(self, img_msg: Image, detections_msg: DetectionArray):
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        new_detections = [[
            detection.bbox.center.position.x - detection.bbox.size.x / 2,
            detection.bbox.center.position.y - detection.bbox.size.y / 2,
            detection.bbox.center.position.x + detection.bbox.size.x / 2,
            detection.bbox.center.position.y + detection.bbox.size.y / 2,
            detection.score,
            detection.class_id,
            detection
        ] for detection in detections_msg.detections]

        current_tracks = {}
        # Update tracks with new detections based on highest IOU
        for track_id, old_box in self.track_mappings.items():
            best_iou = 0
            best_detection = None
            for detection in new_detections:
                iou = compute_iou(old_box, detection[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_detection = detection

            if best_detection and best_iou > 0.3:  # Adjust IOU threshold as needed
                current_tracks[track_id] = best_detection[:4]
                best_detection[6].id = str(track_id)
                new_detections.remove(best_detection)

        # Assign new IDs to unmatched detections
        for detection in new_detections:
            if self.id_pool:
                new_id = self.id_pool.pop(0)
                current_tracks[new_id] = detection[:4]
                detection[6].id = str(new_id)

        self.track_mappings = current_tracks
        self._pub.publish(detections_msg)


def main():
    rclpy.init()
    node = TrackingNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
