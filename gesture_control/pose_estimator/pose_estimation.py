from multiprocessing import Process, Event, Array, Value, Lock, log_to_stderr
import logging
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from dataclasses import dataclass
from typing import Any, Sequence, Tuple
import mediapipe
from enum import Flag, IntEnum
import mediapipe as mp
import cv2
import ctypes

@dataclass
class PoseEstimator2DConfig:
    distortion_maps:Tuple[Any, Any] # distortion maps as returned by cv2.initUndistortRectifyMap
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float  = 0.5
    
@dataclass
class SharedNumpyArray:
    '''
    Wrapper for Numpy array in shared memory.
    '''
    shape: Tuple[int, ...]   # shared array shape
    dtype: DTypeLike         # shared array dtype
    arr: Array               # shared array buffer
    changed_flag: Value      # change flag

    def __init__(self, shape:Tuple[int,...], dtype:DTypeLike, arr:Array=None, changed_flag:Value=None):
        self.shape = shape
        self.dtype = dtype
        self.arr = arr if arr is not None else Array(np.ctypeslib.as_ctypes_type(dtype), int(np.product(shape)))
        self.changed_flag = changed_flag if changed_flag is not None else Value(ctypes.c_bool)

    def as_numpy(self) -> ArrayLike:
        return np.frombuffer(self.arr.get_obj(), dtype=self.dtype).reshape(self.shape)

    def has_changed(self) -> bool:
        return self.changed_flag.value

    def set_changed(self, value) -> None:
        self.changed_flag.value = value

class ColorSpace(IntEnum):
    RGB=0
    GRAY=cv2.COLOR_GRAY2RGB
    BGR=cv2.COLOR_BGR2RGB
    HSV=cv2.COLOR_HSV2RGB

@dataclass
class SharedImage(SharedNumpyArray):
    color: ColorSpace

    def __init__(self, width:int, height:int, color:ColorSpace, *args, **kwargs):
        shape = (height, width)

        if color != ColorSpace.GRAY:
            shape = (*shape, 3)
            
        super(SharedImage, self).__init__(shape=shape,dtype=np.uint8, *args, **kwargs)
        self.color = color
    
    @property
    def width(self):
        return self.shape[1]
    
    @property
    def height(self):
        return self.shape[0]

class PoseAnnotationType(Flag):
    NONE = 0
    BODY = 1
    LEFT_HAND = 2
    RIGHT_HAND = 4
    FACE = 8
    HANDS = LEFT_HAND | RIGHT_HAND
    ALL = BODY | HANDS | FACE

def draw_landmarks(image:ArrayLike, results:Any, mode:PoseAnnotationType) -> None:
    if mode & PoseAnnotationType.FACE:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.face_landmarks,
            mp.solutions.holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
    
    if mode & PoseAnnotationType.BODY:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles
            .get_default_pose_landmarks_style())

    if mode & PoseAnnotationType.LEFT_HAND:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style())
    
    if mode & PoseAnnotationType.RIGHT_HAND:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style())

class PoseEstimator2DProcess(Process):
    '''
    Background process to perform image rectification and 2D pose estimation.

    Uses the MediaPipe Holistic solution.

    Input GRAY, output BGR.

    Shutdown by setting stop_event.
    '''

    NUM_BODY_LANDMARKS = 33
    NUM_HAND_LANDMARKS = 21
    NUM_FACE_LANDMARKS = 468

    def __init__(self,
                 config:PoseEstimator2DConfig,
                 in_lock:Lock,
                 in_image:SharedImage,
                 out_lock:Lock,                            # shared lock for all output
                 out_image:SharedImage,                # undistorted image
                 out_body_landmarks:SharedNumpyArray,       # (33, 2)
                 out_left_hand_landmarks:SharedNumpyArray,  # (21, 2)
                 out_right_hand_landmarks:SharedNumpyArray, # (21, 2)
                 out_face_landmarks: SharedNumpyArray,      # (468, 2)
                 stop_event:Event,
                 out_annotate:PoseAnnotationType = PoseAnnotationType.NONE,
                 *args, **kwargs):

        super(PoseEstimator2DProcess, self).__init__(*args, **kwargs)

        self.config = config
        
        # init inputs
        self.in_lock = in_lock
        self.in_changed = in_image.changed_flag
        self.in_image = in_image.as_numpy()

        # init outputs
        self.out_lock = out_lock
        self.out_changed = out_image.changed_flag  # this change flag is used for all output
        self.out_image = out_image.as_numpy()
        
        self.out_body_landmarks = out_body_landmarks.as_numpy()
        self.out_left_hand_landmarks = out_left_hand_landmarks.as_numpy()
        self.out_right_hand_landmarks = out_right_hand_landmarks.as_numpy()
        self.out_face_landmarks = out_face_landmarks.as_numpy()

        # stop event
        self.stop_event = stop_event

        # annotation
        self.out_annotate = out_annotate

        # set color conversion
        self._color_cvt = in_image.color if in_image.color != ColorSpace.RGB else None

        # logging
        self.logger = log_to_stderr()
        self.logger.setLevel(logging.INFO)

    def run(self):
        with mediapipe.solutions.holistic.Holistic(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        ) as model:

            while not self.stop_event.is_set():
                # self.logger.debug('Waiting for image...')

                locked = self.in_lock.acquire(timeout=0.1)

                self.logger.debug('Locked: {}'.format(locked))
                
                if locked:
                    if self.in_changed.value:
                        self.logger.debug('Attempting to process...')

                        # make a numpy copy and release the lock
                        image = self.in_image.copy()
                        self.in_changed.value = False # set valid to False to indicate consumed
                        self.in_lock.release()

                        # undistort
                        image = cv2.remap(src=image,
                                          map1=self.config.distortion_maps[0],
                                          map2=self.config.distortion_maps[1],
                                          interpolation=cv2.INTER_LINEAR)

                        # convert color if necessary
                        image.flags.writeable = False
                        if self._color_cvt is not None:
                            image = cv2.cvtColor(image, self._color_cvt)

                        # process the image
                        results = model.process(image)

                        # make writeable
                        image.flags.writeable = True

                        # annotate
                        draw_landmarks(image, results, self.out_annotate)

                        # convert to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        # convert landmarks

                        body_landmarks = self.landmarks_to_numpy(self.NUM_BODY_LANDMARKS, results.pose_landmarks)
                        left_hand_landmarks = self.landmarks_to_numpy(self.NUM_HAND_LANDMARKS, results.left_hand_landmarks)
                        right_hand_landmarks = self.landmarks_to_numpy(self.NUM_HAND_LANDMARKS, results.right_hand_landmarks)
                        face_landmarks = self.landmarks_to_numpy(self.NUM_FACE_LANDMARKS, results.face_landmarks)

                        # write to outputs
                        with self.out_lock:
                            np.copyto(self.out_image, image)
                            np.copyto(self.out_body_landmarks, body_landmarks)
                            np.copyto(self.out_left_hand_landmarks, left_hand_landmarks)
                            np.copyto(self.out_right_hand_landmarks, right_hand_landmarks)
                            np.copyto(self.out_face_landmarks, face_landmarks)
                            self.out_changed.value = True # set True to indicate changed
                    else:
                        self.in_lock.release()

    def landmarks_to_numpy(self, n, landmarks:Any) -> ArrayLike:
        if landmarks is None:
            return np.inf*np.ones((n, 2))
        else:
            return np.array([(l.x, l.y) for l in landmarks.landmark])

class BodyLandmarks(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP  = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class HandLandmarks(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20