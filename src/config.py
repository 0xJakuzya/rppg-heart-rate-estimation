# algorithm
RPPG_METHOD: str = "POS"
FILTER_TYPE: str = "chebyshev2"

# camera 
CAMERA_INDEX: int = 0         
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FPS_TARGET: int = 15

# window
WINDOW_NAME: str = "rPPG"
PLOT_H: int = 80               

# overlay text 
FONT_SCALE: float = 0.55
FONT_COLOR: tuple[int, int, int] = (180, 180, 180)
FONT_THICKNESS: int = 1

# signal buffer
BUFFER_SEC: int = 10

# detrend filter 
DETREND_LAMBDA: float = 100.0 

# chebyshev type II bandpass
CHEBY_LO: float = 0.7
CHEBY_HI: float = 2.5
CHEBY_ORDER: int = 2
CHEBY_RS: float = 40.0

# hr estimation
HR_LO_HZ: float = 0.75
HR_HI_HZ: float = 2.5

# mediapipe face model 
FACE_MODEL_PATH: str = "face_landmarker.task"
FACE_MAX_NUM: int = 1
FACE_MIN_DETECTION_CONFIDENCE: float = 0.5
FACE_MIN_TRACKING_CONFIDENCE: float = 0.5

#  roi landmark  
LEFT_CHEEK_IDX: list[int] = [50, 101, 118, 117, 116, 123, 147, 213, 192, 214]
RIGHT_CHEEK_IDX: list[int] = [280, 330, 347, 346, 345, 352, 376, 433, 416, 434]
FOREHEAD_IDX: list[int] = [109, 67, 103, 54, 21, 162, 127, 10, 356, 389, 251, 284, 332, 297, 338]
MULTI_ROI_COUNT: int = 8
ROI_PATCH_SIZE: int = 24
ROI_PREVIEW_SCALE: int = 4
ROI_PREVIEW_MARGIN: int = 6

# model
CNN_BASE_CHANNELS: int = 32  
CNN_WINDOW: int = 300        
CNN_EPOCHS: int = 50
CNN_BATCH_SIZE: int = 16
CNN_LR: float = 1e-3
CNN_PPG_LO: float = 0.5       
CNN_PPG_HI: float = 3.5
CNN_MODEL_PATH: str = "cnn.pth"
PATCH_MODEL_SPATIAL_CHANNELS: int = 32
PATCH_MODEL_TEMPORAL_CHANNELS: int = 64

# shift loss
SHIFT_LOSS_MAX_SHIFT_SEC: float = 0.33
SHIFT_LOSS_FPS: float = 15.0
SHIFT_LOSS_EPS: float = 1e-8
