from ultralytics import YOLO

model = YOLO('yolo12-pose.yaml')  # Architecture config for YOLOv12 pose

model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='my_yolo12_pose_model',
    project='yolo12_pose_training'
)
