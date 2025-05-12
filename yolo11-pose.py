from ultralytics import YOLO

model = YOLO('yolo11m-pose.pt') 

model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='my_yolo11_pose_model',
    project='yolo11_pose_training'
)
