from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt') 

model.train(
    data='robot_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='robot_yolo11_pose_model_n',
    project='yolo11_pose_training'
)
