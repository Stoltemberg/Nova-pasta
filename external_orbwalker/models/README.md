# YOLO Models Directory

Place your trained YOLOv8 model here as `orbwalker_yolo.pt`.

## How to train:

1. Capture 2000-5000 screenshots of League of Legends gameplay
2. Label them with LabelImg or CVAT using these classes:
   - `enemy_champion` (class 0)
   - `enemy_minion` (class 1) 
   - `enemy_turret` (class 2)
   - `ally_minion` (class 3)
   - `jungle_monster` (class 4)
3. Train with ultralytics:
   ```
   yolo detect train data=lol_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```
4. Copy `runs/detect/train/weights/best.pt` to this directory as `orbwalker_yolo.pt`
