from os import name
from ultralytics import YOLO, settings

settings.update({'runs_dir': './oyster_runs/'})

model = YOLO('yolov8l-cls.pt')

results = model.train(data="./Oyster Shell/", epochs=150, name="oyster")
