from os import name
from ultralytics import YOLO, settings

settings.update({'runs_dir': './fish_runs/'})

model = YOLO('yolov8s-cls.pt')

results = model.train(data="./8 Fish Species/", epochs=10, name="fish")
