"""
In this file, I am running the base version of YOLOv11-Nano to have a performance baseline
"""

from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model('https://ultralytics.com/images/bus.jpg')

for r in results:
    r.show()
    r.save(filename='result.jpg')