import json
import matplotlib.pyplot as plt

with open('detections.json', 'r') as f:
    data = json.load(f)


detection_types = {"detector": 0, "kalman": 0, "csrt": 0}

for i in range(len(data)):
    elm = data[i]
    class_det = elm['source']
    detection_types[class_det] += 1

keys = list(detection_types.keys())
values = list(detection_types.values())

bars = plt.bar(keys, values, color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')

plt.title('Total Detections by Source Type', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Detection Source', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.bar(detection_types.keys(), detection_types.values())
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.0, 
        yval + (max(values) * 0.01),
        f'{yval}',
        ha='center',
        va='bottom',
        fontweight='bold'
    )

plt.ylim(0, max(values) * 1.15)

plt.show()