"""
Run the model at inference to get the stats of test val and adversarial (can use say birds to check false positives)
"""

from ultralytics import YOLO

MODEL_PATH = 'runs/May14-01-21/weights/best.pt'
DATA_YAML = 'data/augmented_data/test_airplanes/data.yaml'

def evaluate_model():
    model = YOLO(MODEL_PATH)
    
    results = model.val(
        data=DATA_YAML,
        split='test', 
        imgsz=640,
        plots=True,
        save_json=True,
        verbose=True
    )
    
    print(f"Adversarial Results:")
    print(f"results: {results.results_dict}")

if __name__ == "__main__":
    evaluate_model()