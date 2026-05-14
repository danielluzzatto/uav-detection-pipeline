"""
Run the model at inference to get the stats of test val and adversarial (can use say birds to check false positives)
"""

from ultralytics import YOLO

MODEL_PATH = 'runs/aug_run_12May18-01/weights/best.pt'
DATA_YAML = 'data/augmented_data/data.yaml'

def evaluate_model(split='val'):
    """
    Evaluates the model on a specific split: 'train', 'val', or 'test', 'test_adversarial'.
    """
    model = YOLO(MODEL_PATH)
    
    results = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=640,
        plots=True,
        save_json=True,
        verbose=True
    )
    
    print(f"Results for {split} split:")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")


if __name__ == "__main__":
    # val test or test_birds
    evaluate_model(split='test_birds')
