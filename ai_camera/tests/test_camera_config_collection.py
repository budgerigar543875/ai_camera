from ai_camera_buzzer import CameraConfigCollection
from pathlib import Path

def test_load_config():
    DIR = Path(__file__).resolve().parent

    configs = CameraConfigCollection(f"{DIR}/test_camera_config.json")

    assert configs.model_path() == "path/to/model.tflite"
    assert configs.label_path() == "path/to/labels.txt"

    person = configs.get("person")
    assert person.enabled() is True
    assert person.threshold() == 0.7
    
    cat = configs.get("cat")
    assert cat.enabled() is True
    assert cat.threshold() == 0.6
    
    dog = configs.get("dog")
    assert dog.enabled() is False
    assert dog.threshold() == 0.5

    assert configs.get("unknown") is None