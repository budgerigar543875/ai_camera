from ai_camera_buzzer import DrawConfigCollection
from pathlib import Path

def test_load_config():
    DIR = Path(__file__).resolve().parent

    configs = DrawConfigCollection(f"{DIR}/test_draw_config.json")

    person = configs.get("person")
    assert person.bg_color() == (100,101,102)
    assert person.text_color() == (110,111,112)
    assert person.thickness() == 1.1
    
    cat = configs.get("cat")
    assert cat.bg_color() == (200,201,202)
    assert cat.text_color() == (210,211,212)
    assert cat.thickness() == 2.2
    
    dog = configs.get("dog")
    assert dog.bg_color() == (300,301,302)
    assert dog.text_color() == (310,311,312)
    assert dog.thickness() == 3.3

    assert configs.get("unknown") is None