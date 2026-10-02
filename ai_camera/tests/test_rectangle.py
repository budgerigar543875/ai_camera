from ai_camera_buzzer import Rectangle

def test_top_left():
    rect = Rectangle(10, 20, 30, 40)
    assert rect.top_left() == (20, 10)

def test_bottom_right():
    rect = Rectangle(10, 20, 30, 40)
    assert rect.bottom_right() == (40, 30)

def test_offset():
    rect = Rectangle(10, 20, 30, 40)
    offset = rect.offset(3, 7)
    
    assert offset.top_left() == (23, 17)
    assert offset.bottom_right() == (43, 37)

def test_scale():
    rect = Rectangle(10, 20, 30, 40)
    scale = rect.scale(5)

    assert scale.top_left() == (100, 50)
    assert scale.bottom_right() == (200, 150)

def test_scale_xy():
    rectangle = Rectangle(10, 40, 50, 80)
    scale = rectangle.scale_xy(8, 9)

    assert scale.top_left() == (320, 90)
    assert scale.bottom_right() == (640, 450)

def test_clipping_1():
    rect = Rectangle(-100, -200, 1000, 3000)
    clipping = rect.clipping(20, 40, 60, 80)

    assert clipping.top_left() == (20, 40)
    assert clipping.bottom_right() == (60, 80)
    
def test_clipping_2():
    rect = Rectangle(100, 200, 300, 400)
    clipping = rect.clipping(190, 10, 410, 310)

    assert clipping.top_left() == (200, 100)
    assert clipping.bottom_right() == (400, 300)