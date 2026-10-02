import RPi.GPIO as GPIO
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import time
import json
import datetime

class Logger:
    def log(message:str):
        print(f"{datetime.datetime.now()} {message}")

class Gpio:
    BOARD = GPIO.BOARD
    BCM = GPIO.BCM

    def __init__(self, mode):
        self.used_channels = set()
        self.old_channel_state = {}
        GPIO.setmode(mode)

    def setup(self, channel, direction):
        self.used_channels.add(channel)
        self.old_channel_state[channel] = -1
        GPIO.setup(channel, direction)
    
    def output(self, channel, value):
        if channel not in self.used_channels:
            self.setup(channel, GPIO.OUT)
        GPIO.output(channel, value)

    def on(self, channel):
        self.output(channel, GPIO.HIGH)
        if self.old_channel_state[channel] != 1:
            self.old_channel_state[channel] = 1
            Logger.log(f"[ON] Channel {channel}")

    def off(self, channel):
        self.output(channel, GPIO.LOW)
        if self.old_channel_state[channel] != 0:
            self.old_channel_state[channel] = 0
            Logger.log(f"[OFF] Channel {channel}")

    def cleanup(self):
        Logger.log("Cleanup GPIO")
        for channel in self.used_channels:
            self.off(channel)
        GPIO.cleanup()

class Camera:
    def __init__(self, width:int, height:int):
        self.camera = Picamera2()
        self.camera.configure(
            self.camera.create_preview_configuration(
                main={"size": (width, height)}
            )
        )

    def start(self, wait_sec:float=2):
        Logger.log("Start Camera")
        self.camera.start()
        time.sleep(wait_sec)
    
    def capture(self):
        frame = self.camera.capture_array()
        return Image(cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB))

    def stop(self):
        Logger.log("Stop Camera")
        self.camera.stop()

class Image:
    def __init__(self, image):
        self._image = image

    def resize(self, to_width:int, to_height:int, dtype):
        h, w = self._image.shape[:2]
        scale = min(to_width / w, to_height / h)
        rh = int(h * scale)
        rw = int(w * scale)

        cx = (to_width - rw) // 2
        cy = (to_height - rh) // 2

        resized = np.zeros(
            (to_height, to_width, 3),
            dtype=dtype
        )

        resized[cy:cy+rh, cx:cx+rw] = cv2.resize(self._image, (rw, rh))
        return ResizeResult(
            Image(resized),
            scale,
            (cx, cy)
        )
    
    def raw_image(self):
        return self._image

class ResizeResult:
    def __init__(self, resized_image:Image, resize_scale:float, offset):
        self._image = resized_image
        self._scale = resize_scale
        self._offset = offset

    def image(self) -> Image:
        return self._image
    
    def scale(self) -> float:
        return self._scale
    
    def offset(self):
        return self._offset

class CameraConfig:
    def __init__(self, config):
        self._config = config

    def enabled(self) -> bool:
        return self._config["enabled"]

    def threshold(self) -> float:
        return self._config["threshold"]

class CameraConfigCollection:
    def __init__(self, config_path:str):
        self._classes = {}
        with open(config_path, 'r', encoding="utf-8")as f:
            items = json.load(f)

        self._model_path = items["model_path"]
        self._label_path = items["label_path"]

        for key in items["classes"]:
            self._classes[key] = CameraConfig(items["classes"][key])

    def model_path(self) -> str:
        return self._model_path

    def label_path(self) -> str:
        return self._label_path
    
    def get(self, class_name:str) -> CameraConfig | None:
        return self._classes[class_name] if class_name in self._classes else None

class DrawConfig:
    def __init__(self, config):
        self._config = config

    def bg_color(self) -> tuple:
        return tuple(self._config["bg_color"])

    def text_color(self) -> tuple:
        return tuple(self._config["text_color"])

    def thickness(self) -> float:
        return self._config["thickness"]

class DrawConfigCollection:
    def __init__(self, config_path:str):
        self.items = {}
        with open(config_path, 'r', encoding="utf-8")as f:
            items = json.load(f)
        for key in items:
            self.items[key] = DrawConfig(items[key])
    
    def get(self, class_name:str) -> DrawConfig | None:
        return self.items[class_name] if class_name in self.items else None
    
class Detector:
    def __init__(self, camera_confs:CameraConfigCollection):
        self._interpreter = tflite.Interpreter(model_path=camera_confs.model_path())
        self._interpreter.allocate_tensors()
        self._labels = {}
        with open(camera_confs.label_path(), 'r', encoding="utf-8") as f:
            for line in f:
                key, value = line.strip().split("  ")
                self._labels[int(key.strip())] = value.strip()
        self._camera_confs = camera_confs

    def input_details(self):
        return self._interpreter.get_input_details()
    
    def output_details(self):
        return self._interpreter.get_output_details()
    
    def input_shape(self):
        return self.input_details()[0]["shape"]
    
    def input_dtype(self):
        return self.input_details()[0]["dtype"]

    def detect(self, image:Image):
        org_h, org_w = image.raw_image().shape[:2]
        shape = self.input_shape()
        rh, rw = shape[1:3]
        resized = image.resize(rw, rh, self.input_dtype())

        input_data = np.expand_dims(
            resized.image().raw_image(),
            axis=0
        ).astype(self.input_dtype())

        self._interpreter.set_tensor(
            self.input_details()[0]["index"], 
            input_data
        )
        self._interpreter.invoke()

        output_details = self.output_details()
        boxes = self._interpreter.get_tensor(output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(output_details[1]["index"])[0]
        scores = self._interpreter.get_tensor(output_details[2]["index"])[0]

        results = []
        offset = resized.offset()
        restoration_scale = 1 / resized.scale()
        for index in range(len(boxes)):
            cls_name = self._labels[int(classes[index])]
            cls = self._camera_confs.get(cls_name)
            if cls is None or not cls.enabled():
                continue
            score = scores[index]
            if score < cls.threshold():
                continue

            results.append(DetectResult(
                cls_name,
                score,
                (
                    Rectangle(*boxes[index])
                        .scale_xy(rw, rh)
                        .offset(-offset[0], -offset[1])
                        .scale(restoration_scale)
                        .clipping(0, 0, org_w, org_h)
                )
            ))

        return results

class Rectangle:
    def __init__(
            self,
            top_left_y:float,
            top_left_x:float,
            bottom_right_y:float,
            bottom_right_x:float
        ):
        self._tl_y = top_left_y
        self._tl_x = top_left_x
        self._br_y = bottom_right_y
        self._br_x = bottom_right_x
    
    def top_left(self):
        return (int(self._tl_x), int(self._tl_y))

    def bottom_right(self):
        return (int(self._br_x), int(self._br_y))

    def offset(self, x:float, y:float):
        return Rectangle(
            self._tl_y + y,
            self._tl_x + x,
            self._br_y + y,
            self._br_x + x
        )

    def scale_xy(self, x:float, y:float):
        return Rectangle(
            self._tl_y * y,
            self._tl_x * x,
            self._br_y * y,
            self._br_x * x
        )

    def scale(self, scale:float):
        return Rectangle(
            self._tl_y * scale,
            self._tl_x * scale,
            self._br_y * scale,
            self._br_x * scale
        )
    
    def clipping(self, min_x:int, min_y:int, max_x:int, max_y:int):
        return Rectangle(
            max(min_y, min(max_y, self._tl_y)),
            max(min_x, min(max_x, self._tl_x)),
            max(min_y, min(max_y, self._br_y)),
            max(min_x, min(max_x, self._br_x))
        )

class DetectResult:
    def __init__(self, class_name:str, confidence:float, rectangle:Rectangle):
        self._class_name = class_name
        self._confidence = confidence
        self._rectangle = rectangle

    def class_name(self):
        return self._class_name
    
    def rectangle(self):
        return self._rectangle
    
    def confidence(self):
        return self._confidence

class Window:
    def __init__(self, name:str):
        self.window_name = name
        self.already_logged_show_message = False

    def show(self, image:Image):
        if not self.already_logged_show_message:
            self.already_logged_show_message = True
            Logger.log(f"Show Window - {self.window_name}")
        cv2.imshow(self.window_name, image.raw_image())

    def get_key(self, wait_msec:int):
        return cv2.waitKey(wait_msec) & 0xFF

    def close(self):
        Logger.log(f"Close Window - {self.window_name}")
        cv2.destroyWindow(self.window_name)

class Drawer:
    def rectangle(image:Image, rectangle:Rectangle, color, thickness):
        cv2.rectangle(
            image.raw_image(),
            rectangle.top_left(),
            rectangle.bottom_right(),
            color,
            thickness
        )

    def text(image:Image, text:str, position, font, font_scale, color, thickness):
        cv2.putText(
            image.raw_image(),
            text,
            position.top_left(),
            font,
            font_scale,
            color,
            thickness
        )

if __name__ == "__main__":

    LED = 17
    BUZZER = 27

    gpio = None
    camera = None
    window = None

    try:
        gpio = Gpio(Gpio.BCM)
        gpio.off(LED)
        gpio.off(BUZZER)

        window = Window("AI Camera with LED and Buzzer")

        camera_config = CameraConfigCollection("camera_config.json")
        detector = Detector(camera_config)

        draw_config = DrawConfigCollection("draw_config.json")

        camera = Camera(640, 360)
        camera.start()

        while True:
            image = camera.capture()

            results = detector.detect(image)

            detected = len(results) > 0
            for index in range(len(results)):
                result = results[index]
                class_name = result.class_name()
                class_cnf = draw_config.get(class_name)

                Drawer.rectangle(
                    image,
                    result.rectangle(),
                    class_cnf.bg_color(),
                    class_cnf.thickness()
                )
                Drawer.text(
                    image,
                    f"[{class_name}] {result.confidence():.2f}",
                    result.rectangle().offset(5, -10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    class_cnf.text_color(),
                    class_cnf.thickness()
                )

            if detected:
                gpio.on(LED)
                gpio.on(BUZZER)
            else:
                gpio.off(LED)
                gpio.off(BUZZER)

            window.show(image)

            if window.get_key(1) == ord('q'):
                Logger.log("Exit - Pressed 'q'.")
                break
    except KeyboardInterrupt:
        Logger.log("Exit - Pressed 'Ctrl + C'.")
    except Exception as e:
        Logger.log(f"Exit - Raised Exception.('{e}')")
    finally:
        if gpio is not None:
            gpio.cleanup()
        if camera is not None:
            camera.stop()
        if window is not None:
            window.close()
