import cv2
from collections import deque
import numpy as np


class LiveStream:
    def __init__(self, web=True, cam="web", ip=None, num_blocks=10):
        self.web = web
        self.cam = cam
        self.ip = ip
        self.num_blocks = int(num_blocks)
        self.buffer = deque(maxlen=10)

        # Инициализация детектора лиц
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if self.cam == "phone":
            print("Поток с телефона")
            # Адрес видеопотока
            self.capture = cv2.VideoCapture(ip + "/video")
            if not self.capture.isOpened():
                print("Error: Could not open video capture.")

            # Увеличение размера эффекта
            self.scale_factor = 1.2

        elif self.cam == "web":
            cam_port = 0
            print("Поток с вебкамеры")
            self.capture = cv2.VideoCapture(cam_port)
            if not self.capture.isOpened():
                print("Error: Could not open video capture.")

            # Увеличение размера эффекта
            self.scale_factor = 1.2

    def update_buffer(self, coords: tuple):
        self.buffer.append(coords)

    def pixelate_image(self, image, blocks=3):
        # divide the input image into NxN blocks
        (h, w) = image.shape[:2]
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")
        # loop over the blocks in both the x and y direction
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                # compute the starting and ending (x, y)-coordinates
                # for the current block
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                # extract the ROI using NumPy array slicing, compute the
                # mean of the ROI, and then draw a rectangle with the
                # mean RGB values over the ROI in the original image
                roi = image[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
        # return the pixelated blurred image
        return image

    def generate_frames(self):
        while True:
            try:
                _, frame = self.capture.read()
                if frame is None or frame.size == 0:
                    print("Error: Empty frame received.")
                    continue

                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.detector.detectMultiScale(img_gray, 1.1, 19)

                for (x, y, w, h) in faces:
                    # bbox лица
                    # frame = cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h),
                    #                       color=(0, 255, 0), thickness=2)

                    # Вычисления чтобы наложить отмасштабированный эффект на лицо
                    # center_x, center_y = x + w // 2, y + h // 2 - int(h*0.2)    # Смещение в сторону лба
                    center_x, center_y = x + w // 2, y + h // 2
                    scaled_width = int(w * self.scale_factor)
                    scaled_height = int(h * self.scale_factor)

                    # Определение границ вставки
                    y1, y2 = center_y - scaled_height // 2, center_y + scaled_height // 2
                    x1, x2 = center_x - scaled_width // 2, center_x + scaled_width // 2

                    # # Выделение области лица
                    face_img = frame[y1:y2, x1:x2]

                    # Замена соответствующей области в frame
                    frame[y1:y2, x1:x2] = self.pixelate_image(face_img, blocks=self.num_blocks)

                    self.update_buffer((x1, y1, x2, y2))

                    # Накладываем текст на bbox
                    # cv2.putText(frame, "no text", (x, y-10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color=(0, 255, 0), thickness=2)

                if not faces:
                    x1, y1, x2, y2 = self.buffer[-1]
                    frame[y1:y2, x1:x2] = self.pixelate_image(frame[y1:y2, x1:x2], blocks=self.num_blocks)

                # Отображение кадра
                if self.web:
                    # Веб версия
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Десктоп версия
                    cv2.imshow('livestream', frame)
            except Exception as e:
                print(f"{e}")
                # Отображение кадра
                if self.web:
                    # Веб версия
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Десктоп версия
                    cv2.imshow('livestream', frame)

            finally:
                # Выход из цикла при нажатии клавиши 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def __del__(self):
        # Освобождение ресурсов
        self.capture.release()
        cv2.destroyAllWindows()
