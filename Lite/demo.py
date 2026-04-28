import sys
import cv2
import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QComboBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter

from detector import Detector, Get, Size

CANVAS_W = 200
CANVAS_H = 200

MENU_ITEMS = [
    (Get.FACE,       'Face'),
    (Get.FOREHEAD,   'Forehead'),
    (Get.EYES_BROWS, 'Eyes + Brows'),
    (Get.R_EYE_BROW, 'Right Eye + Brow'),
    (Get.L_EYE_BROW, 'Left Eye + Brow'),
    (Get.GLASSES,    'Glasses'),
    (Get.EYES,       'Eyes'),
    (Get.R_EYE,      'Right Eye'),
    (Get.L_EYE,      'Left Eye'),
    (Get.BROWS,      'Brows'),
    (Get.R_BROW,     'Right Brow'),
    (Get.L_BROW,     'Left Brow'),
    (Get.NOSE,       'Nose'),
    (Get.MOUTH,      'Mouth'),
    (Get.MASK,       'Mask'),
    (Get.CHIN,       'Chin'),
]

QSS = '''
QMainWindow, QWidget { background: #0c0c12; color: #c0c0e0; }
QLabel#feed   { background: #060609; border: 1px solid #1a1a2a; }
QLabel#canvas {
    background: #060609;
    border: 1px solid #1a1a2a;
}
QLabel#tag {
    color: #333360;
    font-size: 8px;
    font-family: 'Courier New', monospace;
    letter-spacing: 2px;
    padding: 3px 0;
}
QComboBox {
    background: #0e0e18;
    color: #8080b0;
    border: 1px solid #1e1e30;
    border-radius: 2px;
    padding: 4px 10px;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    min-width: 180px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #444470;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background: #0e0e18;
    color: #8080b0;
    border: 1px solid #1e1e30;
    selection-background-color: #1a1a30;
    selection-color: #c0c0ff;
    font-family: 'Courier New', monospace;
    font-size: 10px;
}
'''

def to_pixmap(img_rgb: np.ndarray) -> QPixmap:
    img_rgb = np.ascontiguousarray(img_rgb)
    h, w, ch = img_rgb.shape
    qi = QImage(img_rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)

class CameraThread(QThread):
    frame = pyqtSignal(np.ndarray)

    def __init__(self, cam_id=0):
        super().__init__()
        self._id      = cam_id
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self._id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self._running:
            ok, f = cap.read()
            if ok:
                self.frame.emit(f)
            self.msleep(30)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LYNX · Section Probe')
        self.setStyleSheet(QSS)
        self._section = Get.FACE
        self._lm      = None
        self._setup()
        self._cam = CameraThread(0)
        self._cam.frame.connect(self._on_frame)
        self._cam.start()

    def _setup(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay  = QHBoxLayout(root)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        self.feed = QLabel()
        self.feed.setObjectName('feed')
        self.feed.setFixedSize(480, 360)
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.feed)

        right     = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(6)
        right_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.canvas = QLabel()
        self.canvas.setObjectName('canvas')
        self.canvas.setFixedSize(CANVAS_W, CANVAS_H)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._blank = QPixmap(CANVAS_W, CANVAS_H)
        self._blank.fill(QColor('#060609'))
        self.canvas.setPixmap(self._blank)

        self.tag = QLabel('')
        self.tag.setObjectName('tag')
        self.tag.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.combo = QComboBox()
        for section_id, label in MENU_ITEMS:
            self.combo.addItem(label, section_id)
        self.combo.currentIndexChanged.connect(self._on_select)

        right_lay.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignHCenter)
        right_lay.addWidget(self.tag,    alignment=Qt.AlignmentFlag.AlignHCenter)
        right_lay.addSpacing(10)
        right_lay.addWidget(self.combo,  alignment=Qt.AlignmentFlag.AlignHCenter)
        right_lay.addStretch()

        lay.addWidget(right)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())

        self.detector = Detector()

    def _on_select(self, idx: int):
        self._section = self.combo.itemData(idx)

    def _on_frame(self, frame: np.ndarray):
        frame = cv2.flip(frame, 1)

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi   = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.feed.setPixmap(
            QPixmap.fromImage(qi).scaled(
                480, 360,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        lm, _ = self.detector.detect(frame)

        if lm is not None:
            ok, crop = self.detector.detect_section(frame, self._section, landmarks=lm)
            if ok and crop is not None:
                ch, cw = crop.shape[:2]
                pix = to_pixmap(crop)
                bg  = QPixmap(CANVAS_W, CANVAS_H)
                bg.fill(QColor('#060609'))

                p = QPainter(bg)
                x  = (CANVAS_W - cw) // 2
                y  = (CANVAS_H - ch) // 2
                p.drawPixmap(x, y, pix)
                p.end()
                self.canvas.setPixmap(bg)
                self.tag.setText(f'{cw} x {ch}')
                return

        self.canvas.setPixmap(self._blank)
        self.tag.setText('—')

    def closeEvent(self, e):
        self._cam.stop()
        super().closeEvent(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
