import sys, cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTextBrowser,
    QGridLayout, QVBoxLayout, QSizePolicy, QFileDialog, 
    QLabel, QPushButton, QComboBox
)
from PyQt6.QtCore import Qt, QThread, QTimer, QMutex, QMutexLocker, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from detector import Detector, Draw

QSS = '''
    QMainWindow, QWidget {
        background-color: #1a1a1a;
        color: #d0d0d0;
        font-family: 'Consolas', monospace;
        font-size: 13px;
    }

    QPushButton {
        background-color: #2e2e2e;
        color: #c8c8c8;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 6px 16px;
        min-width: 90px;
    }
    QPushButton:hover  { background-color: #3a3a3a; border-color: #666; }
    QPushButton:pressed{ background-color: #222; }
    QPushButton:disabled{ color: #555; border-color: #333; }
    QPushButton#active {
        background-color: #3c3c3c;
        border-color: #888;
        color: #fff;
    }

    QLabel {
        color: #888;
        font-size: 11px;
    }
    QLabel#ImageLabel {
        background-color: #111;
        border: 1px solid #333;
        border-radius: 3px;
        color: #444;
        qproperty-alignment: AlignCenter;
    }

    QTextBrowser {
        font-family: 'Consolas', monospace;
        font-size: 12px;
        font-weight: 600;
        
        color: #FFFFFF;
        background-color: #222;
        
        border: 1px solid #333;
        border-radius: 4px;
        padding: 4px;
    }

    QScrollBar:vertical {
        width: 0px;
        margin: 0px;
    }
'''

class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, object, object, object, object)

    def __init__(self, detector: Detector):
        super().__init__()
        self._det = detector
        self._running = False
        self._mutex = QMutex()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        
        self._running = True
        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
            ok, frame = cap.read()
            if not ok: break

            lm, bbox, blendshape, transformation = self._det.detect(frame)
            self.frame_ready.emit(frame.copy(), lm, bbox, blendshape, transformation)
    
        cap.release()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait()

class VideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, object, object, object, object)
    finished    = pyqtSignal()

    def __init__(self, path: str, detector: Detector):
        super().__init__()
        self._path = path
        self._det  = detector
        self._running = False
        self._mutex = QMutex()

    def run(self):
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))

        self._running = True
        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break

            ok, frame = cap.read()
            if not ok: break

            lm, bbox, blendshape, transformation = self._det.detect(frame)
            self.frame_ready.emit(frame.copy(), lm, bbox, blendshape, transformation)
            self.msleep(delay)

        cap.release()
        self.finished.emit()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait()

def menu(attr_name: str, attr_type: type):
    box = QComboBox()

    for name in dir(Draw):
        value = getattr(Draw, name)
        if not name.startswith('__') and isinstance(value, attr_type):
            if name != attr_name:
                box.addItem(name, value)

    def update_class():
        setattr(Draw, attr_name, box.currentData())

    box.currentIndexChanged.connect(update_class)

    current_val = getattr(Draw, attr_name, None)
    if current_val is not None:
        idx = box.findData(current_val)
        if idx != -1:
            box.setCurrentIndex(idx)
        else:
            update_class()
    elif box.count() > 0:
        update_class()

    return box

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Face Detector')
        self.resize(820, 520)

        self.detector = Detector()
        self._worker = None

        Detector.ACTIVATE_ENV = True
        Detector.ACTIVATE_PAD = True
        Detector.ACTIVATE_GA = True

        self._build()
        self.setStyleSheet(QSS)

    def _build(self):
        self.setCentralWidget(QWidget())
        main_grid = QGridLayout(self.centralWidget())
        main_grid.setContentsMargins(20, 20, 20, 20)
        main_grid.setSpacing(20)

        main_grid.setRowStretch(0, 2)
        main_grid.setRowStretch(1, 1)
        main_grid.setColumnStretch(0, 1)
        main_grid.setColumnStretch(1, 0)
        main_grid.setColumnStretch(2, 0)
        main_grid.setColumnStretch(3, 0)

        self._canvas_1 = QLabel('No Source')
        self._canvas_1.setObjectName('ImageLabel')
        self._canvas_1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        main_grid.addWidget(self._canvas_1, 0, 0)

        self._textbox_1 = QTextBrowser()
        self._textbox_1.setReadOnly(True)
        main_grid.addWidget(self._textbox_1, 1, 0)

        self._textbox_2 = QTextBrowser()
        self._textbox_2.setReadOnly(True)
        self._textbox_2.setFixedWidth(224)
        main_grid.addWidget(self._textbox_2, 0, 1, 2, 1)

        self._textbox_3 = QTextBrowser()
        self._textbox_3.setReadOnly(True)
        self._textbox_3.setFixedWidth(224)
        main_grid.addWidget(self._textbox_3, 0, 2, 2, 1)

        side_box = QVBoxLayout()
        side_box.setSpacing(15)
        main_grid.addLayout(side_box, 0, 3, 2, 1)

        self._canvas_2 = QLabel('No Faces')
        self._canvas_2.setObjectName('ImageLabel')
        self._canvas_2.setFixedSize(224, 224)
        self._canvas_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_box.addWidget(self._canvas_2, alignment=Qt.AlignmentFlag.AlignHCenter)

        side_box.addSpacing(30)
        side_box.addWidget(QLabel('Draw.MODE'), alignment=Qt.AlignmentFlag.AlignHCenter)
        side_box.addWidget(menu('MODE', int), alignment=Qt.AlignmentFlag.AlignHCenter)

        side_box.addSpacing(30)
        side_box.addWidget(QLabel('Draw.COLOR'), alignment=Qt.AlignmentFlag.AlignHCenter)
        side_box.addWidget(menu('COLOR', tuple), alignment=Qt.AlignmentFlag.AlignHCenter)

        side_box.addSpacing(30)

        self._cam_btn = QPushButton('Camera')
        self._img_btn = QPushButton('Image / Video')
        self._stop_btn = QPushButton('Stop')
        self._stop_btn.setEnabled(False)

        for btn in (self._cam_btn, self._img_btn, self._stop_btn):
            side_box.addWidget(btn)

        side_box.addStretch()

        self._status = QLabel('Idle')
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_box.addWidget(self._status)

        self._cam_btn.clicked.connect(self._open_camera)
        self._img_btn.clicked.connect(self._open_media)
        self._stop_btn.clicked.connect(self._stop)

    def _update_frame(self, frame: np.ndarray, landmarks, bbox, blendshapes, transformation):
        copy = frame.copy()
        if landmarks is not None:
            frame = self.detector.draw(frame, landmarks, bbox)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        sh, sw = frame.shape[:2]
        cw, ch = self._canvas_1.width(), self._canvas_1.height()
        scale = min(cw / sw, ch / sh)
        dw, dh = int(sw * scale), int(sh * scale)

        self._canvas_1.setPixmap(
            self.pixmap(cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA))
        )

        if landmarks is not None and bbox is not None:
            aligned = self.detector.align(copy, landmarks, bbox)
            if aligned is not None:
                self._canvas_2.clear()
                self._canvas_2.setPixmap(
                    self.pixmap(cv2.resize(aligned, (224, 224), interpolation=cv2.INTER_LANCZOS4))
                )

                y, p, r, ys, ps, rs = self.detector.euler_angles(frame, landmarks, transformation)
                le, re, les, res = self.detector.eye_aspect_ratio(frame, landmarks)
                mar, ms = self.detector.mouth_aspect_ratio(frame, landmarks)

                _, env = self.detector.env_status
                _, pad = self.detector.pad_status
                _, g,a = self.detector.ga_status

                text_23 = []
                for group, features in blendshapes.items():
                    text_23.append(f'= {group.upper()}')
                    for feature, value in features.items():
                        text_23.append(f'  {feature:<15} :: {value:.4f}')
                        text_23.append(f'  {"--"*round(value*10)}')
                    text_23.append('')

                self._textbox_1.setText((
                    f'YAW   :: {y:+08.4f} :: {ys}\n'
                    f'PITCH :: {p:+08.4f} :: {ps}\n'
                    f'YAW   :: {r:+08.4f} :: {rs}\n\n'
                    f'EAR :: {le:.4f} [{les}]  ||  {re:.4f} [{res}] ::  ||  MAR :: {mar:.4f} :: {ms}\n\n'

                    f'Brightness :: {env["brightness"]:+08.4f} :: {env["brightness_flag"]}\n'
                    f'Occupancy  :: {env["occupancy"]:+08.4f} :: {env["occupancy_flag"]}\n'
                    f'Jitter     :: {env["jitter"]:+08.4f} :: {env["jitter_flag"]}\n\n'

                    f'PAD :: Ready {pad["ready"]} :: Live {pad["live"]} :: Score {pad["score"]:.4f}\n\n'
                    f'Gender :: {g}   ||   Age :: {a}'
                ))

                self._textbox_2.setText('\n'.join(text_23[:46]))
                self._textbox_3.setText('\n'.join(text_23[46:]))

        else:
            self._canvas_2.clear()
            self._canvas_2.setText('No Faces')

    def _video_finished(self):
        self._enable_btns(True, 'Video Ended')

    def _stop_worker(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def _enable_btns(self, enabled: bool, label: str = ''):
        for btn in (self._cam_btn, self._img_btn):
            btn.setEnabled(enabled)
        self._stop_btn.setEnabled(not enabled)

        if label:
            self._status.setText(label)

    def _open_camera(self):
        self._stop_worker()
        self._worker = CameraWorker(self.detector)
        self._worker.frame_ready.connect(self._update_frame)
        self._worker.start()
        self._enable_btns(False, 'Camera')

    def _open_media(self):
        self._stop_worker()
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image / Video', '', 'Images (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mov *.mkv)'
        )
        if not path:
            return
    
        if path.split('.')[-1] in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
            frame = cv2.imread(path)
            if frame is None:
                self._status.setText('Failed to load image')
                return

            self.detector = Detector()
            lm, bbox, blendshape = self.detector.detect(frame)

            self._update_frame(frame, lm, bbox, blendshape)
            self._status.setText('Image')
        else:
            self._worker = VideoWorker(path, self.detector)
            self._worker.frame_ready.connect(self._update_frame)
            self._worker.finished.connect(self._video_finished)
            self._worker.start()
            self._enable_btns(False, 'Video')

    def _stop(self):
        def clear():
            self._canvas_1.clear()
            self._canvas_1.setText('No Source')
            self._canvas_2.clear()
            self._canvas_2.setText('No Faces')

        self._stop_worker()
        self._enable_btns(True, 'Idle')
        # QTimer.singleShot(500, clear)

    def closeEvent(self, event):
        self._stop_worker()
        super().closeEvent(event)

    @staticmethod
    def pixmap(img_rgb: np.ndarray) -> QPixmap:
        h, w, ch = img_rgb.shape
        return QPixmap.fromImage(
            QImage(img_rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())