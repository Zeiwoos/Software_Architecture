import sys
import os
import cv2
import time
import dlib
import numpy as np
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils  # 添加此行
from PySide6.QtCore import QTimer, Qt, QStringListModel
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QListView, QTabWidget, QWidget, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap

class DriverMonitorSystem(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("驾驶员危险行为检测系统")
        self.setGeometry(100, 100, 800, 600)  # Increased width to accommodate feedback panel

        # Initialize the face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # EAR and MAR thresholds for fatigue detection
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.7
        self.EAR_CONSEC_FRAMES = 48  # Approx 1.6 seconds of closed eyes (30 FPS)

        self.eye_blink_counter = 0
        self.alarm_triggered = False

        # Main Window Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Video and Monitor tabs
        self.video_tab = QWidget()
        self.setup_video_tab()

        self.monitor_tab = QWidget()
        self.setup_monitor_tab()

        self.log_tab = QWidget()
        self.setup_log_tab()

        # Adding tabs
        self.tabs.addTab(self.video_tab, "视频")
        self.tabs.addTab(self.monitor_tab, "监控")
        self.tabs.addTab(self.log_tab, "日志")

        # Camera setup
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.recent_frames = []
        self.is_recording = False
        self.logs = []

    def setup_video_tab(self):
        layout = QVBoxLayout()
        self.video_list_view = QListView()
        self.model = QStringListModel()
        self.video_list_view.setModel(self.model)
        self.video_list_view.doubleClicked.connect(self.play_selected_video)
        layout.addWidget(QLabel("已保存的视频文件:"))
        layout.addWidget(self.video_list_view)
        self.video_tab.setLayout(layout)
        self.load_video_files()

    def load_video_files(self):
        saved_videos_dir = "./saved_videos/"
        if not os.path.exists(saved_videos_dir):
            os.makedirs(saved_videos_dir)
        video_files = [f for f in os.listdir(saved_videos_dir) if f.endswith(".avi")]
        self.model.setStringList(video_files)

    def play_selected_video(self, index):
        video_file = self.model.data(index)
        video_path = os.path.join("./saved_videos/", video_file)
        os.system(f"start {video_path}")

    def setup_monitor_tab(self):
        layout = QHBoxLayout()  # Changed to horizontal layout to accommodate feedback
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.detect_button = QPushButton("检测到危险行为")
        self.detect_button.clicked.connect(self.trigger_alarm)
        self.save_button = QPushButton("保存10秒内视频")
        self.save_button.clicked.connect(self.save_recent_video)
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.detect_button)
        video_layout.addWidget(self.save_button)

        # Feedback Panel
        feedback_layout = QVBoxLayout()
        self.feedback_label = QLabel("反馈信息:")
        self.feedback_text = QLabel("")
        feedback_layout.addWidget(self.feedback_label)
        feedback_layout.addWidget(self.feedback_text)

        layout.addLayout(video_layout)
        layout.addLayout(feedback_layout)

        self.monitor_tab.setLayout(layout)

    def setup_log_tab(self):
        layout = QVBoxLayout()
        self.log_label = QLabel("危险行为日志:")
        self.log_list_view = QListView()
        self.log_model = QStringListModel()
        self.log_list_view.setModel(self.log_model)
        layout.addWidget(self.log_label)
        layout.addWidget(self.log_list_view)
        self.log_tab.setLayout(layout)

    def trigger_alarm(self):
        self.logs.append(f"警报触发: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_model.setStringList(self.logs)
        self.detect_button.setText("警报触发中...")
        QTimer.singleShot(2000, self.reset_alarm)

    def reset_alarm(self):
        self.detect_button.setText("检测到危险行为")

    def save_recent_video(self):
        if len(self.recent_frames) == 0:
            return
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        saved_videos_dir = "./saved_videos/"
        if not os.path.exists(saved_videos_dir):
            os.makedirs(saved_videos_dir)
        video_path = os.path.join(saved_videos_dir, f"video_{int(time.time())}.avi")
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        for frame in self.recent_frames:
            out.write(frame)
        out.release()
        self.load_video_files()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                landmarks = self.shape_predictor(gray, face)
                landmarks_np = face_utils.shape_to_np(landmarks)

                # Extract eye and mouth regions
                left_eye = landmarks_np[42:48]
                right_eye = landmarks_np[36:42]
                mouth = landmarks_np[60:68]

                # Calculate EAR and MAR
                ear_left = self.eye_aspect_ratio(left_eye)
                ear_right = self.eye_aspect_ratio(right_eye)
                ear = (ear_left + ear_right) / 2.0
                mar = self.mouth_aspect_ratio(mouth)

                # Check if EAR falls below the threshold for drowsiness
                if ear < self.EAR_THRESHOLD:
                    self.eye_blink_counter += 1
                    if self.eye_blink_counter >= self.EAR_CONSEC_FRAMES:
                        self.feedback_text.setText("警告: 疲劳驾驶检测到！")
                else:
                    self.eye_blink_counter = 0
                    self.feedback_text.setText("")

                if mar > self.MAR_THRESHOLD:
                    self.feedback_text.setText("警告: 打哈欠检测到！")

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

            # Save recent frames for the last 10 seconds
            self.recent_frames.append(frame)
            if len(self.recent_frames) > 200:
                self.recent_frames.pop(0)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[3], mouth[7])  # 上下垂直距离
        B = dist.euclidean(mouth[0], mouth[4])  # 左右水平距离
        mar = A / B
        return mar


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = DriverMonitorSystem()
    window.show()

    window.start_camera()

    sys.exit(app.exec())