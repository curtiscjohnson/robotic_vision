import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from deepface import DeepFace


class WebcamApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up webcam
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Set up label for video feed
        self.label = QLabel()

        # Set up buttons
        self.auth_button = QPushButton('Authenticate')
        self.enroll_button = QPushButton('Enroll')
        self.save_button = QPushButton('Save Image')
        self.save_button.setEnabled(False)

        # Set up text input and output regions
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText('Enter user ID')
        self.text_output = QTextEdit()

        # Set up instruction label
        self.instruction_list = [
            'Look straight at the camera.',
        ]

        # Connect buttons to methods
        self.auth_button.clicked.connect(self.authenticate)
        self.enroll_button.clicked.connect(self.enroll)
        self.save_button.clicked.connect(self.save_image)

        # Set up layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.user_id_input)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.auth_button)
        self.layout.addWidget(self.enroll_button)
        self.layout.addWidget(self.text_output)
        self.setLayout(self.layout)

        # Start the camera
        self.timer.start(1000 // 30)

        self.photo_database = './database/'

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame  # Store the frame data
            image = QImage(frame, frame.shape[1], frame.shape[0],
                           QImage.Format_RGB888).rgbSwapped()
            self.label.setPixmap(QPixmap.fromImage(image))

    def authenticate(self):
        # Authentication logic goes here
        self.text_output.append('Authenticating...')

        #capture current image
        ret = DeepFace.find(self.current_frame,
                                db_path=self.photo_database)
        # print(ret)
        min_similarity = 1
        min_name = ''
        for r in ret:
            # print(r['identity'])
            file = r["identity"].iloc[0]
            name = os.path.basename(file).split('.')[0]
            similarity = r["distance"].iloc[0]

            if min_similarity > similarity:
                min_similarity = similarity
                min_name = name

            print(name, similarity)

        if min_similarity < 0.6:
            self.text_output.append(f'Welcome {min_name}!')
        else:
            self.text_output.append(f'Authentication failed!')

    def enroll(self):
        self.save_button.setEnabled(True)
        # Enrollment logic goes here
        self.text_output.append(
            f'Starting enrollment for {self.user_id_input.text()}...')
        self.instruction_index = 0
        self.text_output.append(self.instruction_list[
            self.instruction_index])  # Display first instruction

    def save_image(self):
        # Save image logic goes here
        filepath = f"{self.photo_database}/{self.user_id_input.text()}"

        cv2.imwrite(f"{filepath}.jpg", self.current_frame)
        self.text_output.append(f"Image saved.")

        self.instruction_index += 1
        if self.instruction_index < len(self.instruction_list):
            self.text_output.append(self.instruction_list[
                self.instruction_index])  # Display next instruction
        else:
            self.text_output.append('Enrollment completed.')
            self.save_button.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = WebcamApp()
    window.show()

    sys.exit(app.exec_())
