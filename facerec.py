"""
Face Recognition System v2.0
Developed with PyQt6 and OpenCV

Features:
- Real-time face detection and recognition
- Training mode with webcam capture
- Recognition mode with confidence display
- Database management (clean all data)
- Performance optimizations for CPU
- FPS counter and confidence statistics
"""

import sys
import os
import cv2
import numpy as np
import shutil
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QLabel, QMessageBox, QInputDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QElapsedTimer

class FaceRecognitionApp(QMainWindow):
    """Main application window class"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System v2.0")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize UI components
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create control buttons
        self.btn_train = QPushButton("Train Model")
        self.btn_recognize = QPushButton("Start Recognition")
        self.btn_clean = QPushButton("Clean Database")
        self.layout.addWidget(self.btn_train)
        self.layout.addWidget(self.btn_recognize)
        self.layout.addWidget(self.btn_clean)
        
        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_label)
        
        # Initialize video processing timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # System configuration
        self.cap = None  # Video capture object
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = None  # Face recognizer model
        self.label_map = {}  # Mapping of label IDs to names
        self.mode = None  # Current operation mode
        
        # Training parameters
        self.images = []  # Collected face images
        self.target_size = (200, 200)  # Face image dimensions
        self.max_images = 100  # Max images per training session
        self.current_person_dir = ""  # Current training subject's directory
        
        # Performance metrics
        self.frame_count = 0
        self.fps = 0
        self.fps_timer = QElapsedTimer()
        
        # Recognition statistics
        self.recognition_stats = {'total_confidence': 0, 'count': 0}
        self.last_stat_time = datetime.now()
        
        # Connect button signals
        self.btn_train.clicked.connect(self.start_training)
        self.btn_recognize.clicked.connect(self.start_recognition)
        self.btn_clean.clicked.connect(self.cleanup_database)

    def start_training(self):
        """Initialize face collection for model training"""
        # Get subject name from user input
        name, ok = QInputDialog.getText(self, "Input Name", "Enter person's name:")
        if not ok or not name.strip():
            QMessageBox.warning(self, "Error", "Name cannot be empty!")
            return
        
        # Create directory for training images
        self.current_person_dir = os.path.join("dataset", name)
        os.makedirs(self.current_person_dir, exist_ok=True)
        self.images = []
        
        # Initialize video capture with reduced resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 640x480 resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open camera!")
            return
        
        # Start training mode
        self.mode = "training"
        self.fps_timer.start()
        self.timer.start(30)  # ~33 FPS

    def start_recognition(self):
        """Initialize face recognition mode"""
        if not os.path.exists("face_model.yml"):
            QMessageBox.critical(self, "Error", "Model not found! Train first.")
            return
        
        try:
            # Load trained model and label mapping
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read("face_model.yml")
            _, _, self.label_map = self.prepare_training_data("dataset")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open camera!")
            return
        
        # Start recognition mode
        self.mode = "recognition"
        self.fps_timer.start()
        self.last_stat_time = datetime.now()
        self.timer.start(30)

    def update_frame(self):
        """Main video processing loop"""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Calculate FPS
        self.frame_count += 1
        if self.fps_timer.elapsed() > 1000:  # Update every second
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_timer.restart()
        
        # Mirror frame for natural display
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3,  # Balance between speed and accuracy
            minNeighbors=5,    # Fewer neighbors = faster detection
            minSize=(100, 100) # Minimum face size
        )
        
        # Process based on current mode
        if self.mode == "training":
            self.process_training(frame, gray, faces)
        elif self.mode == "recognition":
            self.process_recognition(frame, gray, faces)
        
        # Add FPS counter to display
        cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display processed frame
        self.display_image(frame)

    def process_training(self, frame, gray, faces):
        """Handle face collection for training"""
        for (x, y, w, h) in faces:
            if len(self.images) < self.max_images:
                # Extract and resize face region
                face_roi = cv2.resize(gray[y:y+h, x:x+w], self.target_size)
                self.images.append(face_roi)
                
                # Draw face bounding box and status
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(
                    frame, f"Collected: {len(self.images)}/{self.max_images}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
            else:
                # Stop collection when target reached
                self.finish_training()
                return

    def process_recognition(self, frame, gray, faces):
        """Handle face recognition and display results"""
        current_time = datetime.now()
        for (x, y, w, h) in faces:
            # Prepare face region for recognition
            face_roi = cv2.resize(gray[y:y+h, x:x+w], self.target_size)
            
            # Perform prediction
            label, confidence = self.recognizer.predict(face_roi)
            
            # Determine recognition result
            if confidence < 85:  # Confidence threshold
                name = self.label_map.get(label, "Unknown")
                color = (0, 255, 0)  # Green for recognized
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
            
            # Update recognition statistics
            self.recognition_stats['total_confidence'] += confidence
            self.recognition_stats['count'] += 1
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame, f"{name} ({confidence:.1f})", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
        
        # Print statistics every 10 seconds
        if (current_time - self.last_stat_time) >= timedelta(seconds=10):
            if self.recognition_stats['count'] > 0:
                avg_conf = self.recognition_stats['total_confidence'] / self.recognition_stats['count']
                print(f"[{datetime.now().time()}] Average confidence: {avg_conf:.2f}%")
            else:
                print(f"[{datetime.now().time()}] No faces detected")
            
            # Reset statistics
            self.recognition_stats = {'total_confidence': 0, 'count': 0}
            self.last_stat_time = current_time

    def finish_training(self):
        """Finalize training process and save model"""
        self.timer.stop()
        self.cap.release()
        
        # Save collected images
        existing = len(os.listdir(self.current_person_dir))
        for i, img in enumerate(self.images):
            cv2.imwrite(os.path.join(
                self.current_person_dir, 
                f"{existing + i}.jpg"
            ), img)
        
        # Train new model
        try:
            faces, labels, self.label_map = self.prepare_training_data("dataset")
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save("face_model.yml")
            QMessageBox.information(self, "Success", "Model trained successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def prepare_training_data(self, data_folder):
        """Prepare training dataset from collected images"""
        faces = []
        labels = []
        label_map = {}
        
        # Get list of subjects
        persons = sorted([
            d for d in os.listdir(data_folder) 
            if os.path.isdir(os.path.join(data_folder, d))
        ])
        
        # Process each subject's images
        for label_id, person in enumerate(persons):
            label_map[label_id] = person
            person_dir = os.path.join(data_folder, person)
            
            # Process each image file
            for file in os.listdir(person_dir):
                if not file.lower().endswith(('.jpg', '.png')):
                    continue
                
                img_path = os.path.join(person_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Validate image dimensions
                if img is not None and img.shape == self.target_size:
                    faces.append(img)
                    labels.append(label_id)
        
        return faces, labels, label_map

    def cleanup_database(self):
        """Delete all training data and trained model"""
        # Confirmation dialog
        reply = QMessageBox.question(
            self, 'Confirmation',
            'This will delete ALL trained data and models! Continue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return

        try:
            # Remove dataset directory
            if os.path.exists("dataset"):
                shutil.rmtree("dataset")
                os.makedirs("dataset", exist_ok=True)
            
            # Remove trained model
            if os.path.exists("face_model.yml"):
                os.remove("face_model.yml")
            
            # Reset internal state
            self.recognizer = None
            self.label_map = {}
            
            QMessageBox.information(
                self, "Success", 
                "Database cleaned successfully!\nAll data and models removed."
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Database cleanup failed: {str(e)}"
            )

    def display_image(self, frame):
        """Convert OpenCV frame to QPixmap and display in UI"""
        # Convert color space from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage from numpy array
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def closeEvent(self, event):
        """Cleanup resources when closing window"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())