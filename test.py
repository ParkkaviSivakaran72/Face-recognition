import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import time
import math
from datetime import datetime
import csv
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(('SAPI.SpVoice'))
    speak.Speak(str1)
# Get your screen resolution
try:
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    print(f"Screen resolution: {screen_width}x{screen_height}")
except:
    # Default laptop resolution if tkinter is not available
    screen_width = 1920
    screen_height = 1080
    print(f"Using default resolution: {screen_width}x{screen_height}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

ATTENDENCE_DIR = os.path.join(BASE_DIR, "attendence")
os.makedirs(DATA_DIR, exist_ok=True)

def create_gradient_background(width, height, color1, color2, direction='vertical'):
    """Create a gradient background"""
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    if direction == 'vertical':
        for i in range(height):
            ratio = i / height
            for j in range(width):
                background[i, j] = [
                    int(color1[0] * (1 - ratio) + color2[0] * ratio),
                    int(color1[1] * (1 - ratio) + color2[1] * ratio),
                    int(color1[2] * (1 - ratio) + color2[2] * ratio)
                ]
    else:  # horizontal
        for j in range(width):
            ratio = j / width
            for i in range(height):
                background[i, j] = [
                    int(color1[0] * (1 - ratio) + color2[0] * ratio),
                    int(color1[1] * (1 - ratio) + color2[1] * ratio),
                    int(color1[2] * (1 - ratio) + color2[2] * ratio)
                ]
    
    return background

def create_animated_background(width, height, frame_count):
    """Create an animated gradient background"""
    # Create a dynamic color shift based on frame count
    time_factor = (frame_count % 1000) / 1000.0
    
    # Base colors that shift over time
    color1 = [
        int(20 + 30 * math.sin(time_factor * 2 * math.pi)),
        int(30 + 40 * math.sin(time_factor * 2 * math.pi + 2)),
        int(60 + 50 * math.sin(time_factor * 2 * math.pi + 4))
    ]
    
    color2 = [
        int(60 + 40 * math.sin(time_factor * 2 * math.pi + 1)),
        int(40 + 30 * math.sin(time_factor * 2 * math.pi + 3)),
        int(100 + 60 * math.sin(time_factor * 2 * math.pi + 5))
    ]
    
    return create_gradient_background(width, height, color1, color2, 'vertical')

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw main rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_modern_panel(img, x, y, w, h, title, content, color_scheme='blue'):
    """Draw a modern information panel"""
    # Color schemes
    schemes = {
        'blue': {'bg': (40, 40, 80), 'border': (100, 150, 255), 'text': (255, 255, 255), 'title': (150, 200, 255)},
        'green': {'bg': (40, 80, 40), 'border': (100, 255, 150), 'text': (255, 255, 255), 'title': (150, 255, 200)},
        'red': {'bg': (80, 40, 40), 'border': (255, 100, 150), 'text': (255, 255, 255), 'title': (255, 150, 200)},
        'purple': {'bg': (60, 40, 80), 'border': (180, 100, 255), 'text': (255, 255, 255), 'title': (200, 150, 255)}
    }
    
    colors = schemes.get(color_scheme, schemes['blue'])
    
    # Draw panel background with transparency effect
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colors['bg'], -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    # Draw border
    draw_rounded_rectangle(img, (x, y), (x + w, y + h), colors['border'], 2, 10)
    
    # Draw title
    cv2.putText(img, title, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['title'], 2)
    
    # Draw content
    y_offset = 60
    for line in content:
        cv2.putText(img, line, (x + 15, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 1)
        y_offset += 25

def create_phone_frame(img, x, y, w, h):
    """Create a modern phone frame design"""
    # Phone outer frame
    phone_color = (40, 40, 40)
    cv2.rectangle(img, (x - 20, y - 30), (x + w + 20, y + h + 30), phone_color, -1)
    
    # Phone screen bezel
    bezel_color = (20, 20, 20)
    cv2.rectangle(img, (x - 10, y - 20), (x + w + 10, y + h + 20), bezel_color, -1)
    
    # Screen border
    cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 255), 2)
    
    # Home button
    cv2.circle(img, (x + w // 2, y + h + 45), 15, (60, 60, 60), -1)
    cv2.circle(img, (x + w // 2, y + h + 45), 15, (100, 100, 100), 2)
    
    # Camera
    cv2.circle(img, (x + w // 2, y - 45), 8, (30, 30, 30), -1)
    cv2.circle(img, (x + w // 2, y - 45), 8, (80, 80, 80), 2)

# Load training data
names_path = os.path.join(DATA_DIR, "names.pkl")
faces_path = os.path.join(DATA_DIR, "face_datas.pkl")
attendance_path = os.path.join(ATTENDENCE_DIR, "attendance.csv")    

try:
    with open(faces_path, 'rb') as f:
        FACES = pickle.load(f)
    with open(names_path, 'rb') as f:
        LABELS = pickle.load(f)
except FileNotFoundError:
    print("No training data found. Please run the data collection script first.")
    exit()

# Process the loaded data
min_len = min(len(FACES), len(LABELS))
if len(FACES.shape) > 2:
    FACES = FACES[:min_len]
    FACES = FACES.reshape(FACES.shape[0], -1)
else:
    FACES = [np.array(face).flatten() for face in FACES[:min_len]]
    FACES = np.array(FACES)

LABELS = np.array(LABELS[:min_len])

print("FACES shape:", FACES.shape)
print("LABELS shape:", LABELS.shape)
print("Number of people in database:", len(set(LABELS)))

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Start video capture
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a named window and set it to fullscreen
cv2.namedWindow('Face Recognition System', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Face Recognition System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Variables for dynamic display
frame_count = 0
recognition_history = {}
start_time = time.time()
attendance_logged = set()  # Track who has been logged today

column_names = ['name', 'date', 'time']

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = time.time()
    
    # Create animated background
    display_frame = create_animated_background(screen_width, screen_height, frame_count)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Phone screen coordinates (centered)
    phone_screen_w = int(screen_width * 0.25)
    phone_screen_h = int(screen_height * 0.6)
    phone_screen_x = int(screen_width * 0.5) - phone_screen_w // 2
    phone_screen_y = int(screen_height * 0.2)
    
    # Create phone frame
    create_phone_frame(display_frame, phone_screen_x, phone_screen_y, phone_screen_w, phone_screen_h)
    
    # Resize webcam frame to fit phone screen
    resized_frame = cv2.resize(frame, (phone_screen_w, phone_screen_h))
    
    # Place the webcam feed on the phone screen
    display_frame[phone_screen_y:phone_screen_y + phone_screen_h, 
                  phone_screen_x:phone_screen_x + phone_screen_w] = resized_frame
    
    # Process face recognition
    current_recognitions = []
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, (50, 50)).flatten().reshape(1, -1)
        
        # Predict the label
        distances, indices = knn.kneighbors(resized_face)
        closest_label = LABELS[indices[0][0]]
        closest_distance = distances[0][0]
        
        # Get current timestamp
        ts = time.time()
        current_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        current_time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        
        # Check if file exists
        exist = os.path.isfile(attendance_path)
        
        
        label_text = f"{closest_label}"
        color = (0, 255, 0)  # Green for recognized
        status = "RECOGNIZED"
        current_recognitions.append(closest_label)
            
            # Auto-save attendance for recognized faces (only once per day)
        attendance_key = f"{closest_label}_{current_date}"
        if attendance_key not in attendance_logged:
            attendance_logged.add(attendance_key)
            attendance_data = [closest_label, current_date, current_time_str]
            
            speak(f"Attendance logged for {closest_label} at {current_time_str}")
            time.sleep(5)
                
            if exist:
                with open(attendance_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance_data)
            else:
                with open(attendance_path, '+a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(column_names)
                    writer.writerow(attendance_data)
                
            print(f"Attendance logged for {closest_label} at {current_time_str}")
        
        
        # Scale face coordinates to phone screen
        scale_x = phone_screen_w / frame.shape[1]
        scale_y = phone_screen_h / frame.shape[0]
        
        face_x = int(x * scale_x) + phone_screen_x
        face_y = int(y * scale_y) + phone_screen_y
        face_w = int(w * scale_x)
        face_h = int(h * scale_y)
        
        # Draw rectangle and label on phone screen
        cv2.rectangle(display_frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 255, 255), 3)
        cv2.rectangle(display_frame, (face_x, face_y), (face_x + face_w, face_y + face_h), color, 2)
        
        # Label background
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(display_frame, (face_x, face_y - 35), (face_x + label_size[0] + 10, face_y), (0, 0, 0), -1)
        cv2.putText(display_frame, label_text, (face_x + 5, face_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Status indicator
        cv2.putText(display_frame, status, (face_x, face_y + face_h + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Update recognition history
    if current_recognitions:
        for name in current_recognitions:
            if name in recognition_history:
                recognition_history[name] += 1
            else:
                recognition_history[name] = 1
    
    # Left panel - System Status
    status_content = [
        f"Status: {'Active' if ret else 'Inactive'}",
        f"Detected Faces: {len(faces)}",
        f"People in DB: {len(set(LABELS))}",
        f"Frame: {frame_count}",
        f"Uptime: {int(current_time - start_time)}s"
    ]
    draw_modern_panel(display_frame, 50, 100, 300, 200, "SYSTEM STATUS", status_content, 'blue')
    
    # Left panel - Recognition History
    history_content = []
    sorted_recognitions = sorted(recognition_history.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_recognitions[:5]:
        history_content.append(f"{name}: {count}")
    if not history_content:
        history_content = ["No recognitions yet"]
    
    draw_modern_panel(display_frame, 50, 350, 300, 200, "RECOGNITION LOG", history_content, 'green')
    
    # Right panel - Controls
    controls_content = [
        "'q' or ESC - Exit",
        "'r' - Reset history",
        "'s' - Screenshot",
        "'f' - Toggle fullscreen",
        "'h' - Help"
    ]
    draw_modern_panel(display_frame, screen_width - 350, 100, 300, 200, "CONTROLS", controls_content, 'purple')
    
    # Right panel - Attendance Status
    attendance_content = [
        f"Logged Today: {len(attendance_logged)}",
        f"Auto-save: Enabled",
        f"CSV Path: attendance.csv",
        "",
        "Recent Attendance:"
    ]
    
    # Show recently logged attendance
    recent_logged = list(attendance_logged)[-3:]  # Last 3 entries
    for entry in recent_logged:
        name = entry.split('_')[0]
        attendance_content.append(f"✓ {name}")
    
    draw_modern_panel(display_frame, screen_width - 350, 350, 300, 200, "ATTENDANCE", attendance_content, 'red')
    
    # Main title with glow effect
    title_text = "FACE RECOGNITION SYSTEM"
    title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
    title_x = (screen_width - title_size[0]) // 2
    title_y = 80
    
    # Glow effect
    for i in range(5, 0, -1):
        cv2.putText(display_frame, title_text, (title_x, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (100, 100, 255), i * 2)
    
    # Main title
    cv2.putText(display_frame, title_text, (title_x, title_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
    
    # Subtitle
    subtitle = "Real-time AI-powered Face Detection & Recognition with Attendance"
    subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    subtitle_x = (screen_width - subtitle_size[0]) // 2
    cv2.putText(display_frame, subtitle, (subtitle_x, title_y + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    # Bottom status bar
    status_bar_y = screen_height - 50
    cv2.rectangle(display_frame, (0, status_bar_y), (screen_width, screen_height), (30, 30, 30), -1)
    cv2.putText(display_frame, f"Frame: {frame_count} | Faces: {len(faces)} | Time: {int(current_time - start_time)}s", 
               (50, status_bar_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Show current time (fixed the variable name conflict)
    display_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(display_frame, display_time, (screen_width - 200, status_bar_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    cv2.imshow('Face Recognition System', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC key
        break
    elif key == ord('r'):  # Reset recognition history
        recognition_history.clear()
        attendance_logged.clear()
        print("Recognition history and attendance log cleared")
    elif key == ord('f'):  # Toggle fullscreen
        cv2.setWindowProperty('Face Recognition System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif key == ord('s'):  # Save screenshot
        screenshot_path = os.path.join(BASE_DIR, f"screenshot_{frame_count}.jpg")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"Screenshot saved: {screenshot_path}")
    elif key == ord('h'):  # Help
        print("\n=== HELP ===")
        print("q or ESC: Exit the application")
        print("r: Reset recognition history and attendance log")
        print("s: Save screenshot")
        print("f: Toggle fullscreen mode")
        print("h: Show this help")
        print("Attendance is automatically saved when faces are recognized")

video.release()
cv2.destroyAllWindows()
print("Face Recognition System terminated.")
print(f"Total frames processed: {frame_count}")
print(f"Session duration: {int(current_time - start_time)} seconds")
print(f"Total attendance entries logged: {len(attendance_logged)}")
if recognition_history:
    print("Final recognition statistics:")
    for name, count in sorted(recognition_history.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {count} detections")