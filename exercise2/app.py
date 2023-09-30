from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from flask_socketio import SocketIO
import pafy
#file chuan
app = Flask(__name__)
socketio = SocketIO(app)
url = "https://www.youtube.com/watch?v=GmqdepTNzEo"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

video_path = "Entrancetest\\Task1\\Videos\\Video2.mp4" #Enter 
cap = cv2.VideoCapture(best.url)
model = YOLO('yolov8n.pt')

# Initialize a variable to count detected "người" (person) objects
object_count = 0
name = ''

@app.route("/")
def index():
    return render_template("index.html")

def read_from_webcam():
    global object_count, name  # Use the global object_count and name variables
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, agnostic_nms=True, vid_stride=1)
        annotated_frame = results[0].plot()
        names = model.names

        # Check the class of the first detected object
        for r in results:
            for c in r.boxes.cls:
                class_name = names[int(c)]
                if class_name == "person":
                    object_count += 1  # Increment object_count if class is "người"
                name = class_name

        # Emit the object count and class name through WebSocket
        socketio.emit('update', {'object_count': object_count, 'class_name': name})

        # Convert the annotated_frame to bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as bytes
        yield (b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n--frame\r\n')

@app.route("/image_feed")
def image_feed():
    return Response(read_from_webcam(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', debug=False)
