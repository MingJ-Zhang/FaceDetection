from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

cascPath = "./haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        
        
        # Capture frame-by-frame
        ret, frames = video_capture.read()

        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frames)
        if not ret:
                break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
