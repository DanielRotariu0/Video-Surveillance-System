import os

from flask import Flask, render_template, Response, request, url_for, jsonify
from werkzeug.utils import redirect
import threading
import datetime
import numpy as np
import cv2
from redmail import gmail

app = Flask(__name__)

camera = cv2.VideoCapture(0)
alarm_activated = False
authorized = False

# mail notifications
dest_email = "sm.laborator00@gmail.com"
last_email_time = datetime.datetime(2022, 1, 1)

motion_detected_timer = None
motion_detected = False


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            global authorized
            authorized = True
            return redirect(url_for('index'))
        else:
            error = "Invalid username or password"
    return render_template('login.html', error=error)


@app.route('/logout', methods=['POST'])
def logout():
    global authorized
    authorized = False
    return redirect(url_for('login'))


def gen_frames():  # generate frame by frame from camera
    global motion_detected, motion_detected_timer
    previous_frame = None
    while True and camera is not None:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if alarm_activated:
                # 2. Prepare image; grayscale and blur
                prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

                # 2. Calculate the difference
                if previous_frame is None:
                    # First frame; there is no previous one yet
                    previous_frame = prepared_frame
                    continue

                # calculate difference and update previous frame
                diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
                previous_frame = prepared_frame

                # 3. Dilute the image a bit to make differences more seeable; more suitable for contour detection
                kernel = np.ones((5, 5))
                diff_frame = cv2.dilate(diff_frame, kernel, 1)

                # 4. Only take different areas that are different enough (>20 / 255)
                thresh_frame = cv2.threshold(src=diff_frame, thresh=30, maxval=255, type=cv2.THRESH_BINARY)[1]

                if np.sum(thresh_frame) > 0 and not motion_detected:
                    print("Video Surveillance System: Motion detected, timer started")
                    motion_detected = True
                    motion_detected_timer = datetime.datetime.now()

                if motion_detected:
                    now = datetime.datetime.now()
                    if (now - motion_detected_timer).total_seconds() <= 0.75:
                        cv2.imwrite("frame.jpg", frame)
                        motion_detected = False
                        print("Video Surveillance System: Frame saved")

                        if (now - last_email_time).total_seconds() > 30:
                            print("Sending email")
                            mail_task = threading.Thread(target=send_mail_notification)
                            mail_task.start()

                # 6. Find and draw contours
                contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2,
                                 lineType=cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET", "POST"])
def index():
    global authorized
    if authorized:
        global alarm_activated, last_email_time, dest_email, motion_detected
        color = 'red'
        if request.method == 'POST':
            if 'status' in request.form.keys():
                status = request.form['status']
                if status == "1":
                    alarm_activated = True
                    color = 'green'
                    last_email_time = datetime.datetime(2022, 1, 1)
                else:
                    alarm_activated = False
                    color = 'red'

            if 'email' in request.form.keys():
                email = request.form['email']
                print(email)
                if email != '':
                    dest_email = email
                    print("Video Surveillance System: Changed email to: " + dest_email)

        message = "Activated" if alarm_activated else "Deactivated"

        return render_template('index.html', alarm_activated=message, color=color)
    else:
        return redirect(url_for('login'))


def send_mail_notification():
    global last_email_time
    gmail.username = "sm.laborator00@gmail.com"
    gmail.password = "smlaborator01A"
    last_email_time = datetime.datetime.now()
    gmail.send(
        subject="Video Surveillance System",
        receivers=[dest_email],
        html="""
                <h1>Motion detected at """ + str(last_email_time) + """</h1>
                {{ frame }}
            """,
        body_images={"frame": "frame.jpg"}
    )
    print("Video Surveillance System: Email sent at " + str(last_email_time))


@app.route('/info', methods=['GET'])
def info():
    stream = os.popen('/usr/bin/vcgencmd get_mem arm')
    output = stream.read()
    cpu = output.split("=")[1]

    stream = os.popen('/usr/bin/vcgencmd get_mem gpu')
    output = stream.read()
    gpu = output.split("=")[1]

    stream = os.popen('cat /sys/class/thermal/thermal_zone0/temp')
    output = stream.read()
    temperature = int(output)/1000
    return jsonify(cpu=cpu, gpu=gpu, temperature=temperature)


if __name__ == '__main__':
    app.run(debug=True)
