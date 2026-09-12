import os
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))

face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE, "modelo", "haarcascade_frontalface_default.xml")
)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)

    cv2.imshow("Deteccao de Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
