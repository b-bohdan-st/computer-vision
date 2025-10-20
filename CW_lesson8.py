import cv2

# face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
# face_cascade_2 = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')

# face_cascade_lbp = cv2.CascadeClassifier('data/lbpcascade/lbpcascade_frontalface_improved.xml')
#для слабих пк

face_net = cv2.dnn.readNetFromCaffe("data/dnn/deploy.prototxt", "data/dnn/model.caffemodel")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # faces = face_cascade.detectMultiScale(gray,
    #     scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
    # # scaleFactor - коефіцієнт масшатування; minNeighbors - кількість перевірок
    #
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     cv2.putText(frame, "Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    #
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = frame[y:y + h, x:x + w]
    #
    #     eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 10, minSize = (15, 15))
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    #
    #     smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor = 1.7, minNeighbors = 10, minSize = (25, 25))
    #     for (sx, sy, sw, sh) in smiles:
    #         cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
    #
    # cv2.putText(frame, f"Faces detected: {len(faces)}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)

    if not ret: break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)

    detections = face_net.forward()

    # for face in detections.shape[2]:
    #

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()