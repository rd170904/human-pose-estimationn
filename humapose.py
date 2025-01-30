import cv2 as cv
import matplotlib.pyplot as plt
cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FPS, 10)
cap.set(3, 800)  # Set width
cap.set(4, 800)  # Set height

if not cap.isOpened():
    cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise IOError("cannot open video capture")

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()  # Unpacking the frame properly
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # Process frame with the model
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output, first 19 elements

    assert len(BODY_PARTS) == out.shape[1]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        threshold = 0.1
        points.append((int(x), int(y)) if conf > threshold else None)

    # Draw the pose lines
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # Display performance
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Display the frame
    cv.imshow("Pose Estimation", frame)

    # Break the loop if the user presses 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):  # Stop when 'q' is pressed
        break

# Release the capture and close windows
cap.release()
cv.destroyAllWindows()
