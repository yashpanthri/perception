import cv2

# Open USB camera (index might be 0, 1, etc.)
cap = cv2.VideoCapture(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Video writer setup
out = cv2.VideoWriter(
    "lane_video_mounted.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    30,  # FPS
    (1280, 720)
)

# Create window (resizable)
cv2.namedWindow("Lane Camera Feed", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Write to video file
    out.write(frame)

    # Show live frame
    cv2.imshow("Lane Camera Feed", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
