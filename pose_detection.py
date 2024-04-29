import cv2
import mediapipe as mp

# Initialize the MediaPipe Pose module.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Change model complexity to 0 for the most lightweight model
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utility for posing landmark connections.
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Failed to open webcam.")
    exit()

while True:
    # Read a frame from the webcam.
    isTrue, frame = cap.read()
    if not isTrue:
        print("Error: Failed to grab frame.")
        break

    # Flip the frame horizontally for a mirror view, convert BGR to RGB.
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # Process the image and detect poses.
    results = pose.process(image)

    # Convert the image back to BGR for displaying.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # If pose landmarks are detected, draw them.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Display the image.
    cv2.imshow('MediaPipe Pose', image)

    # Exit loop if ESC is pressed.
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all OpenCV windows to free resources.
cap.release()
cv2.destroyAllWindows()
