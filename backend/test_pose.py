import cv2
import mediapipe as mp

# Initialize MediaPipe Pose class
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Setup the Pose function
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

print("Starting Trainova AI Test... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    # and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        # Draw the skeleton connections
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
        
        # Extract coordinates of the Nose (just to prove we have data)
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        cv2.putText(image, 
                    f'Nose X: {nose.x:.2f}, Y: {nose.y:.2f}', 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2, 
                    cv2.LINE_AA)

    cv2.imshow('Trainova Pose Test', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()