import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Specify landmark indices for eyes, eyebrows, and lips
selected_landmarks = {
    "left_eye": [33, 133, 160, 159, 158, 144, 153, 154, 155, 163, 7],
    "right_eye": [263, 362, 387, 386, 385, 373, 380, 374, 381, 382, 384],
    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    "upper_lip": [0, 37, 267, 269, 270, 409, 310, 415, 308],
    "lower_lip": [17, 84, 181, 91, 146, 61, 185, 40, 39]
}

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for feature, indices in selected_landmarks.items():
                for idx in indices:
                    # Get the landmark coordinates
                    x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame.shape[0])

                    # Draw the selected landmarks on the frame
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
