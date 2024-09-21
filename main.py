import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture("C:/Users/abhirup kumar ghosh/OneDrive/Pictures/Camera Roll/model.mp4")

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

face_drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

selected_landmarks = {
    "right_eye": [33, 133, 160, 159, 158, 144, 153, 154, 163, 7],
    "left_eye": [263, 362, 387, 386, 385, 373, 380, 374, 382, 384],
    "right_eyebrow": [70, 63, 105, 66, 107],
    "left_eyebrow": [336, 296, 334, 293, 300],
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
}

initial_pos = {}

while True:
    sucs, frame = cap.read()
    if not sucs:
        print("Failed to read video.")
        break
    
    colorImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    RgbVideo = faceMesh.process(colorImg)

    if RgbVideo.multi_face_landmarks:
        dists = []  
        for faceLms in RgbVideo.multi_face_landmarks:
            for feature, indices in selected_landmarks.items():
                for idx in indices:
                    h, w, _ = frame.shape
                    x = int(faceLms.landmark[idx].x * w)
                    y = int(faceLms.landmark[idx].y * h)

                    if idx not in initial_pos:
                        initial_pos[idx] = (x, y)
                    
                    initial_x, initial_y = initial_pos[idx]
                    distance = math.sqrt((x - initial_x) ** 2 + (y - initial_y) ** 2)

                    
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  
                    
                    
                    dists.append(f"{idx},{distance:.2f}")

        print("Landmark ID,dist Traveled (pixels)")
        print("\n".join(dists))

    cv2.imshow("VIDEO", frame)
    if cv2.waitKey(10) == ord('a'):
        break
