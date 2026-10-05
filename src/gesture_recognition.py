import cv2
import mediapipe as mp
import joblib
import time

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("model/gesture_model.pkl")

# -----------------------------
# MEDIAPIPE
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# CAMERA
# -----------------------------

cap = cv2.VideoCapture(0)

prev_time = time.time()
while True:

    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract 63 features
        features = []

        for landmark in hand_landmarks.landmark:

            features.append(landmark.x)
            features.append(landmark.y)
            features.append(landmark.z)

        # Predict
        prediction = model.predict([features])[0]

        # Confidence
        probabilities = model.predict_proba([features])[0]

        confidence = max(probabilities) * 100

        # Display
        cv2.putText(
            frame,
            f"Gesture: {prediction}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
       
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()