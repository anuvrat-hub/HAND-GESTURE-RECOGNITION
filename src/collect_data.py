import cv2
import mediapipe as mp
import csv
import os

# -----------------------------
# SETTINGS
# -----------------------------

GESTURE = input("Enter gesture name: ")

SAMPLES = 500

# -----------------------------
# CREATE DATASET FOLDER
# -----------------------------

os.makedirs("dataset", exist_ok=True)

csv_file = "dataset/gestures.csv"

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

# -----------------------------
# CSV FILE
# -----------------------------

file_exists = os.path.exists(csv_file)

with open(csv_file, "a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:

        header = []

        for i in range(21):
            header.append(f"x{i}")
            header.append(f"y{i}")
            header.append(f"z{i}")

        header.append("gesture")

        writer.writerow(header)

    count = 0

    print("Get ready...")
    print("Show your gesture to the camera.")
    print("Press Q to stop.")

    while count < SAMPLES:

        success, frame = cap.read()

        if not success:
            print("Camera not detected")
            break

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

            # Extract 63 values
            row = []

            for landmark in hand_landmarks.landmark:

                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            # Add gesture name
            row.append(GESTURE)

            # Save
            writer.writerow(row)

            count += 1

        # Display progress
        cv2.putText(
            frame,
            f"Gesture: {GESTURE}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {count}/{SAMPLES}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Data collection completed.")