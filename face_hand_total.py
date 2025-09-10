# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# correct_face = None
# correct_hand = None

# start_time = time.time()
# captured = False

# result_text = ""
# color = (255, 255, 255)

# def extract_hand_landmarks(hand_landmarks):
#     return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

# def normalize_landmarks(landmarks):
#     landmarks = landmarks.reshape(-1, 3)
#     origin = landmarks[0]
#     landmarks -= origin
#     norm = np.linalg.norm(landmarks)
#     if norm > 0:
#         landmarks /= norm
#     return landmarks.flatten()

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     frame = cv2.flip(frame, 1)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     face_results = face_mesh.process(frame_rgb)
#     hand_results = hands.process(frame_rgb)

#     current_face = None
#     if face_results.multi_face_landmarks:
#         face_landmarks = face_results.multi_face_landmarks[0]
#         current_face = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark]).flatten()

#         for lm in face_landmarks.landmark:
#             x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
#             cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
 
#     current_hand = None
#     if hand_results.multi_hand_landmarks:
#         hand_landmarks = hand_results.multi_hand_landmarks[0]
#         current_hand = extract_hand_landmarks(hand_landmarks)
#         normalized_hand = normalize_landmarks(current_hand)

#         mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     current_time = time.time()
#     elapsed = current_time - start_time

#     if not captured:
#         if elapsed < 1:
#             countdown = "5"
#         elif elapsed < 2:
#             countdown = "4"
#         elif elapsed < 3:
#             countdown = "3"
#         elif elapsed < 4:
#             countdown = "2"
#         elif elapsed < 5:
#             countdown = "1"
#         else:
#             countdown = "촬영 중..."
#             # 정답 저장
#             if current_face is not None:
#                 correct_face = current_face.copy()
#                 print("Face captured as reference.")
#             if current_hand is not None:
#                 correct_hand = normalize_landmarks(current_hand)
#                 print("Hand captured as reference.")
#             result_text = "Reference Captured"
#             color = (255, 255, 0)
#             captured = True
#             time.sleep(1) 

#         cv2.putText(frame, countdown, (frame.shape[1]//2 - 50, frame.shape[0]//2),
#                     cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)

#     elif captured:
#         messages = []
#         color = (0, 255, 0)

#         if correct_face is not None and current_face is not None:
#             face_dist = np.linalg.norm(correct_face - current_face)
#             if face_dist < 0.5:
#                 messages.append("Face: True")
#             else:
#                 messages.append("Face: False")
#                 color = (0, 0, 255)

#         if correct_hand is not None and current_hand is not None:
#             hand_dist = np.linalg.norm(correct_hand - normalized_hand)
#             if hand_dist < 0.3:
#                 messages.append("Hand: True")
#             else:
#                 messages.append("Hand: False")
#                 color = (0, 0, 255)

#         result_text = " | ".join(messages)

#     if result_text:
#         cv2.putText(frame, result_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

#     cv2.imshow("Face & Hand Match", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import mediapipe as mp

# # Mediapipe 모듈 초기화
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# mp_drawing = mp.solutions.drawing_utils

# # 웹캠 열기
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # 좌우 반전 및 색 변환
#     frame = cv2.flip(frame, 1)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # 얼굴, 손 검출
#     face_results = face_mesh.process(frame_rgb)
#     hand_results = hands.process(frame_rgb)

#     # 얼굴 랜드마크 표시
#     if face_results.multi_face_landmarks:
#         face_landmarks = face_results.multi_face_landmarks[0]
#         for lm in face_landmarks.landmark:
#             x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
#             cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

#     # 손 랜드마크 표시
#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # 화면 출력
#     cv2.imshow("Face & Hand Detection", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == 27:  # ESC 키 종료
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import asyncio
from bleak import BleakScanner
import numpy as np

# ---------------- BLE 설정 ----------------
TARGET_MAC = "88:4A:EA:62:CA:DD"  # 감지할 블루투스 장치 MAC
ble_device_found = False           # BLE 장치 감지 여부 플래그

# ---------------- Mediapipe 초기화 ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils

# ---------------- 웹캠 열기 ----------------
PI_CAMERA_URL = "http://192.168.137.118:5000/video_feed"
cap = cv2.VideoCapture(PI_CAMERA_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

mediapipe_active = False

# ---------------- BLE 스캔 비동기 함수 ----------------
async def ble_scan_loop():
    global ble_device_found
    while True:
        try:
            devices = await BleakScanner.discover(backend="dotnet")  # WinRT 충돌 방지
            ble_device_found = any(d.address.upper() == TARGET_MAC for d in devices)
        except Exception as e:
            print("BLE 스캔 오류:", e)
        await asyncio.sleep(1)

# ---------------- 메인 루프 (OpenCV + Mediapipe) ----------------
async def main_loop():
    global mediapipe_active
    # BLE 스캔 task 시작
    asyncio.create_task(ble_scan_loop())

    WINDOW_WIDTH = 960    # 창 절반 크기
    WINDOW_HEIGHT = 540
    WINDOW_NAME = "Face & Hand Detection"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    while True:
        success, frame = cap.read()
        if not success:
            await asyncio.sleep(0.01)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        active = ble_device_found

        # BLE 장치가 켜져 있을 때만 Mediapipe 처리
        if active:
            if not mediapipe_active:
                print("BLE 장치 감지됨 → Mediapipe 시작")
                mediapipe_active = True

            face_results = face_mesh.process(frame_rgb)
            hand_results = hands.process(frame_rgb)

            # 얼굴 랜드마크 표시
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # 손 랜드마크 표시
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                    )
        else:
            if mediapipe_active:
                print("BLE 장치 꺼짐 → Mediapipe 일시정지")
                mediapipe_active = False

        # --- 창 안에서 오른쪽 절반에 영상 배치 ---
        frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        screen = 255 * np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)  # 흰 배경
        screen[:, :] = frame_resized  # 화면 전체에 영상 표시 (원하면 오른쪽 절반만 사용 가능)

        cv2.imshow(WINDOW_NAME, screen)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

        await asyncio.sleep(0.001)

# ---------------- 실행 ----------------
asyncio.run(main_loop())

cap.release()
cv2.destroyAllWindows()
