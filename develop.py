import cv2
import mediapipe as mp
import numpy as np
import ctypes
import pyautogui

# 마우스 안전모드 비활성화
pyautogui.FAILSAFE = False

# 모니터 해상도 가져오기 (Windows 환경)
try:
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
except Exception as e:
    screen_width, screen_height = 1920, 1080
print(f"Monitor resolution: {screen_width} x {screen_height}")

# 두 개의 웹캠 열기: 내부캠(인덱스 0)와 외부캠(인덱스 1)
cap_internal = cv2.VideoCapture(0)
cap_external = cv2.VideoCapture(1)
if not cap_internal.isOpened() or not cap_external.isOpened():
    print("Cannot open one or both cameras")
    exit()

# 두 캠 해상도 동일하게 설정 (1280 x 720)
cap_internal.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_internal.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_external.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_external.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Mediapipe FaceMesh 초기화 (각각 별도 객체)
mp_face_mesh = mp.solutions.face_mesh
face_mesh_internal = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh_external = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 사용할 왼쪽 iris landmark 인덱스 (왼쪽 눈만 사용)
LEFT_IRIS = [469, 470, 471, 472]

def get_landmark_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def compute_pupil_center(landmarks, frame_width, frame_height):
    xs, ys = [], []
    for idx in LEFT_IRIS:
        x, y = get_landmark_coords(landmark=landmarks[idx], width=frame_width, height=frame_height)
        xs.append(x)
        ys.append(y)
    return (int(np.mean(xs)), int(np.mean(ys)))

####################################
# Calibration Phase (Combined Cameras)
####################################
print("Calibration:\n - Ensure both cameras are capturing you.\n - Move your eye to cover your full gaze range.\n - Press 'c' to finish calibration.")
calib_points = []  # 누적할 정규화 좌표 (norm_x, norm_y) (두 캠 결합)

while True:
    ret_int, frame_int = cap_internal.read()
    ret_ext, frame_ext = cap_external.read()
    if not ret_int or not ret_ext:
        print("One of the cameras failed to capture a frame.")
        break

    # 좌우 반전
    frame_int = cv2.flip(frame_int, 1)
    frame_ext = cv2.flip(frame_ext, 1)
    
    # 두 캠 모두 해상도 1280x720 (내부캠: 1280x720, 외부캠: 1280x720)
    h_int, w_int, _ = frame_int.shape
    h_ext, w_ext, _ = frame_ext.shape
    
    rgb_int = cv2.cvtColor(frame_int, cv2.COLOR_BGR2RGB)
    rgb_ext = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
    
    results_int = face_mesh_internal.process(rgb_int)
    results_ext = face_mesh_external.process(rgb_ext)
    
    pupil_int = None
    pupil_ext = None
    if results_int.multi_face_landmarks:
        landmarks_int = results_int.multi_face_landmarks[0].landmark
        pupil_int = compute_pupil_center(landmarks_int, w_int, h_int)
        cv2.circle(frame_int, pupil_int, 5, (0,255,0), -1)
    if results_ext.multi_face_landmarks:
        landmarks_ext = results_ext.multi_face_landmarks[0].landmark
        pupil_ext = compute_pupil_center(landmarks_ext, w_ext, h_ext)
        cv2.circle(frame_ext, pupil_ext, 5, (0,255,0), -1)
        
    # 정규화 좌표 계산: 각 캠에서 얻은 좌표를 계산 후 결합
    norm_x_int, norm_y_int = (None, None)
    norm_x_ext, norm_y_ext = (None, None)
    if pupil_int is not None:
        norm_x_int = pupil_int[0] / w_int
        norm_y_int = pupil_int[1] / h_int
    if pupil_ext is not None:
        norm_x_ext = pupil_ext[0] / w_ext
        norm_y_ext = pupil_ext[1] / h_ext
    # 수평은 두 캠의 평균, 수직은 두 캠의 **평균**로 결합
    if norm_x_int is not None and norm_x_ext is not None:
        norm_x = (norm_x_int + norm_x_ext) / 2.0
        norm_y = (norm_y_int + norm_y_ext) / 2.0
    elif norm_x_int is not None:
        norm_x, norm_y = norm_x_int, norm_y_int
    elif norm_x_ext is not None:
        norm_x, norm_y = norm_x_ext, norm_y_ext
    else:
        continue

    calib_points.append((norm_x, norm_y))
    
    # 시각적 피드백: 두 프레임을 좌우 결합하여 보여줌
    combined_calib = np.hstack((frame_int, frame_ext))
    cv2.putText(combined_calib, "Calibration: Press 'c' to finish", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Calibration", combined_calib)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
cv2.destroyWindow("Calibration")
if len(calib_points) == 0:
    print("No calibration data collected.")
    cap_internal.release()
    cap_external.release()
    exit()

# 캘리브레이션 좌표의 최소/최대 계산 (정규화)
norm_xs = [p[0] for p in calib_points]
norm_ys = [p[1] for p in calib_points]
min_norm_x, max_norm_x = min(norm_xs), max(norm_xs)
min_norm_y, max_norm_y = min(norm_ys), max(norm_ys)
print(f"Calibration bounding box (normalized, union): x: {min_norm_x:.3f}-{max_norm_x:.3f}, y: {min_norm_y:.3f}-{max_norm_y:.3f}")

####################################
# Tracking Phase (with Mouse Control)
####################################
# 마우스 제어 시작 시, 커서를 모니터 중앙으로 설정
prev_mouse = (screen_width / 2, screen_height / 2)
pyautogui.moveTo(int(prev_mouse[0]), int(prev_mouse[1]))

print("Tracking started. Press 'q' or ESC to quit.")
smoothing_factor = 0.2  # 지수 스무싱 계수 (0 ~ 1)

while True:
    ret_int, frame_int = cap_internal.read()
    ret_ext, frame_ext = cap_external.read()
    if not ret_int or not ret_ext:
        print("A camera frame is missing during tracking.")
        break
    frame_int = cv2.flip(frame_int, 1)
    frame_ext = cv2.flip(frame_ext, 1)
    # 두 캠 모두 1280x720
    h_int, w_int, _ = frame_int.shape
    h_ext, w_ext, _ = frame_ext.shape
    
    rgb_int = cv2.cvtColor(frame_int, cv2.COLOR_BGR2RGB)
    rgb_ext = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
    
    results_int = face_mesh_internal.process(rgb_int)
    results_ext = face_mesh_external.process(rgb_ext)
    
    pupil_int = None
    pupil_ext = None
    if results_int.multi_face_landmarks:
        landmarks_int = results_int.multi_face_landmarks[0].landmark
        pupil_int = compute_pupil_center(landmarks_int, w_int, h_int)
    if results_ext.multi_face_landmarks:
        landmarks_ext = results_ext.multi_face_landmarks[0].landmark
        pupil_ext = compute_pupil_center(landmarks_ext, w_ext, h_ext)
        
    norm_x_int, norm_y_int = (None, None)
    norm_x_ext, norm_y_ext = (None, None)
    if pupil_int is not None:
        norm_x_int = pupil_int[0] / w_int
        norm_y_int = pupil_int[1] / h_int
    if pupil_ext is not None:
        norm_x_ext = pupil_ext[0] / w_ext
        norm_y_ext = pupil_ext[1] / h_ext
        
    if norm_x_int is not None and norm_x_ext is not None:
        norm_x = (norm_x_int + norm_x_ext) / 2.0
        norm_y = (norm_y_int + norm_y_ext) / 2.0
    elif norm_x_int is not None:
        norm_x, norm_y = norm_x_int, norm_y_int
    elif norm_x_ext is not None:
        norm_x, norm_y = norm_x_ext, norm_y_ext
    else:
        continue

    # 매핑: 캘리브레이션 bounding box(합집합)을 기준으로 (0~1) 상대 좌표로 변환
    relative_norm_x = (norm_x - min_norm_x) / (max_norm_x - min_norm_x)
    relative_norm_y = (norm_y - min_norm_y) / (max_norm_y - min_norm_y)
    # 목표 마우스 좌표: 모니터 전체에 매핑 (모니터 중앙이 0.5, 0.5)
    target_mouse_x = relative_norm_x * screen_width
    target_mouse_y = relative_norm_y * screen_height

    # 지수 스무싱 적용
    smoothed_mouse_x = prev_mouse[0] + smoothing_factor * (target_mouse_x - prev_mouse[0])
    smoothed_mouse_y = prev_mouse[1] + smoothing_factor * (target_mouse_y - prev_mouse[1])
    pyautogui.moveTo(int(smoothed_mouse_x), int(smoothed_mouse_y))
    prev_mouse = (smoothed_mouse_x, smoothed_mouse_y)
    
    # 디버그용: 두 캠의 화면을 좌우 결합하여 표시
    combined_tracking = np.hstack((frame_int, frame_ext))
    cv2.imshow("Tracking", combined_tracking)
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:
        break

cap_internal.release()
cap_external.release()
face_mesh_internal.close()
face_mesh_external.close()
cv2.destroyAllWindows()
