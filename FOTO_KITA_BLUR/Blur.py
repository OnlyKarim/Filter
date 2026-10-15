import cv2
import mediapipe as mp
import numpy as np
import os

# Inisialisasi MediaPipe Hands & Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- FUNGSI DETEKSI GESTUR JARI ---
def finger_up(tip, pip, landmarks):
    return landmarks[tip].y < landmarks[pip].y

def is_thumb_only(landmarks):
    # Hanya jempol yang terangkat / thumbs-up (Jempol -> Filter Anjing)
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    # 4 jari lainnya terlipat
    other_down = not index_up and not middle_up and not ring_up and not pinky_up
    # Jempol terangkat ke atas
    thumb_up = (landmarks[4].y < landmarks[3].y) or (landmarks[4].y < landmarks[2].y)

    return other_down and thumb_up

def is_middle_finger(landmarks):
    # Hanya jari tengah yang terangkat (Jari Tengah -> Filter Beruang)
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    return middle_up and not index_up and not ring_up and not pinky_up

def is_one_finger(landmarks):
    # Hanya jari telunjuk yang terangkat (1 Jari -> Filter Kelinci)
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    return index_up and not middle_up and not ring_up and not pinky_up

def is_peace(landmarks):
    # 2 jari terangkat (telunjuk & tengah -> Blur)
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    return index_up and middle_up and not ring_up and not pinky_up

def is_three_fingers(landmarks):
    # 3 jari terangkat (telunjuk, tengah, manis -> Fisheye)
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    return index_up and middle_up and ring_up and not pinky_up

def is_hi5(landmarks):
    # Semua 4 jari atas terangkat dan jempol terbuka (5 Jari -> Normal)
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    thumb_extended = (landmarks[4].y < landmarks[2].y) or (
        abs(landmarks[4].x - landmarks[0].x) > abs(landmarks[2].x - landmarks[0].x)
    )

    return index_up and middle_up and ring_up and pinky_up and thumb_extended


# --- PROSES DAN MEMISAHKAN GAMBAR FILTER HEWAN ---
# Membaca gambar dengan aman di Windows (mendukung path dengan karakter khusus/Unicode seperti '文档')
def load_image_safe(filepath):
    if os.path.exists(filepath):
        data = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return None

def crop_transparent(img_rgba):
    if img_rgba is None or img_rgba.shape[2] < 4:
        return img_rgba
    alpha = img_rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(ys) > 0 and len(xs) > 0:
        return img_rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return img_rgba

script_dir = os.path.dirname(os.path.abspath(__file__))

def load_filter_parts(filename, split_ratio=0.55):
    filepath = os.path.join(script_dir, filename)
    full_img = load_image_safe(filepath)
    if full_img is None:
        full_img = load_image_safe(filename)

    if full_img is not None:
        print(f"[OK] Filter '{filename}' berhasil dimuat.")
        h, w = full_img.shape[:2]
        split_y = int(h * split_ratio)
        ears_raw = full_img[0:split_y, :]
        nose_raw = full_img[split_y:h, :]
        ears = crop_transparent(ears_raw)
        nose = crop_transparent(nose_raw)
        return ears, nose
    else:
        print(f"[ERROR] File '{filename}' tidak ditemukan di:\n{filepath}")
        return None, None

# Muat ketiga filter: Kelinci, Anjing, dan Beruang
bunny_ears, bunny_nose = load_filter_parts("FILTER KELINCI.png", split_ratio=0.55)
dog_ears, dog_nose = load_filter_parts("FILTER ANJING.png", split_ratio=0.55)
bear_ears, bear_nose = load_filter_parts("FILTER BERUANG.png", split_ratio=0.55)


# --- FUNGSI ROTASI & OVERLAY TRANSPARAN (ALPHA BLENDING) ---
def rotate_image(image, angle):
    """Memutar gambar RGBA dengan memperluas kanvas agar tidak terpotong."""
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

def overlay_transparent(background, overlay, center_x, center_y, size=None, angle=0.0):
    """Menempelkan gambar ber-alpha di koordinat pusat (center_x, center_y) dengan ukuran dan rotasi."""
    if overlay is None:
        return background
    
    if size is not None:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
        
    if abs(angle) > 0.5:
        overlay = rotate_image(overlay, angle)
        
    h_ov, w_ov = overlay.shape[:2]
    h_bg, w_bg = background.shape[:2]
    
    x = int(round(center_x - w_ov / 2.0))
    y = int(round(center_y - h_ov / 2.0))
    
    # Batasi koordinat agar tidak keluar dari frame / crash
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_bg, x + w_ov), min(h_bg, y + h_ov)
    
    if x2 <= x1 or y2 <= y1:
        return background
        
    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)
    
    sub_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    sub_background = background[y1:y2, x1:x2]
    
    if sub_overlay.shape[2] == 4:
        alpha = (sub_overlay[:, :, 3] / 255.0)[:, :, np.newaxis]
        sub_background[:] = (alpha * sub_overlay[:, :, :3] + (1.0 - alpha) * sub_background).astype(np.uint8)
    else:
        sub_background[:] = sub_overlay[:, :, :3]
        
    return background


# --- PETA KOORDINAT UNTUK EFEK FISHEYE ---
w_frame, h_frame = 640, 360
cx, cy = w_frame / 2.0, h_frame / 2.0
x_grid, y_grid = np.meshgrid(np.arange(w_frame, dtype=np.float32), np.arange(h_frame, dtype=np.float32))
dx = (x_grid - cx) / cx
dy = (y_grid - cy) / cy
r = np.sqrt(dx**2 + dy**2)
r_norm = r / np.sqrt(2.0)
factor = 0.45 + 0.55 * (r_norm ** 2)
map_x = (cx + dx * cx * factor).astype(np.float32)
map_y = (cy + dy * cy * factor).astype(np.float32)


# --- INISIALISASI KAMERA ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# Mode efek default
current_mode = "normal"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 360))
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Deteksi Gestur Tangan untuk Pindah Mode
    hand_result = hands.process(rgb)
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            if is_thumb_only(landmarks):
                current_mode = "dog"       # Jempol -> Filter Anjing
            elif is_middle_finger(landmarks):
                current_mode = "bear"      # Jari Tengah -> Filter Beruang
            elif is_one_finger(landmarks):
                current_mode = "rabbit"    # 1 Jari (Telunjuk) -> Filter Kelinci
            elif is_peace(landmarks):
                current_mode = "blur"      # 2 Jari -> Blur
            elif is_three_fingers(landmarks):
                current_mode = "fisheye"   # 3 Jari -> Fisheye
            elif is_hi5(landmarks):
                current_mode = "normal"    # 5 Jari -> Normal (Hapus Efek)

    # 2. Terapkan Efek Sesuai Mode Aktif
    if current_mode == "blur":
        frame = cv2.GaussianBlur(frame, (41, 41), 0)
        
    elif current_mode == "fisheye":
        frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
    elif current_mode in ["rabbit", "dog", "bear"]:
        # Jalankan Face Mesh saat mode filter hewan aktif
        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                lms = face_landmarks.landmark

                # Titik referensi wajah
                forehead = np.array([lms[10].x * w_frame, lms[10].y * h_frame])
                chin = np.array([lms[152].x * w_frame, lms[152].y * h_frame])
                nose_tip = np.array([lms[4].x * w_frame, lms[4].y * h_frame])
                left_cheek = np.array([lms[234].x * w_frame, lms[234].y * h_frame])
                right_cheek = np.array([lms[454].x * w_frame, lms[454].y * h_frame])

                # Lebar wajah untuk skala filter
                face_width = float(np.linalg.norm(right_cheek - left_cheek))
                if face_width < 1.0:
                    face_width = 100.0

                # Vektor arah atas kepala (dagu ke dahi)
                up_vec = forehead - chin
                face_height = float(np.linalg.norm(up_vec))
                if face_height < 1.0:
                    up_unit = np.array([0.0, -1.0])
                else:
                    up_unit = up_vec / face_height

                # Sudut kemiringan kepala (roll angle)
                dx = right_cheek[0] - left_cheek[0]
                dy = right_cheek[1] - left_cheek[1]
                tilt_angle = np.degrees(np.arctan2(dy, dx))
                rot_angle = -tilt_angle

                # Tentukan parameter filter sesuai hewan yang dipilih
                if current_mode == "rabbit":
                    cur_ears, cur_nose = bunny_ears, bunny_nose
                    ears_scale = 1.15
                    ears_offset = 0.45
                    nose_scale = 0.48
                    nose_offset = 0.15
                elif current_mode == "dog":
                    cur_ears, cur_nose = dog_ears, dog_nose
                    ears_scale = 1.25
                    ears_offset = 0.28
                    nose_scale = 0.50
                    nose_offset = 0.08
                elif current_mode == "bear":
                    cur_ears, cur_nose = bear_ears, bear_nose
                    ears_scale = 1.15
                    ears_offset = 0.42
                    nose_scale = 0.46
                    nose_offset = 0.12

                # --- 1. Menempelkan Telinga (Di Atas Dahi) ---
                if cur_ears is not None:
                    ears_w = int(face_width * ears_scale)
                    aspect_ears = cur_ears.shape[0] / cur_ears.shape[1]
                    ears_h = int(ears_w * aspect_ears)

                    ears_center = forehead + up_unit * (ears_h * ears_offset)
                    frame = overlay_transparent(
                        frame, cur_ears,
                        center_x=ears_center[0], center_y=ears_center[1],
                        size=(ears_w, ears_h), angle=rot_angle
                    )

                # --- 2. Menempelkan Hidung / Moncong ---
                if cur_nose is not None:
                    nose_w = int(face_width * nose_scale)
                    aspect_nose = cur_nose.shape[0] / cur_nose.shape[1]
                    nose_h = int(nose_w * aspect_nose)

                    nose_center = nose_tip - up_unit * (nose_h * nose_offset)
                    frame = overlay_transparent(
                        frame, cur_nose,
                        center_x=nose_center[0], center_y=nose_center[1],
                        size=(nose_w, nose_h), angle=rot_angle
                    )

    # HUD Status Mode di Layar
    mode_text = {
        "rabbit": "KELINCI (Telunjuk)",
        "dog": "ANJING (Jempol)",
        "bear": "BERUANG (Jari Tengah)",
        "blur": "BLUR (2 Jari)",
        "fisheye": "FISHEYE (3 Jari)",
        "normal": "NORMAL (5 Jari)"
    }.get(current_mode, current_mode.upper())

    overlay_hud = frame.copy()
    cv2.rectangle(overlay_hud, (10, 10), (280, 48), (20, 20, 20), -1)
    cv2.addWeighted(overlay_hud, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Mode: {mode_text}", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

    # Tampilkan output
    cv2.imshow("Multi-effect Camera", frame)

    # Kontrol Keyboard:
    # ESC / 'q': Keluar
    # '1'/'r': Kelinci | '4'/'d': Anjing | '5'/'m': Beruang | '2'/'b': Blur | '3'/'f': Fisheye | '0'/'n': Normal
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('1') or key == ord('r'):
        current_mode = "rabbit"
    elif key == ord('4') or key == ord('d') or key == ord('j'):
        current_mode = "dog"
    elif key == ord('5') or key == ord('m') or key == ord('t'):
        current_mode = "bear"
    elif key == ord('2') or key == ord('b'):
        current_mode = "blur"
    elif key == ord('3') or key == ord('f'):
        current_mode = "fisheye"
    elif key == ord('0') or key == ord('n'):
        current_mode = "normal"

cap.release()
cv2.destroyAllWindows()
