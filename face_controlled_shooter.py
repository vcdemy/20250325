import pygame
import cv2
import numpy as np
import math
from pygame import mixer
from collections import deque
import mediapipe as mp
import random
import sys
import time

# 初始化 Pygame
pygame.init()
mixer.init()

# 設定遊戲視窗
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Face Controlled Space Shooter")

# 顏色定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# 載入音效
try:
    shoot_sound = mixer.Sound("shoot.wav")
    explosion_sound = mixer.Sound("explosion.wav")
except Exception as e:
    print(f"Warning: Sound files not found or error loading sounds: {e}")
    shoot_sound = mixer.Sound(buffer=b'')
    explosion_sound = mixer.Sound(buffer=b'')

# 初始化 MediaPipe Face Mesh
print("正在初始化 MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 移動平均值計算器
class MovingAverage:
    def __init__(self, size=5):
        self.size = size
        self.values = deque(maxlen=size)
        
    def add(self, value):
        self.values.append(value)
        
    def get_average(self):
        if not self.values:
            return None
        return sum(self.values) / len(self.values)

# 載入遊戲素材
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # 創建一個三角形的飛船
        self.image = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, WHITE, [(20, 0), (0, 40), (40, 40)])
        self.original_image = self.image.copy()
        self.rect = self.image.get_rect()
        self.rect.centerx = WINDOW_WIDTH // 2
        self.rect.bottom = WINDOW_HEIGHT - 10
        self.speed = 5
        self.shooting_direction = [0, -1]  # 預設向上射擊
        self.angle = 0
        self.target_x = self.rect.centerx
        self.smoothing = 0.2  # 平滑係數 (0-1)

    def update(self):
        # 平滑移動
        dx = self.target_x - self.rect.centerx
        self.rect.centerx += int(dx * self.smoothing)
        self.rect.centerx = max(20, min(WINDOW_WIDTH - 20, self.rect.centerx))
        
        # 根據射擊方向旋轉飛船
        self.angle = math.degrees(math.atan2(-self.shooting_direction[1], self.shooting_direction[0])) - 90
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, RED, [(15, 30), (0, 0), (30, 0)])
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, WINDOW_WIDTH - self.rect.width)
        self.rect.y = random.randint(-100, -40)
        self.speed = random.randint(2, 5)
        self.health = 1

    def update(self):
        self.rect.y += self.speed
        if self.rect.top > WINDOW_HEIGHT:
            self.rect.x = random.randint(0, WINDOW_WIDTH - self.rect.width)
            self.rect.y = random.randint(-100, -40)

class Bullet(pygame.sprite.Sprite):
    def __init__(self, pos, direction):
        super().__init__()
        self.image = pygame.Surface((4, 12), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255, 255, 0), (2, 2), 2)  # 子彈頭
        pygame.draw.rect(self.image, (255, 200, 0), (1, 4, 2, 8))  # 子彈尾
        self.original_image = self.image.copy()
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.direction = direction
        self.speed = 15
        self.angle = math.degrees(math.atan2(-direction[1], direction[0])) - 90
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        
    def update(self):
        self.rect.x += self.direction[0] * self.speed
        self.rect.y += self.direction[1] * self.speed
        if (self.rect.bottom < 0 or self.rect.top > WINDOW_HEIGHT or
            self.rect.right < 0 or self.rect.left > WINDOW_WIDTH):
            self.kill()

# 初始化攝影機
def init_camera():
    print("正在初始化攝影機...")
    for i in range(2):  # 嘗試前兩個攝影機設備
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"成功開啟攝影機 {i}")
            # 設定攝影機解析度
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # 檢查攝影機設定是否成功
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"攝影機解析度設定為: {actual_width}x{actual_height}")
            
            # 測試讀取一幀
            ret, frame = cap.read()
            if ret:
                print("攝影機測試讀取成功")
                return cap
            else:
                print(f"攝影機 {i} 無法讀取畫面")
                cap.release()
        else:
            print(f"無法開啟攝影機 {i}")
    
    print("錯誤：無法找到可用的攝影機")
    return None

cap = init_camera()
if cap is None:
    print("無法初始化攝影機，程式即將退出")
    pygame.quit()
    sys.exit(1)

# 移動平均值追蹤器
face_x_avg = MovingAverage(size=5)
eye_direction_x_avg = MovingAverage(size=3)
eye_direction_y_avg = MovingAverage(size=3)

# 創建精靈群組
all_sprites = pygame.sprite.Group()
bullets = pygame.sprite.Group()
enemies = pygame.sprite.Group()
player = Player()
all_sprites.add(player)

# 創建敵機
for i in range(5):
    enemy = Enemy()
    all_sprites.add(enemy)
    enemies.add(enemy)

# 遊戲主迴圈
clock = pygame.time.Clock()
running = True
last_shot_time = 0
shot_delay = 200  # 射擊延遲（毫秒）
debug_info = []  # 用於顯示除錯資訊
score = 0
frame_count = 0
last_frame_time = time.time()
fps_update_interval = 1.0  # 每秒更新一次 FPS
fps = 0

def process_face_mesh(frame):
    # 轉換為 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe 處理
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
    
    return results.multi_face_landmarks[0]

while running:
    current_time = time.time()
    frame_count += 1
    
    # 計算並更新 FPS
    if current_time - last_frame_time >= fps_update_interval:
        fps = frame_count / (current_time - last_frame_time)
        frame_count = 0
        last_frame_time = current_time
    
    # 事件處理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # 讀取攝影機畫面
    ret, frame = cap.read()
    if not ret:
        print("錯誤：無法讀取攝影機畫面")
        continue

    # 翻轉畫面（鏡像）
    # frame = cv2.flip(frame, 1)  # 1 表示水平翻轉
    debug_frame = frame.copy()
    
    # 處理臉部特徵
    face_landmarks = process_face_mesh(frame)
    debug_info = []  # 清空除錯資訊
    debug_info.append(f"FPS: {fps:.1f}")
    
    if face_landmarks:
        debug_info.append("偵測到臉部")
        
        # 獲取臉部中心點（用於控制飛機位置）
        nose_tip = face_landmarks.landmark[4]
        # 由於畫面已經翻轉，需要調整 x 座標的計算
        x_pos = int((1 - nose_tip.x) * WINDOW_WIDTH)  # 反轉 x 座標
        face_x_avg.add(x_pos)
        avg_x = face_x_avg.get_average()
        if avg_x is not None:
            player.target_x = avg_x
        
        # 在除錯影像上標記鼻子位置
        nose_px = int(nose_tip.x * frame.shape[1])
        nose_py = int(nose_tip.y * frame.shape[0])
        cv2.circle(debug_frame, (nose_px, nose_py), 5, (255, 0, 0), -1)
        
        # 獲取眼睛位置（用於控制射擊方向）
        left_eye = face_landmarks.landmark[468]
        right_eye = face_landmarks.landmark[473]
        
        # 在除錯影像上標記眼睛位置
        left_eye_px = int(left_eye.x * frame.shape[1])
        left_eye_py = int(left_eye.y * frame.shape[0])
        right_eye_px = int(right_eye.x * frame.shape[1])
        right_eye_py = int(right_eye.y * frame.shape[0])
        cv2.circle(debug_frame, (left_eye_px, left_eye_py), 3, (0, 255, 0), -1)
        cv2.circle(debug_frame, (right_eye_px, right_eye_py), 3, (0, 255, 0), -1)
        
        # 計算眼睛中心點
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2

        eye_center_px = int(eye_center_x * frame.shape[1])
        eye_center_py = int(eye_center_y * frame.shape[0])

        cv2.circle(debug_frame, (eye_center_px, eye_center_py), 3, (0, 255, 0), -1)
        
        
        # 計算射擊方向（相對於臉部中心）
        # dx = eye_center_x - nose_tip.x
        # dy = eye_center_y - nose_tip.y
        dx = left_eye.x - right_eye.x
        dy = left_eye.y - right_eye.y

        
        # 平滑化方向
        eye_direction_x_avg.add(dx)
        eye_direction_y_avg.add(dy)
        
        avg_dx = eye_direction_x_avg.get_average()
        avg_dy = eye_direction_y_avg.get_average()
        
        if avg_dx is not None and avg_dy is not None:
            # 計算角度（注意：y軸方向需要反轉）
            # angle = math.atan2(-avg_dy, avg_dx)
            angle = math.atan2(avg_dy, -avg_dx)
            angle_degrees = math.degrees(angle)
            print(f"{avg_dx:.2f}, {avg_dy:.2f}, {angle_degrees:.2f}")
            
            # 限制角度在 -45 到 45 度之間（相對於垂直向上）
            angle_degrees = max(-45, min(45, angle_degrees))
            angle_radians = math.radians(angle_degrees)
            
            # 計算射擊方向（基於垂直向上的方向）
            player.shooting_direction = [math.sin(angle_radians), -math.cos(angle_radians)]
            
            # 在除錯影像上畫出射擊方向
            direction_end = (
                int(eye_center_px - player.shooting_direction[0] * 50),
                int(eye_center_py + player.shooting_direction[1] * 50)
            )
            cv2.line(debug_frame, (eye_center_px, eye_center_py), direction_end, (0, 255, 255), 2)
            
            # 自動射擊
            current_time = pygame.time.get_ticks()
            if current_time - last_shot_time > shot_delay:
                bullet = Bullet(player.rect.center, player.shooting_direction)
                all_sprites.add(bullet)
                bullets.add(bullet)
                last_shot_time = current_time
                shoot_sound.play()
            
            # 添加除錯資訊
            debug_info.append(f"Angle: {angle_degrees:.1f}°")
            debug_info.append(f"Direction: ({player.shooting_direction[0]:.2f}, {player.shooting_direction[1]:.2f})")
    else:
        debug_info.append("未偵測到臉部")
    
    # 更新所有精靈
    all_sprites.update()
    
    # 檢測子彈和敵機的碰撞
    hits = pygame.sprite.groupcollide(enemies, bullets, True, True)
    for hit in hits:
        score += 1
        explosion_sound.play()
        enemy = Enemy()
        all_sprites.add(enemy)
        enemies.add(enemy)
    
    # 繪製遊戲畫面
    screen.fill(BLACK)
    
    # 繪製半透明的攝影機畫面
    try:
        debug_surface = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
        debug_surface = np.rot90(debug_surface)
        debug_surface = pygame.surfarray.make_surface(debug_surface)
        debug_surface = pygame.transform.scale(debug_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        debug_surface.set_alpha(128)
        screen.blit(debug_surface, (0, 0))
    except Exception as e:
        print(f"錯誤：無法繪製攝影機畫面: {e}")
    
    # 繪製遊戲精靈
    all_sprites.draw(screen)
    
    # 繪製分數
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (WINDOW_WIDTH - 100, 10))
    
    # 繪製除錯資訊
    for i, text in enumerate(debug_info):
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (10, 10 + i * 30))
    
    # 更新顯示
    pygame.display.flip()
    clock.tick(60)

# 清理資源
print("正在關閉攝影機...")
cap.release()
pygame.quit() 