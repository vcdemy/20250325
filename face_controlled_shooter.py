import pygame
import cv2
import numpy as np
import math
from pygame import mixer
from collections import deque
import mediapipe as mp
import random

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
except:
    print("Warning: Sound files not found. Creating dummy sounds.")
    shoot_sound = mixer.Sound(buffer=b'')
    explosion_sound = mixer.Sound(buffer=b'')

# 初始化 MediaPipe Face Mesh
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
        self.shooting_direction = [0, -1]
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

def cv2_frame_to_pygame_surface(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    return frame

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

# 初始化攝影機
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 移動平均值追蹤器
face_x_avg = MovingAverage(size=5)
eye_direction_x_avg = MovingAverage(size=3)
eye_direction_y_avg = MovingAverage(size=3)

# 遊戲主迴圈
clock = pygame.time.Clock()
running = True
last_shot_time = 0
shot_delay = 200  # 射擊延遲（毫秒）
debug_info = []  # 用於顯示除錯資訊
score = 0

while running:
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
        continue

    # 翻轉畫面（鏡像）
    frame = cv2.flip(frame, 1)  # 1 表示水平翻轉
    debug_frame = frame.copy()
    
    # 轉換為 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe 處理
    results = face_mesh.process(rgb_frame)
    debug_info = []  # 清空除錯資訊
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # 獲取臉部中心點（用於控制飛機位置）
        nose_tip = face_landmarks.landmark[4]
        x_pos = int(nose_tip.x * WINDOW_WIDTH)
        face_x_avg.add(x_pos)
        avg_x = face_x_avg.get_average()
        if avg_x is not None:
            player.target_x = avg_x
        
        # 在除錯影像上標記鼻子位置
        nose_px = int(nose_tip.x * frame.shape[1])
        nose_py = int(nose_tip.y * frame.shape[0])
        cv2.circle(debug_frame, (nose_px, nose_py), 5, (255, 0, 0), -1)
        
        # 獲取眼睛位置（用於控制射擊方向）
        left_eye = face_landmarks.landmark[468]  # 左眼中心
        right_eye = face_landmarks.landmark[473]  # 右眼中心
        
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
        
        # 計算射擊方向（相對於臉部中心）
        dx = eye_center_x - nose_tip.x
        dy = eye_center_y - nose_tip.y
        
        # 平滑化方向
        eye_direction_x_avg.add(dx)
        eye_direction_y_avg.add(dy)
        
        avg_dx = eye_direction_x_avg.get_average()
        avg_dy = eye_direction_y_avg.get_average()
        
        if avg_dx is not None and avg_dy is not None:
            # 計算角度（注意：y軸方向需要反轉）
            angle = math.atan2(-avg_dy, avg_dx)
            # 將角度轉換為度數，90度為向上
            angle_degrees = math.degrees(angle)
            
            # 根據角度計算射擊方向
            # 90度為向上，大於90度向左傾斜，小於90度向右傾斜
            # 限制角度在45-135度之間
            angle_degrees = max(45, min(135, angle_degrees))
            angle_radians = math.radians(angle_degrees)
            
            # 計算射擊方向（修正方向計算）
            player.shooting_direction = [-math.sin(angle_radians), -math.cos(angle_radians)]
            
            # 在除錯影像上畫出射擊方向
            direction_end = (
                int(nose_px + player.shooting_direction[0] * 50),
                int(nose_py + player.shooting_direction[1] * 50)
            )
            cv2.line(debug_frame, (nose_px, nose_py), direction_end, (0, 255, 255), 2)
            
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
    
    # 更新所有精靈
    all_sprites.update()
    
    # 檢測子彈和敵機的碰撞
    hits = pygame.sprite.groupcollide(enemies, bullets, True, True)
    for hit in hits:
        score += 1
        explosion_sound.play()
        # 創建新的敵機
        enemy = Enemy()
        all_sprites.add(enemy)
        enemies.add(enemy)
    
    # 繪製遊戲畫面
    screen.fill(BLACK)
    
    # 繪製半透明的攝影機畫面
    debug_surface = cv2_frame_to_pygame_surface(debug_frame)
    debug_surface = pygame.transform.scale(debug_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
    debug_surface.set_alpha(128)
    screen.blit(debug_surface, (0, 0))
    
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
cap.release()
pygame.quit() 