import wave
import struct
import numpy as np

def create_sound(filename, frequency, duration, volume=0.5):
    # 設定參數
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    # 生成音波
    t = np.linspace(0, duration, num_samples)
    wave_data = np.sin(2 * np.pi * frequency * t)
    
    # 將數據轉換為16位整數
    wave_data = (wave_data * 32767).astype(np.int16)
    
    # 創建WAV文件
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # 單聲道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        
        # 寫入數據
        for sample in wave_data:
            wav_file.writeframes(struct.pack('h', sample))

# 創建射擊音效
create_sound('shoot.wav', 440, 0.1)
# 創建爆炸音效
create_sound('explosion.wav', 220, 0.2) 