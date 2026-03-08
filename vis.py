import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

y, sr = librosa.load("loop.mp3")

start = int(3 * sr)
y_cut = y[start:]

n_repeat = 4
y_double = np.concatenate([y_cut] * n_repeat)
y_double = np.concatenate([y_double, y_cut[:int(2*sr)]] )
segment_duration = len(y_cut) / sr
#print(len(y_cut), sr, len(y_cut)/sr, segment_duration, len(y_cut)/11)

C = librosa.cqt(
    y_double,
    sr=sr,
    bins_per_octave=12,
    n_bins=84
)

C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
C_db = np.clip(C_db, -100, 0)

C_vis = C_db + 100

plt.figure(figsize=(12,6))
	
norm = PowerNorm(gamma=4, vmin=0, vmax=100)

img = librosa.display.specshow(
    C_vis,
    sr=sr,
    x_axis=None,
    y_axis='cqt_note',
    bins_per_octave=12,
    norm=norm
)

# 你想显示的 dB
db_labels = [-100, -30, -10, -3, 0]

# 根据变换计算刻度位置
ticks = [d + 100 for d in db_labels]

cbar = plt.colorbar()
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{d} dB" for d in db_labels])
cbar.set_label("强度")

hop_length = 512
frames_per_second = sr / hop_length

ticks = [
    (i+1) * segment_duration * frames_per_second
    for i in range(n_repeat)
]

labels = [f"第{i+1}遍结束" for i in range(n_repeat)]

for t in ticks:
    plt.axvline(t, color='white', linestyle='--', alpha=0.8)

plt.xticks(ticks, labels)

plt.xlabel("时间")
plt.ylabel("音高")
plt.title("小老虎按开关视频频谱分析")

plt.tight_layout()
plt.show()