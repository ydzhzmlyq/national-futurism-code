import cv2, os, sys
from tqdm import tqdm

SRC_DIR   = 'movies'          # 原片目录
DST_DIR   = 'frames'          # 输出根目录
INTERVAL  = 2                 # 秒

os.makedirs(DST_DIR, exist_ok=True)

def extract(movie_path, out_dir):
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps * INTERVAL)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            sec = idx // fps
            cv2.imwrite(f'{out_dir}/{sec:05d}.jpg', frame)
        idx += 1
    cap.release()

for f in os.listdir(SRC_DIR):
    if not f.endswith(('.mp4','.mkv','.mov')): continue
    movie_path = os.path.join(SRC_DIR, f)
    out_dir = os.path.join(DST_DIR, os.path.splitext(f)[0])
    os.makedirs(out_dir, exist_ok=True)
    extract(movie_path, out_dir)
print('✅ 分帧完成')