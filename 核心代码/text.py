from tqdm import tqdm
import time

# 创建三个进度条
progress_bar1 = tqdm(total=100, desc="Progress 1")
progress_bar2 = tqdm(total=100, desc="Progress 2")
progress_bar3 = tqdm(total=100, desc="Progress 3")
progress_bar4 = tqdm(total=100, desc="Progress 4")
progress_bar5 = tqdm(total=100, desc="Progress 5")

time.sleep(0.1)

for i in range(100):
    time.sleep(0.05)
    progress_bar1.update(1)

for i in range(100):
    time.sleep(0.05)
    progress_bar2.update(1)

for i in range(100):
    time.sleep(0.05)
    progress_bar3.update(1)

for i in range(100):
    time.sleep(0.05)
    progress_bar4.update(1)

for i in range(100):
    time.sleep(0.05)
    progress_bar5.update(1)