import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

region_dir = Path(
    "/autofs/bal14/zqwu/CellularTables/TME_benchmark_data/"
    "HNC_Wu2022/processed/regions/s271_c001_v001_r001_reg001"
)

coordinates = pd.read_csv(region_dir / "coordinates.csv").set_index("cell_id")
expression = pd.read_csv(region_dir / "expression.csv").set_index("cell_id")

markers = ["DAPI", "CD3e", "PanCK"]
colors = ["blue", "green", "red"]

# 保证坐标和 expression 的 cell_id 对齐
cell_ids = coordinates.index.intersection(expression.index)
coordinates = coordinates.loc[cell_ids]
expression = expression.loc[cell_ids]

x = np.rint(coordinates["x"].to_numpy() - coordinates["x"].min()).astype(int)
y = np.rint(coordinates["y"].to_numpy() - coordinates["y"].min()).astype(int)

radius = 2
height = y.max() + 2 * radius + 1
width = x.max() + 2 * radius + 1

image = np.zeros((height, width, len(markers)), dtype=np.float32)

# 按 EVA process_local_data.py 中的规则，把细胞信号扩展成小圆盘
offsets = [
    (dx, dy)
    for dy in range(-radius, radius + 1)
    for dx in range(-radius, radius + 1)
    if dx * dx + dy * dy <= radius * radius
]

for marker_index, marker in enumerate(markers):
    values = expression[marker].fillna(0).to_numpy(dtype=np.float32)

    for dx, dy in offsets:
        xx = x + dx + radius
        yy = y + dy + radius

        valid = (
            (xx >= 0) & (xx < width) &
            (yy >= 0) & (yy < height)
        )

        np.maximum.at(
            image[..., marker_index],
            (yy[valid], xx[valid]),
            values[valid],
        )

# 每个通道独立进行 robust scaling
for channel in range(image.shape[-1]):
    values = image[..., channel]
    low, high = np.percentile(values, [1, 99.5])

    if high > low:
        image[..., channel] = np.clip(
            (values - low) / (high - low),
            0,
            1,
        )

# 三通道灰度图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for axis, marker, channel in zip(axes, markers, image.transpose(2, 0, 1)):
    axis.imshow(channel, cmap="gray", vmin=0, vmax=1)
    axis.set_title(marker)
    axis.axis("off")

fig.tight_layout()
fig.savefig("s271_channels.png", dpi=200)
plt.close(fig)

# RGB overlay: DAPI=蓝色, CD3e=绿色, PanCK=红色
overlay = np.zeros((height, width, 3), dtype=np.float32)

for channel, color in enumerate(colors):
    if color == "red":
        overlay[..., 0] += image[..., channel]
    elif color == "green":
        overlay[..., 1] += image[..., channel]
    elif color == "blue":
        overlay[..., 2] += image[..., channel]

overlay = np.clip(overlay, 0, 1)
plt.imsave("s271_overlay.png", overlay)

print("saved s271_channels.png")
print("saved s271_overlay.png")