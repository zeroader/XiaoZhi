"""Create small, deterministic runtime assets for the 320x240 LCD."""
from pathlib import Path

from PIL import Image


root = Path(__file__).resolve().parents[1]
source = root / "main" / "assets" / "vision"
target = source / "runtime"
target.mkdir(parents=True, exist_ok=True)

# The final background includes enlarged status cards and a speech-text channel.
Image.open(source / "vision_dashboard_5.png").convert("RGB").resize(
    (320, 240), Image.Resampling.LANCZOS
).save(target / "vision_dashboard_5.png", optimize=True)


def fit_transparent(input_path: Path, output_path: Path, size: tuple[int, int]):
    image = Image.open(input_path).convert("RGBA")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.alpha_composite(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    canvas.save(output_path, optimize=True)


fit_transparent(source / "correct_posture.png", target / "vision_posture_correct.png", (76, 70))
fit_transparent(source / "incorrect_posture.png", target / "vision_posture_incorrect.png", (76, 70))

# emotion_emoji.png is a color-coded 4-column x 2-row sprite sheet:
# angry, contempt, disgust, fear / happy, neutral, sad, surprise.
labels = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
sheet = Image.open(source / "emotion_emoji.png").convert("RGBA")
cell_w, cell_h = sheet.width // 4, sheet.height // 2
for index, label in enumerate(labels):
    cell = sheet.crop(((index % 4) * cell_w, (index // 4) * cell_h,
                       (index % 4 + 1) * cell_w, (index // 4 + 1) * cell_h))
    # Ignore the faint outer glow when fitting, while preserving the generated
    # alpha at the face edge. This gives all eight icons the same visual size.
    bounds = cell.getchannel("A").point(lambda value: 255 if value >= 240 else 0).getbbox()
    if bounds is None:
        raise ValueError(f"Missing emoji in sprite cell: {label}")
    cell = cell.crop(bounds)
    cell.thumbnail((56, 56), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (58, 58), (0, 0, 0, 0))
    canvas.alpha_composite(cell, ((58 - cell.width) // 2, (58 - cell.height) // 2))
    canvas.save(target / f"vision_emotion_{label}.png", optimize=True)

print(f"Generated runtime assets in {target}")
