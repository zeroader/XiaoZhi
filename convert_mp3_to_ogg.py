"""将 main/voice 目录中的 MP3 递归转换为 OGG。

运行前请确保已安装 ffmpeg，并且 ffmpeg 已加入 PATH。
"""

from pathlib import Path
import shutil
import subprocess
import sys


VOICE_DIR = Path(__file__).resolve().parent / "main" / "voice"


def convert_one(mp3_path: Path) -> bool:
    """转换单个 MP3；目标文件已存在时跳过。"""
    ogg_path = mp3_path.with_suffix(".ogg")

    if ogg_path.exists():
        print(f"跳过（目标已存在）：{ogg_path}")
        return True

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp3_path),
        "-c:a",
        "libopus",
        "-b:a",
        "64k",
        str(ogg_path),
    ]

    print(f"转换：{mp3_path} -> {ogg_path}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"失败：{mp3_path}\n{result.stderr}", file=sys.stderr)
        return False

    return True


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("错误：未找到 ffmpeg，请先安装并将其加入 PATH。", file=sys.stderr)
        return 1

    if not VOICE_DIR.is_dir():
        print(f"错误：目录不存在：{VOICE_DIR}", file=sys.stderr)
        return 1

    mp3_files = list(VOICE_DIR.rglob("*.mp3")) + list(VOICE_DIR.rglob("*.MP3"))
    if not mp3_files:
        print(f"未找到 MP3 文件：{VOICE_DIR}")
        return 0

    success = sum(convert_one(path) for path in sorted(set(mp3_files)))
    print(f"完成：成功或跳过 {success}/{len(set(mp3_files))} 个文件")
    return 0 if success == len(set(mp3_files)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
