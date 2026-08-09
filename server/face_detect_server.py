"""
ESP32 在线人脸检测服务器（旧版入口，向后兼容）

本文件已重构为 server/main.py（生物特征感知服务器）。
此处保留旧文件名作为兼容入口，行为与重构前一致：
  - 旧协议: {"type":"face","images":[...]} -> {"detections":[...]}
  - 新协议: {"frame_id","task","image"} 也同时支持

启动方式（与旧版完全相同）:
  1. 安装依赖: pip install -r server_requirements.txt
  2. 下载模型: python face_detect_server.py --download-model
  3. 启动服务: python face_detect_server.py --host 0.0.0.0 --port 8291
"""

import sys
from pathlib import Path

# 确保当前目录在 sys.path 中，以便导入同目录的 main 模块
_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from main import main  # noqa: E402


if __name__ == "__main__":
    print("[Compat] face_detect_server.py is now an alias of server/main.py "
          "(ESP32 Bio-Perception Server)")
    main()
