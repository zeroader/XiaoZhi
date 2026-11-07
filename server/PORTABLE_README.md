# 小智感知服务器 Windows 便携版

适用于 Windows 10/11 64 位电脑。目标电脑无需安装 Python、pip 或开发工具。
模型在本地运行，已附带模型时启动服务无需下载模型。

## 使用方法

1. 将 ZIP **完整解压**到一个文件夹（支持中文路径），保留以下结构：

   ```text
   XiaozhiServer/
     XiaozhiServer.exe
     启动服务.bat
     server_config.json
     lamp_config.json
     models/
       face_detection_yunet_2023mar.onnx
       enet_b0_8_best_afew.onnx
       resnet_ppg_nonmixed.onnx
       resnet_ppg_nonmixed.json
       yolov8n_pose.onnx
   ```

2. 用记事本编辑 `server_config.json`，默认监听 `0.0.0.0:8291`。
   如果需要控制台灯，在 `lamp_config.json` 中填写台灯的 `token` 和 `ip`；
   `ip` 可以填 `auto`。未配置台灯时，视觉接口仍可使用。
3. 双击 `启动服务.bat` 或 `XiaozhiServer.exe`。
4. 在本机浏览器访问 `http://127.0.0.1:8291/health` 检查服务。
5. ESP32 和服务器连接同一局域网，设备访问地址改为这台电脑的局域网 IP，
   例如 `http://192.168.1.100:8291`。如果 Windows 提示防火墙访问，允许专用网络。
6. 用 Ctrl+C 或关闭窗口停止服务。配置修改后重启生效。

请复制整个文件夹。配置文件和模型均在 EXE 外部，可以直接修改或替换。
EXE 自带 Python 运行环境和依赖，首次启动会解压运行组件到系统临时目录，稍有等待属正常。
可选模型缺失会关闭对应功能；人脸模型为启动必需项。

## 配置项

| 配置项 | 含义 |
| --- | --- |
| `host` / `port` | 监听地址 / 端口，默认 `0.0.0.0` / `8291` |
| `face_model` | 必需的 YuNet 人脸模型路径 |
| `emotion_model` | 可选的情绪模型路径 |
| `pose_model` | 保留兼容参数；当前坐姿识别使用人脸几何信息 |
| `buffer_size` | 图像缓存帧数，默认 60 |
| `heart_method` / `bp_method` | rPPG 提取方式：`green`、`pos`、`chrom` |
| `bp_model` / `bp_model_config` | 血压 ONNX 模型及其 JSON 参数文件 |
| `bp_window_seconds` | 血压窗口秒数，默认 7 |
| `bp_strict_quality` | 是否启用严格质量检查，默认 `false` |
| `rppg_max_samples` | rPPG 样本缓存数量，默认 1800 |
| `lamp_config` | 台灯配置文件路径 |

相对路径以 `server_config.json` 所在目录为基准，与启动时所在目录无关。
配置优先级为：命令行参数 > JSON 配置 > 内置默认值。
JSON 必须使用双引号，不支持注释；Windows 路径建议用 `/`，如 `models/face.onnx`。
血压模型参数可直接修改 `models/resnet_ppg_nonmixed.json`。

可以在命令行临时覆盖配置：

```bat
XiaozhiServer.exe --port 9000
XiaozhiServer.exe --config D:/my-config/server_config.json
XiaozhiServer.exe --bp-strict-quality
XiaozhiServer.exe --no-bp-strict-quality
XiaozhiServer.exe --help
```

## 从源码重新打包（仅打包电脑需要 Python）

在 Windows 64 位 Python 环境下，在 `server` 目录运行：

```bat
python build_windows.py
```

脚本自动建立独立打包环境、安装依赖、生成 EXE、复制模型及配置模板并创建 ZIP。
输出目录：`server/dist/XiaozhiServer/`。
压缩包：`server/dist/XiaozhiServer-Windows-x64.zip`。
每次重新打包会更新输出目录中的同名配置，请先保存手动修改过的发布配置。

默认使用配置模板。需要沿用本机实际配置时使用：

```bat
python build_windows.py --include-local-config
```

该选项会把本机配置复制到发布目录，其中可能包含台灯 token，请仅向预期接收者分发。
已有打包依赖时，可加 `--skip-install` 跳过联网安装。
源码直接运行仍可用 `python main.py`，默认模型目录为项目根目录的 `models/`。
若在源码模式使用此发布配置模板，请把模型路径改为 `../models/…`。

打包使用 PyInstaller 的单文件模式；外部数据路径按
[PyInstaller 运行时文档](https://www.pyinstaller.org/en/stable/runtime-information.html)
通过 EXE 位置定位。
