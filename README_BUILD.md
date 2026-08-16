# ESP32 生物特征感知设备 — 构建与运行指南

> 本文档面向协作者，说明如何从零开始编译、烧录固件并部署感知服务器。
> 基础代码来自 XiaoZhi（ESP32 语音助手），本仓库在其上增加了**视觉流水线（人脸 / 情绪 / 坐姿 / 心率）**与配套 MCP 工具。

---

## 1. 项目结构

```
项目根目录/
├── main/                        # ESP32-S3 固件
│   ├── vision/                  # 视觉流水线（Detector 接口 + Online/Remote/Face 检测器 + LCD 叠加）
│   │   ├── vision_pipeline.cc   # 流水线：采集→检测→调度(face_emotion/posture/heart_rate)→显示
│   │   ├── online_detector.cc   # 在线检测器：JPEG→base64→HTTP POST 到感知服务器
│   │   ├── vision_display.cc    # LCD 画框 + 情绪/坐姿/心率状态文字叠加
│   │   └── detector.h           # 检测结果数据结构（含任务类型常量）
│   └── ...                      # XiaoZhi 原有组件（音频、协议、MCP Server 等）
├── server/                      # 生物特征感知服务器（Python/Flask + ONNX 推理）
│   ├── main.py                  # 入口：接收图片、按 task 分发、缓存帧、返回结果
│   ├── protocol.py              # 通信协议（新旧协议判定、请求/响应解析）
│   ├── state.py                 # ServerState：帧缓存 + 各任务最新结果
│   └── detectors/               # face/emotion/pose/heart_rate 四个检测器
├── models/                      # ONNX 模型（不提交到 git，用 --download-model 下载）
│   ├── face_detection_yunet_2023mar.onnx   # 人脸检测（YuNet）
│   ├── enet_b0_8_best_afew.onnx            # 情绪识别（AffectNet 8 类）
│   └── yolov8n_pose.onnx                   # 人体姿态（COCO 17 关键点）
└── partitions/v2/               # 自定义分区表（16MB 设备使用）
```

---

## 2. 硬件要求

- **主控**：ESP32-S3 开发板，**16MB Flash + 8MB PSRAM**（本项目使用 **正点原子 DNESP32S3**）
- **摄像头**：OV2640（QVGA 320x240）
- **屏幕**：LCD（本项目为 320x240，如用其它板子需在 menuconfig 改 Board Type）
- **连接**：USB 数据线（**UART 串口烧录**）

---

## 3. 环境要求

| 组件 | 版本 |
|---|---|
| ESP-IDF | **5.4.4** |
| Python | 3.10+ |
| 目标芯片 | ESP32-S3 (`idf.py set-target esp32s3`) |

---

## 4. 固件编译与烧录

> 推荐使用 **VSCode + Espressif IDF 插件**（方式一）；熟悉命令行的话也可用方式二。

### 4.1 方式一：VSCode ESP-IDF 插件（推荐）

**① 安装插件并配置 IDF 环境**

1. VSCode 扩展市场搜索并安装 **Espressif IDF**（作者 Espressif Systems）
2. 首次使用：`Ctrl+Shift+P` → 输入 `ESP-IDF: Configure ESP-IDF Extension`，
   选择你已安装的 ESP-IDF 路径（本项目为 **v5.4.4**），插件会自动识别工具链/Python 环境
3. 若插件已装好但项目未识别，打开项目根目录后执行 `ESP-IDF: Add vscode configuration folder` 生成 `.vscode/`

**② 底部状态栏操作（插件最常用入口）**

安装并识别项目后，VSCode **底部状态栏**会出现一排 ESP-IDF 控件：

| 状态栏控件 | 作用 | 本项目选择 |
|---|---|---|
| 芯片型号（`ESP-IDF: Set Espressif Device Target`） | 选择目标芯片 | **esp32s3** |
| 串口（`ESP-IDF: Select Port to Use`） | 选择烧录/监视串口 | 你的 USB 串口（如 COM5） |
| **`Build`** 按钮（齿轮/扳手图标） | 编译固件 | 点击即编译 |
| **`Flash`** 按钮（闪电图标） | 烧录到设备（UART） | 点击烧录 |
| **`Monitor`** 按钮（终端图标） | 打开串口监视器 | 查看日志 |

操作顺序：**选芯片 → 选串口 → Build → Flash → Monitor**。

**③ menuconfig（图形化配置界面）**

`Ctrl+Shift+P` → `ESP-IDF: Open Configuration Editor`（或点状态栏 `设置` 图标），
在配置界面中必须修改：

```
Xiaozhi Assistant
└── Board Type
    └── 正点原子DNESP32S3开发板        ← 必须选中（对应 CONFIG_BOARD_TYPE_ATK_DNESP32S3）
```

**④ 其它常用命令（命令面板 `Ctrl+Shift+P`）**

- `ESP-IDF: Build your project`
- `ESP-IDF: Flash your project`
- `ESP-IDF: Monitor Device`
- `ESP-IDF: Set Espressif Device Target`（选 esp32s3）
- `ESP-IDF: Select Port to Use`（选串口）

> 具体用法以插件官方文档为准（`ESP-IDF: ESP-IDF Documentation`）。

### 4.2 方式二：命令行

### 4.2.1 初始化 IDF 环境（Windows）

```powershell
# 以管理员打开 PowerShell，执行 IDF 的 export 脚本（路径按你的实际安装位置）
& "D:\Program Files\esp\v5.4.4\esp-idf\export.ps1"
```

Linux / macOS 使用 `source $IDF_PATH/export.sh`。

### 4.2.2 配置 target 与 menuconfig

```bash
cd <项目根目录>

# 1) 选择芯片（重要：本项目为 ESP32-S3）
idf.py set-target esp32s3

# 2) 打开配置菜单
idf.py menuconfig
```

**menuconfig 必须修改的配置：**

```
Xiaozhi Assistant
└── Board Type
    └── 正点原子DNESP32S3开发板        ← 必须选中（对应 CONFIG_BOARD_TYPE_ATK_DNESP32S3）
```

其余关键配置（`sdkconfig` 中已默认配好，一般无需改动）：

| 配置项 | 值 |
|---|---|
| Flash 大小 | 16MB |
| 分区表 | Custom → `partitions/v2/16m.csv` |
| PSRAM | 使能（Octal，8MB） |

> 若使用其它板子，请在此处改选对应的 Board Type（如 `bread-compact-wifi-s3cam` 等），
> 并确认屏幕/摄像头引脚与你的硬件一致。

### 4.2.3 编译

```bash
idf.py build
```

### 4.2.4 UART 烧录 + 串口监视

```bash
# <PORT> 换成实际串口（Windows 例如 COM5，Linux 例如 /dev/ttyUSB0）
idf.py -p <PORT> flash monitor
```

烧录成功后按 `Ctrl+]` 退出串口监视。设备会自动重启并连接网络（WiFi 配置见 XiaoZhi 原生文档）。

---

## 5. 部署感知服务器（生物特征感知）

服务器负责：接收 ESP32 上传的图片 → 按任务推理 → 缓存最近帧 → 返回结果。
**所有任务频率由 ESP32 控制，服务器不做定时任务。**

### 5.1 安装依赖

```bash
cd server
pip install -r server_requirements.txt
```

依赖：`flask`、`Pillow`、`numpy`、`opencv-python`、`onnxruntime`、`scipy`

### 5.2 下载模型

```bash
python main.py --download-model
```

自动下载 3 个模型到项目根目录 `models/`（约几十 MB）。模型**未提交到 git**，协作者必须执行此步或自行放置模型文件。

### 5.3 启动服务

```bash
python main.py --host 0.0.0.0 --port 8291
```

启动后应看到：

```
  ESP32 Bio-Perception Server
  Listen: http://0.0.0.0:8291
  Tasks : face_emotion / posture / heart_rate
```

> 可选参数：`--face-model` / `--emotion-model` / `--pose-model` / `--buffer-size` / `--threshold`

---

## 6. 设备与服务器对接

设备与服务器运行在同一局域网。开机联网后，通过语音让设备端大模型调用以下 MCP 工具完成配置与启动：

| 步骤 | 工具 | 说明 |
|---|---|---|
| 1. 配置服务器地址 | `self.vision.online_detector.configure` | url=`http://<服务器IP>:8291/detect` |
| 2. 开始持续检测 | `self.vision.start_continuous(task="auto")` | 每帧情绪 + 5s 坐姿 + 10s 心率 |
| 3. 查询数据 | `get_user_emotion` / `get_posture` / `get_heart_rate` | 读缓存，不触发拍照 |

协议要点（详见 `server/protocol.py`）：

```json
// 请求
{ "frame_id": 1, "task": "face_emotion",
  "image": { "data": "<base64 jpeg>", "width": 320, "height": 240, "format": "jpeg" } }

// 响应
{ "frame_id": 1, "task": "face_emotion",
  "result": { "face": {...}, "emotion": {"label": "happy", "confidence": 0.9} },
  "performance": { "decode_ms": 3.0, "infer_ms": 15.0, "total_ms": 20.0 } }
```

---

## 7. 注意事项 / 坑

- **`models/` 和 `build/` 未提交到 git**：协作者 clone 后需执行 `python main.py --download-model` 下载模型，再 `idf.py build`。
- **Board Type 必须选对**：不同板子的摄像头/屏幕/引脚配置差异很大，选错会导致外设不工作。
- **串口烧录**：如一直烧录失败，按住板子 BOOT 键再上电/点击烧录，进入下载模式。
- **大日志不要用 `%lld` / `%zu`**：本项目固件启用 nano printf（`CONFIG_NEWLIB_NANO_FORMAT`），不识别 `%lld`，会打印成字面量并错位后续参数，请用 `%d` + `(int)` 强转。
- **连续检测期间不要调用 `detect_once`**：摄像头被预览线程独占，会并发取帧失败；请改用缓存读取工具（`get_*`）。
- 视觉工具命名约定：所有大模型工具以 `self.vision.*` 开头。
