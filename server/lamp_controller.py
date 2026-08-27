"""
米家台灯控制（MIoT 协议）—— ESP32 通过 HTTP 调用本模块，由服务器转发到台灯。

台灯型号: xiaomi.light.lamp31 (米家台灯2)
通信协议: MIoT over miIO (UDP 54321, AES 加密, 需要 token)
属性 (siid=2 灯光服务):
  piid=1  电源开关  bool
  piid=2  亮度      1~100
  piid=3  色温      2700~6500 (K)

配置: 与 main.py 同目录的 lamp_config.json（已 gitignore，不会提交 token）
- token: 必填（台灯 token，固定不变）
- ip:    可选。留空或填 "auto" 时，服务器会自动扫描局域网找到台灯；
          填具体 IP 时优先用该 IP（更快），连不上时也会自动回退到扫描。
"""

import concurrent.futures
import json
import logging
import socket
from pathlib import Path

# 屏蔽 python-miio 构造 MiotDevice 时的重复告警（扫描时会刷屏）
logging.getLogger("miio").setLevel(logging.ERROR)

try:
    from miio import MiotDevice
except ImportError:  # python-miio 未安装
    MiotDevice = None

# MIoT 属性 id（xiaomi.light.lamp31）
SIID_LIGHT = 2
PIID_POWER = 1
PIID_BRIGHTNESS = 2
PIID_COLOR_TEMP = 3

CONFIG_PATH = Path(__file__).resolve().parent / "lamp_config.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            "lamp_config.json 不存在：请复制 lamp_config.example.json 为 "
            "lamp_config.json 并填入台灯的 token（ip 可选，留空会自动扫描）"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("token"):
        raise RuntimeError("lamp_config.json 缺少 token")
    return cfg


def _local_ip() -> str:
    """返回本机在局域网中的 IP（如 192.168.0.104）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "192.168.0.1"
    finally:
        s.close()


def _probe(ip: str, token: str):
    """尝试用 token 跟某个 IP 握手，成功则返回该 IP（说明就是我们的台灯）。"""
    try:
        dev = MiotDevice(ip=ip, token=token, timeout=2)
        dev.info()
        return ip
    except Exception:
        return None


def discover_lamp_ip(token: str, max_workers: int = 64):
    """
    在局域网内自动扫描，找出台灯的 IP。

    按「距离本机 IP 由近到远」的顺序探测（台灯和服务器通常拿到相邻的
    DHCP 地址），通常 1 秒内就能找到；找不到返回 None。
    """
    local = _local_ip()
    net_prefix = ".".join(local.split(".")[:3])
    try:
        local_octet = int(local.split(".")[3])
    except (ValueError, IndexError):
        local_octet = 1

    octets = sorted(range(2, 255), key=lambda x: abs(x - local_octet))
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = [ex.submit(_probe, f"{net_prefix}.{o}", token) for o in octets]
    try:
        for fut in concurrent.futures.as_completed(futures):
            ip = fut.result()
            if ip:
                return ip
        return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


class MijiaLampController:
    """封装 MiotDevice，只暴露台灯需要的少量操作。"""

    def __init__(self, config: dict = None):
        if MiotDevice is None:
            raise RuntimeError("python-miio 未安装：请执行 pip install python-miio")
        self._cfg = config or _load_config()
        self._token = self._cfg["token"]
        self._device = None
        self._ip = None
        self._connect()

    def _connect(self):
        """解析台灯 IP：优先用配置里的 IP，否则自动扫描。"""
        cfg_ip = (self._cfg.get("ip") or "").strip()
        if cfg_ip and cfg_ip.lower() != "auto":
            try:
                dev = MiotDevice(ip=cfg_ip, token=self._token, timeout=2)
                dev.info()  # 验证连通
                self._device, self._ip = dev, cfg_ip
                return
            except Exception:
                pass  # 配置的 IP 失效，回退到自动扫描

        found = discover_lamp_ip(self._token)
        if not found:
            raise RuntimeError(
                "无法发现台灯：请确认台灯与服务器在同一局域网，"
                "或在 lamp_config.json 里手动填 ip"
            )
        self._device = MiotDevice(ip=found, token=self._token, timeout=3)
        self._ip = found

    def _get(self, piid):
        return self._device.get_property_by(SIID_LIGHT, piid)[0]["value"]

    def get_status(self) -> dict:
        return {
            "power": bool(self._get(PIID_POWER)),
            "brightness": int(self._get(PIID_BRIGHTNESS)),
            "color_temp": int(self._get(PIID_COLOR_TEMP)),
        }

    def set_power(self, on) -> dict:
        on = bool(on)
        self._device.set_property_by(SIID_LIGHT, PIID_POWER, on)
        return {"power": on}

    def set_brightness(self, value) -> dict:
        value = max(1, min(100, int(value)))
        self._device.set_property_by(SIID_LIGHT, PIID_BRIGHTNESS, value)
        return {"brightness": value}

    def set_color_temp(self, value) -> dict:
        value = max(2700, min(6500, int(value)))
        self._device.set_property_by(SIID_LIGHT, PIID_COLOR_TEMP, value)
        return {"color_temp": value}
