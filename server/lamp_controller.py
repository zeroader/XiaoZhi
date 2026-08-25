"""
米家台灯控制（MIoT 协议）—— ESP32 通过 HTTP 调用本模块，由服务器转发到台灯。

台灯型号: xiaomi.light.lamp31 (米家台灯2)
通信协议: MIoT over miIO (UDP 54321, AES 加密, 需要 token)
属性 (siid=2 灯光服务):
  piid=1  电源开关  bool
  piid=2  亮度      1~100
  piid=3  色温      2700~6500 (K)

配置: 与 main.py 同目录的 lamp_config.json（已 gitignore，不会提交 token）
格式见 lamp_config.example.json。
"""

import json
from pathlib import Path

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
            "lamp_config.json 并填入台灯的 ip / token"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("ip") or not cfg.get("token"):
        raise RuntimeError("lamp_config.json 缺少 ip 或 token")
    return cfg


class MijiaLampController:
    """封装 MiotDevice，只暴露台灯需要的少量操作。"""

    def __init__(self, config: dict = None):
        if MiotDevice is None:
            raise RuntimeError("python-miio 未安装：请执行 pip install python-miio")
        self._cfg = config or _load_config()
        self._device = MiotDevice(ip=self._cfg["ip"], token=self._cfg["token"])

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
        # 直接传原生 bool，miIO 层会序列化为 JSON true/false
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
