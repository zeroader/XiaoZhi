#ifndef MIOT_LAMP_CONTROLLER_H
#define MIOT_LAMP_CONTROLLER_H

// 注册「米家台灯」控制相关的 MCP 工具。
//
// 架构（Server 中间层）：
//   ESP32 --HTTP--> 家庭服务器(Python Flask, 端口 8291) --miIO(UDP 54321)--> 台灯
//
// 台灯型号：xiaomi.light.lamp31 (米家台灯2)，MIoT 协议，token 只存放在
// 服务器本地配置 server/lamp_config.json（已 gitignore，不进入 git）。
void RegisterMiotLampMcpTools();

#endif // MIOT_LAMP_CONTROLLER_H
