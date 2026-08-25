/*
 * 米家台灯 MCP 工具（MIoT over LAN，Server 中间层）。
 *
 * 流程：ESP32 通过 HTTP 调用家庭服务器（Python Flask, 端口 8291）的 /lamp 接口，
 * 服务器再用 python-miio 通过 miIO 协议控制台灯。台灯的 token 只存在于服务器本地。
 */

#include "miot_lamp_controller.h"

#include <string>
#include <stdexcept>

#include <esp_log.h>

#include "board.h"
#include "settings.h"
#include "mcp_server.h"

#define TAG "MiotLamp"

namespace {

// 台灯服务器默认端口（与视觉服务器同一个 Flask 进程）
constexpr int kLampServerTimeoutSec = 5;

// 归一化服务器地址：裸 "ip:port" 自动补 http://，去掉末尾斜杠
std::string NormalizeServerUrl(const std::string& raw) {
    std::string url = raw;
    if (url.find("://") == std::string::npos) {
        url = "http://" + url;
    }
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    return url;
}

// 从 NVS 读取已配置的台灯服务器地址
std::string LoadServerUrl() {
    Settings settings("lamp");
    return settings.GetString("server_url");
}

// 向台灯服务器 POST 一个 JSON 请求体，返回响应原文（JSON 字符串）。
// 服务器统一返回 HTTP 200，成功/失败用 body 里的 ok 字段区分。
std::string LampHttpPost(const std::string& base_url, const std::string& json_body) {
    auto network = Board::GetInstance().GetNetwork();
    if (network == nullptr) {
        throw std::runtime_error("network not available");
    }

    std::string url = base_url + "/lamp";
    auto http = network->CreateHttp(kLampServerTimeoutSec);
    if (http == nullptr) {
        throw std::runtime_error("failed to create HTTP client");
    }

    http->SetHeader("Content-Type", "application/json");
    if (!http->Open("POST", url)) {
        http->Close();
        throw std::runtime_error("cannot connect to lamp server: " + url);
    }

    http->Write(json_body.c_str(), json_body.size());
    http->Write("", 0);

    int status = http->GetStatusCode();
    std::string response = http->ReadAll();
    http->Close();

    if (status != 200) {
        throw std::runtime_error("lamp server HTTP " + std::to_string(status) + ": " + response);
    }
    return response;
}

// 校验服务器已配置，否则抛错提示先调用 self.lamp.configure
std::string RequireServerUrl() {
    std::string url = LoadServerUrl();
    if (url.empty()) {
        throw std::runtime_error(
            "lamp server not configured yet, call self.lamp.configure first");
    }
    return url;
}

}  // namespace

void RegisterMiotLampMcpTools() {
    auto& mcp = McpServer::GetInstance();

    mcp.AddTool("self.lamp.configure",
        "Configure the home server that relays commands to the Mijia desk lamp (米家台灯2) over the local network.\n"
        "The lamp server is the SAME Flask process as the vision/detection server, listening on port 8291.\n"
        "Args:\n"
        "  `url` (required): base URL of the lamp server. If the user only gives an IP address "
        "(e.g. '192.168.0.100' or '192.168.2.1'), you MUST expand it to 'http://<IP>:8291'.\n"
        "This setting is saved to flash (NVS) and restored automatically after reboot.",
        PropertyList({
            Property("url", kPropertyTypeString)
        }),
        [](const PropertyList& p) -> ReturnValue {
            std::string url = NormalizeServerUrl(p["url"].value<std::string>());
            Settings settings("lamp", true);
            settings.SetString("server_url", url);
            ESP_LOGI(TAG, "Lamp server URL set to: %s", url.c_str());
            return true;
        });

    mcp.AddTool("self.lamp.get_status",
        "Get the current status of the Mijia desk lamp (米家台灯2): power on/off, brightness (1-100), "
        "color temperature (2700-6500K).",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            std::string url = RequireServerUrl();
            return LampHttpPost(url, "{\"action\":\"get_status\"}");
        });

    mcp.AddTool("self.lamp.set_power",
        "Turn the Mijia desk lamp (米家台灯2) on or off over the local network.\n"
        "Args:\n"
        "  `power` (required): true to turn on, false to turn off.",
        PropertyList({
            Property("power", kPropertyTypeBoolean)
        }),
        [](const PropertyList& p) -> ReturnValue {
            std::string url = RequireServerUrl();
            bool on = p["power"].value<bool>();
            std::string body = "{\"action\":\"set_power\",\"value\":";
            body += on ? "true" : "false";
            body += "}";
            return LampHttpPost(url, body);
        });

    mcp.AddTool("self.lamp.set_brightness",
        "Set the brightness of the Mijia desk lamp (米家台灯2), range 1-100.",
        PropertyList({
            Property("brightness", kPropertyTypeInteger, 1, 100)
        }),
        [](const PropertyList& p) -> ReturnValue {
            std::string url = RequireServerUrl();
            std::string body = "{\"action\":\"set_brightness\",\"value\":"
                + std::to_string(p["brightness"].value<int>()) + "}";
            return LampHttpPost(url, body);
        });

    mcp.AddTool("self.lamp.set_color_temp",
        "Set the color temperature of the Mijia desk lamp (米家台灯2) in Kelvin, range 2700-6500.",
        PropertyList({
            Property("color_temp", kPropertyTypeInteger, 2700, 6500)
        }),
        [](const PropertyList& p) -> ReturnValue {
            std::string url = RequireServerUrl();
            std::string body = "{\"action\":\"set_color_temp\",\"value\":"
                + std::to_string(p["color_temp"].value<int>()) + "}";
            return LampHttpPost(url, body);
        });
}
