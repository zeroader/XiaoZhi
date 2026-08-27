#ifndef LCD_DISPLAY_H
#define LCD_DISPLAY_H

#include "lvgl_display.h"
#include "gif/lvgl_gif.h"

#include <esp_lcd_panel_io.h>
#include <esp_lcd_panel_ops.h>
#include <font_emoji.h>

#include <atomic>
#include <memory>
#include <deque>
#include <string>

#define PREVIEW_IMAGE_DURATION_MS 30000


class LcdDisplay : public LvglDisplay {
protected:
    esp_lcd_panel_io_handle_t panel_io_ = nullptr;
    esp_lcd_panel_handle_t panel_ = nullptr;
    
    lv_draw_buf_t draw_buf_;
    lv_obj_t* status_bar_ = nullptr;
    lv_obj_t* content_ = nullptr;
    lv_obj_t* container_ = nullptr;
    lv_obj_t* side_bar_ = nullptr;
    lv_obj_t* preview_image_ = nullptr;
    lv_obj_t* emoji_label_ = nullptr;
    lv_obj_t* emoji_image_ = nullptr;
    std::unique_ptr<LvglGif> gif_controller_ = nullptr;
    lv_obj_t* emoji_box_ = nullptr;
    lv_obj_t* chat_message_label_ = nullptr;
    lv_obj_t* vision_dashboard_ = nullptr;
    lv_obj_t* vision_background_view_ = nullptr;
    lv_obj_t* vision_heart_label_ = nullptr;
    lv_obj_t* vision_pressure_label_ = nullptr;
    lv_obj_t* vision_posture_label_ = nullptr;
    lv_obj_t* vision_emotion_label_ = nullptr;
    lv_obj_t* vision_heart_icon_ = nullptr;
    lv_obj_t* vision_pressure_icon_ = nullptr;
    lv_obj_t* vision_posture_icon_ = nullptr;
    lv_obj_t* vision_emotion_image_ = nullptr;
    lv_obj_t* vision_posture_image_ = nullptr;
    lv_obj_t* vision_posture_cartoon_ = nullptr;
    lv_obj_t* vision_heart_panel_ = nullptr;
    lv_obj_t* vision_pressure_panel_ = nullptr;
    lv_obj_t* vision_posture_panel_ = nullptr;
    lv_obj_t* vision_emotion_panel_ = nullptr;
    bool vision_hr_alert_active_ = false;
    bool vision_bp_alert_active_ = false;
    bool vision_posture_alert_active_ = false;
    esp_timer_handle_t preview_timer_ = nullptr;
    std::unique_ptr<LvglImage> preview_image_cached_ = nullptr;
    // LVGL/SPI may still reference a previous image after lv_image_set_src().
    // Retain a short history so its PSRAM buffer is not freed during DMA.
    std::deque<std::unique_ptr<LvglImage>> vision_frame_history_;
    std::deque<std::unique_ptr<LvglImage>> vision_asset_history_;
    std::unique_ptr<LvglImage> vision_background_image_ = nullptr;
    std::unique_ptr<LvglImage> vision_posture_asset_image_ = nullptr;
    std::unique_ptr<LvglImage> vision_emotion_asset_image_ = nullptr;
    std::string vision_posture_asset_name_;
    std::string vision_emotion_asset_name_;

    void InitializeLcdThemes();
    void SetupUI();
    virtual bool Lock(int timeout_ms = 0) override;
    virtual void Unlock() override;

protected:
    // 添加protected构造函数
    LcdDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel, int width, int height);
    
public:
    ~LcdDisplay();
    virtual void SetEmotion(const char* emotion) override;
    virtual void SetChatMessage(const char* role, const char* content) override; 
    virtual void SetPreviewImage(std::unique_ptr<LvglImage> image) override;
    virtual void SetVisionPreviewImage(std::unique_ptr<LvglImage> image,
                                       const DetectionResult& result) override;

    // Add theme switching function
    virtual void SetTheme(Theme* theme) override;
};

// SPI LCD显示器
class SpiLcdDisplay : public LcdDisplay {
public:
    SpiLcdDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel,
                  int width, int height, int offset_x, int offset_y,
                  bool mirror_x, bool mirror_y, bool swap_xy);
};

// RGB LCD显示器
class RgbLcdDisplay : public LcdDisplay {
public:
    RgbLcdDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel,
                  int width, int height, int offset_x, int offset_y,
                  bool mirror_x, bool mirror_y, bool swap_xy);
};

// MIPI LCD显示器
class MipiLcdDisplay : public LcdDisplay {
public:
    MipiLcdDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel,
                   int width, int height, int offset_x, int offset_y,
                   bool mirror_x, bool mirror_y, bool swap_xy);
};

#endif // LCD_DISPLAY_H
