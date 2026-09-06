#include "lcd_display.h"
#include "gif/lvgl_gif.h"
#include "settings.h"
#include "lvgl_theme.h"
#include "assets/lang_config.h"

#include <vector>
#include <algorithm>
#include <font_awesome.h>
#include <esp_log.h>
#include <esp_err.h>
#include <esp_lvgl_port.h>
#include <esp_psram.h>
#include <esp_heap_caps.h>
#include <cstring>

#include "board.h"
#include "assets.h"
#include "vision/detector.h"
#include "vision/vision_viewport.h"

#define TAG "LcdDisplay"

LV_FONT_DECLARE(BUILTIN_TEXT_FONT);
LV_FONT_DECLARE(BUILTIN_ICON_FONT);
LV_FONT_DECLARE(font_awesome_30_4);

static void StartVisionAlertAnimation(lv_obj_t* object) {
    lv_obj_set_style_bg_opa(object, LV_OPA_COVER, 0);
    lv_anim_t animation;
    lv_anim_init(&animation);
    lv_anim_set_var(&animation, object);
    lv_anim_set_values(&animation, LV_OPA_30, LV_OPA_COVER);
    lv_anim_set_duration(&animation, 650);
    lv_anim_set_playback_duration(&animation, 650);
    lv_anim_set_repeat_count(&animation, LV_ANIM_REPEAT_INFINITE);
    lv_anim_set_exec_cb(&animation, [](void* var, int32_t value) {
        lv_obj_set_style_opa(static_cast<lv_obj_t*>(var), static_cast<lv_opa_t>(value), 0);
    });
    lv_anim_start(&animation);
}

static const char* EmotionLabelZh(const std::string& label) {
    if (label == "happy") return "开心";
    if (label == "sad") return "悲伤";
    if (label == "angry") return "生气";
    if (label == "fear") return "害怕";
    if (label == "surprise") return "惊讶";
    if (label == "disgust") return "厌恶";
    if (label == "contempt") return "轻蔑";
    if (label == "neutral") return "平静";
    return label.empty() ? "等待检测" : label.c_str();
}

static const char* EmotionEmojiKey(const std::string& label) {
    if (label == "happy") return "happy";
    if (label == "sad") return "sad";
    if (label == "angry") return "angry";
    if (label == "surprise" || label == "fear") return "surprised";
    if (label == "disgust") return "confused";
    return "neutral";
}

static const char* EmotionAssetKey(const std::string& label) {
    if (label == "angry" || label == "contempt" || label == "disgust" ||
        label == "fear" || label == "happy" || label == "neutral" ||
        label == "sad" || label == "surprise") return label.c_str();
    return "neutral";
}

static void StopVisionAlertAnimation(lv_obj_t* object) {
    lv_anim_delete(object, nullptr);
    // Dashboard cards are transparent over the supplied artwork when normal.
    lv_obj_set_style_opa(object, LV_OPA_COVER, 0);
    lv_obj_set_style_bg_opa(object, LV_OPA_TRANSP, 0);
}

static const lv_point_precise_t kVisionGoodPosturePoints[] = {
    {26, 4}, {22, 8}, {26, 12}, {30, 8}, {26, 4},
    {26, 12}, {26, 29}, {14, 48}, {26, 29}, {38, 48},
    {26, 29}, {12, 22}, {40, 22}
};

static const lv_point_precise_t kVisionBadPosturePoints[] = {
    {32, 4}, {28, 8}, {32, 12}, {36, 8}, {32, 4},
    {32, 12}, {27, 29}, {13, 43}, {27, 29}, {37, 49},
    {27, 29}, {12, 20}, {41, 25}
};

static void SetVisionPostureGraphic(lv_obj_t* object, bool bad) {
    const auto* points = bad ? kVisionBadPosturePoints : kVisionGoodPosturePoints;
    const uint32_t point_count = bad
        ? sizeof(kVisionBadPosturePoints) / sizeof(kVisionBadPosturePoints[0])
        : sizeof(kVisionGoodPosturePoints) / sizeof(kVisionGoodPosturePoints[0]);
    lv_line_set_points(object, points, point_count);
    lv_obj_set_style_line_color(object, bad ? lv_color_hex(0xE05663) : lv_color_hex(0x27A66A), 0);
    lv_obj_set_style_line_width(object, 3, 0);
    lv_obj_set_style_line_rounded(object, true, 0);
}

static void SetVisionEmojiImage(lv_obj_t* object, const LvglImage* image) {
    if (image == nullptr || image->image_dsc() == nullptr) {
        lv_obj_add_flag(object, LV_OBJ_FLAG_HIDDEN);
        return;
    }

    const auto* dsc = image->image_dsc();
    lv_image_set_src(object, dsc);
    // Fit either the 32px or 64px built-in Twemoji into the narrow right tile.
    const lv_coord_t max_size = 52;
    if (dsc->header.w > 0 && dsc->header.h > 0) {
        const lv_coord_t scale_w = max_size * 256 / dsc->header.w;
        const lv_coord_t scale_h = max_size * 256 / dsc->header.h;
        lv_image_set_scale(object, scale_w < scale_h ? scale_w : scale_h);
    }
    lv_obj_set_size(object, max_size, max_size);
    lv_obj_align(object, LV_ALIGN_TOP_MID, 0, 5);
    lv_obj_remove_flag(object, LV_OBJ_FLAG_HIDDEN);
}

std::shared_ptr<LvglImage> LcdDisplay::LoadVisionAsset(const char* name) {
    const auto found = vision_asset_cache_.find(name);
    if (found != vision_asset_cache_.end()) return found->second;
    void* data = nullptr;
    size_t size = 0;
    if (!Assets::GetInstance().GetAssetData(name, data, size)) return nullptr;
    auto image = std::make_shared<LvglRawImage>(data, size);
    vision_asset_cache_.emplace(name, image);
    return image;
}

// Both layout implementations call these helpers while holding the LVGL lock.
void LcdDisplay::PrepareVisionFrame(int bottom_y, bool show_bottom_mask) {
    // Result handling below hides unavailable cards. Do not hide/show valid
    // cards every frame: that invalidates their entire area on every refresh.
    if (vision_bottom_mask_ == nullptr) {
        vision_bottom_mask_ = lv_obj_create(vision_dashboard_);
        lv_obj_remove_style_all(vision_bottom_mask_);
        lv_obj_clear_flag(vision_bottom_mask_, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_style_bg_color(vision_bottom_mask_, lv_color_white(), 0);
        lv_obj_set_style_bg_opa(vision_bottom_mask_, LV_OPA_COVER, 0);
    }
    lv_obj_set_pos(vision_bottom_mask_, 0, bottom_y);
    lv_obj_set_size(vision_bottom_mask_, LV_HOR_RES, LV_VER_RES - bottom_y);
    if (show_bottom_mask) {
        lv_obj_remove_flag(vision_bottom_mask_, LV_OBJ_FLAG_HIDDEN);
    } else {
        // dashboard_5 already contains a fitted light-blue transcript channel
        // and the decorative grass strip. An opaque mask here erased both.
        lv_obj_add_flag(vision_bottom_mask_, LV_OBJ_FLAG_HIDDEN);
    }
}

void LcdDisplay::PublishVisionFrame(std::unique_ptr<LvglImage> image) {
    // Own the new descriptor before publishing it, and retain the previous
    // eight complete frames across updates and ordinary-preview transitions.
    if (preview_image_cached_) {
        vision_frame_history_.push_back(std::move(preview_image_cached_));
    }
    preview_image_cached_ = std::move(image);
    lv_image_set_src(preview_image_, preview_image_cached_->image_dsc());
    while (vision_frame_history_.size() > 8) vision_frame_history_.pop_front();

    // Report frames that actually reached this publishing path. Update the
    // label once per second so the counter itself does not add per-frame UI
    // invalidation or SPI traffic.
    if (vision_fps_label_ != nullptr) {
        const uint64_t now_ms = esp_timer_get_time() / 1000ULL;
        if (vision_fps_window_start_ms_ == 0 ||
            now_ms - vision_fps_window_start_ms_ > 3000) {
            vision_fps_window_start_ms_ = now_ms;
            vision_fps_frame_count_ = 0;
        }
        ++vision_fps_frame_count_;
        const uint64_t elapsed_ms = now_ms - vision_fps_window_start_ms_;
        if (elapsed_ms >= 1000) {
            char fps_text[20];
            const float fps = vision_fps_frame_count_ * 1000.0f / elapsed_ms;
            snprintf(fps_text, sizeof(fps_text), "FPS %.1f", fps);
            if (std::strcmp(lv_label_get_text(vision_fps_label_), fps_text) != 0) {
                lv_label_set_text(vision_fps_label_, fps_text);
            }
            vision_fps_window_start_ms_ = now_ms;
            vision_fps_frame_count_ = 0;
        }
    }

    auto ensure_index = [](lv_obj_t* object, int32_t index) {
        if (object && lv_obj_get_index(object) != index) lv_obj_move_to_index(object, index);
    };
    ensure_index(vision_background_view_, 0);
    ensure_index(preview_image_, vision_background_view_ ? 1 : 0);
    ensure_index(vision_fps_label_, vision_background_view_ ? 2 : 1);
    const bool direct_caption = chat_message_label_ &&
                                lv_obj_get_parent(chat_message_label_) == vision_dashboard_;
    const int32_t last = lv_obj_get_child_count(vision_dashboard_) - 1;
    ensure_index(vision_bottom_mask_, last - (vision_caption_ ? 1 : 0) - (direct_caption ? 1 : 0));
    ensure_index(vision_caption_, last - (direct_caption ? 1 : 0));
    if (direct_caption) ensure_index(chat_message_label_, last);
}

static void SetLabelTextIfChanged(lv_obj_t* label, const char* text) {
    if (label != nullptr && std::strcmp(lv_label_get_text(label), text) != 0) {
        lv_label_set_text(label, text);
    }
}

static void SetVisionTranscriptX(void* object, int32_t x) {
    lv_obj_set_x(static_cast<lv_obj_t*>(object), static_cast<lv_coord_t>(x));
}

static std::string VisionTranscriptSingleLine(const char* text) {
    std::string line = text != nullptr ? text : "";
    for (char& ch : line) {
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    }
    return line;
}

static void StartVisionTranscriptScroll(lv_obj_t* label, lv_coord_t screen_width,
                                        lv_coord_t caption_y, lv_coord_t caption_height) {
    if (label == nullptr) return;

    lv_anim_delete(label, SetVisionTranscriptX);
    const auto line = VisionTranscriptSingleLine(lv_label_get_text(label));
    SetLabelTextIfChanged(label, line.c_str());
    lv_label_set_long_mode(label, LV_LABEL_LONG_CLIP);
    lv_obj_set_width(label, LV_SIZE_CONTENT);
    lv_obj_set_align(label, LV_ALIGN_DEFAULT);
    lv_obj_set_style_base_dir(label, LV_BASE_DIR_LTR, 0);

    const lv_font_t* font = lv_obj_get_style_text_font(label, 0);
    const lv_coord_t line_height = font != nullptr ? font->line_height : caption_height;
    const int32_t zoom = std::min<int32_t>(256, (caption_height - 2) * 256 / line_height);
    lv_obj_set_height(label, line_height);
    lv_obj_set_style_text_align(label, LV_TEXT_ALIGN_LEFT, 0);
    // The 20 px Puhui font has a 25 px line box. Keep that box intact and
    // scale around its top-left corner, independent of the sentence width.
    lv_obj_set_style_transform_pivot_x(label, 0, 0);
    lv_obj_set_style_transform_pivot_y(label, 0, 0);
    lv_obj_set_style_transform_zoom(label, zoom, 0);
    lv_obj_update_layout(label);

    const lv_coord_t text_width = std::max<lv_coord_t>(1, (lv_obj_get_width(label) * zoom + 255) / 256);
    const lv_coord_t text_y = caption_y + (caption_height - (line_height * zoom + 255) / 256) / 2;
    lv_obj_set_pos(label, screen_width, text_y);
    if (line.empty()) return;

    // Constant-speed marquee: every utterance enters from the right and leaves
    // at the left, including short one-line messages.
    lv_anim_t animation;
    lv_anim_init(&animation);
    lv_anim_set_var(&animation, label);
    lv_anim_set_values(&animation, screen_width, -text_width);
    lv_anim_set_duration(&animation,
                         std::max<uint32_t>(4200, (screen_width + text_width) * 20));
    lv_anim_set_path_cb(&animation, lv_anim_path_linear);
    lv_anim_set_repeat_count(&animation, LV_ANIM_REPEAT_INFINITE);
    lv_anim_set_exec_cb(&animation, SetVisionTranscriptX);
    lv_anim_start(&animation);
}

static bool LoadVisionDashboard(void*& data, size_t& size) {
    const char* names[] = {
        "vision_dashboard_5.png",
        "vision_dashboard_4.png",
        "vision_dashboard_3.png",
        "vision_dashboard_2.png",
        "vision_dashboard_1.png",
        "vision_dashboard.png",
    };
    for (const char* name : names) {
        if (Assets::GetInstance().GetAssetData(name, data, size)) {
            ESP_LOGI(TAG, "Loaded vision dashboard asset: %s (%u bytes)", name, (unsigned)size);
            return true;
        }
    }
    ESP_LOGE(TAG, "No vision dashboard asset found in assets index");
    return false;
}

static void StyleVisionCard(lv_obj_t* object, lv_color_t background, lv_color_t border,
                            lv_coord_t radius = 10) {
    lv_obj_set_style_bg_color(object, background, 0);
    lv_obj_set_style_bg_opa(object, LV_OPA_COVER, 0);
    lv_obj_set_style_border_color(object, border, 0);
    lv_obj_set_style_border_width(object, 2, 0);
    lv_obj_set_style_radius(object, radius, 0);
    lv_obj_set_style_pad_all(object, 2, 0);
    lv_obj_set_style_shadow_width(object, 0, 0);
}

static lv_obj_t* CreateVisionPart(lv_obj_t* parent, lv_coord_t x, lv_coord_t y,
                                  lv_coord_t width, lv_coord_t height,
                                  lv_color_t color, lv_coord_t radius) {
    lv_obj_t* part = lv_obj_create(parent);
    lv_obj_set_size(part, width, height);
    lv_obj_set_pos(part, x, y);
    lv_obj_set_style_bg_color(part, color, 0);
    lv_obj_set_style_bg_opa(part, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(part, 0, 0);
    lv_obj_set_style_radius(part, radius, 0);
    lv_obj_set_style_pad_all(part, 0, 0);
    return part;
}

static void CreateVisionBear(lv_obj_t* parent) {
    lv_obj_t* bear = lv_obj_create(parent);
    lv_obj_set_size(bear, 25, 24);
    lv_obj_set_pos(bear, 2, 0);
    lv_obj_set_style_bg_opa(bear, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(bear, 0, 0);
    lv_obj_set_style_pad_all(bear, 0, 0);
    const lv_color_t brown = lv_color_hex(0xB86B35);
    const lv_color_t face = lv_color_hex(0xFFE1AF);
    CreateVisionPart(bear, 2, 1, 8, 8, brown, LV_RADIUS_CIRCLE);
    CreateVisionPart(bear, 15, 1, 8, 8, brown, LV_RADIUS_CIRCLE);
    CreateVisionPart(bear, 4, 4, 18, 18, brown, LV_RADIUS_CIRCLE);
    CreateVisionPart(bear, 8, 10, 10, 7, face, LV_RADIUS_CIRCLE);
    CreateVisionPart(bear, 8, 8, 3, 3, lv_color_hex(0x3A251C), LV_RADIUS_CIRCLE);
    CreateVisionPart(bear, 15, 8, 3, 3, lv_color_hex(0x3A251C), LV_RADIUS_CIRCLE);
}

static lv_obj_t* CreateVisionPostureCartoon(lv_obj_t* parent) {
    lv_obj_t* art = lv_obj_create(parent);
    lv_obj_set_size(art, 68, 70);
    lv_obj_align(art, LV_ALIGN_TOP_MID, 0, 3);
    lv_obj_set_style_bg_opa(art, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(art, 0, 0);
    lv_obj_set_style_pad_all(art, 0, 0);

    // A small seated character behind the live posture skeleton.
    const lv_color_t skin = lv_color_hex(0xFFD09C);
    const lv_color_t hair = lv_color_hex(0x6C351D);
    const lv_color_t shirt = lv_color_hex(0xFFC34D);
    const lv_color_t chair = lv_color_hex(0x76A9D6);
    CreateVisionPart(art, 5, 30, 5, 30, chair, 3);
    CreateVisionPart(art, 9, 53, 43, 6, chair, 3);
    CreateVisionPart(art, 26, 4, 16, 16, hair, LV_RADIUS_CIRCLE);
    CreateVisionPart(art, 29, 7, 10, 10, skin, LV_RADIUS_CIRCLE);
    CreateVisionPart(art, 25, 20, 18, 21, shirt, 6);
    CreateVisionPart(art, 39, 28, 19, 5, skin, 3);
    CreateVisionPart(art, 28, 40, 24, 7, lv_color_hex(0x315B9A), 3);
    CreateVisionPart(art, 47, 44, 6, 21, skin, 3);
    CreateVisionPart(art, 44, 62, 14, 5, lv_color_hex(0x25466F), 3);
    return art;
}

void LcdDisplay::InitializeLcdThemes() {
    auto text_font = std::make_shared<LvglBuiltInFont>(&BUILTIN_TEXT_FONT);
    auto icon_font = std::make_shared<LvglBuiltInFont>(&BUILTIN_ICON_FONT);
    auto large_icon_font = std::make_shared<LvglBuiltInFont>(&font_awesome_30_4);

    // light theme
    auto light_theme = new LvglTheme("light");
    light_theme->set_background_color(lv_color_hex(0xFFFFFF));          //rgb(255, 255, 255)
    light_theme->set_text_color(lv_color_hex(0x000000));                //rgb(0, 0, 0)
    light_theme->set_chat_background_color(lv_color_hex(0xE0E0E0));     //rgb(224, 224, 224)
    light_theme->set_user_bubble_color(lv_color_hex(0x00FF00));         //rgb(0, 128, 0)
    light_theme->set_assistant_bubble_color(lv_color_hex(0xDDDDDD));    //rgb(221, 221, 221)
    light_theme->set_system_bubble_color(lv_color_hex(0xFFFFFF));       //rgb(255, 255, 255)
    light_theme->set_system_text_color(lv_color_hex(0x000000));         //rgb(0, 0, 0)
    light_theme->set_border_color(lv_color_hex(0x000000));              //rgb(0, 0, 0)
    light_theme->set_low_battery_color(lv_color_hex(0x000000));         //rgb(0, 0, 0)
    light_theme->set_text_font(text_font);
    light_theme->set_icon_font(icon_font);
    light_theme->set_large_icon_font(large_icon_font);

    // dark theme
    auto dark_theme = new LvglTheme("dark");
    dark_theme->set_background_color(lv_color_hex(0x000000));           //rgb(0, 0, 0)
    dark_theme->set_text_color(lv_color_hex(0xFFFFFF));                 //rgb(255, 255, 255)
    dark_theme->set_chat_background_color(lv_color_hex(0x1F1F1F));      //rgb(31, 31, 31)
    dark_theme->set_user_bubble_color(lv_color_hex(0x00FF00));          //rgb(0, 128, 0)
    dark_theme->set_assistant_bubble_color(lv_color_hex(0x222222));     //rgb(34, 34, 34)
    dark_theme->set_system_bubble_color(lv_color_hex(0x000000));        //rgb(0, 0, 0)
    dark_theme->set_system_text_color(lv_color_hex(0xFFFFFF));          //rgb(255, 255, 255)
    dark_theme->set_border_color(lv_color_hex(0xFFFFFF));               //rgb(255, 255, 255)
    dark_theme->set_low_battery_color(lv_color_hex(0xFF0000));          //rgb(255, 0, 0)
    dark_theme->set_text_font(text_font);
    dark_theme->set_icon_font(icon_font);
    dark_theme->set_large_icon_font(large_icon_font);

    auto& theme_manager = LvglThemeManager::GetInstance();
    theme_manager.RegisterTheme("light", light_theme);
    theme_manager.RegisterTheme("dark", dark_theme);
}

LcdDisplay::LcdDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel, int width, int height)
    : panel_io_(panel_io), panel_(panel) {
    width_ = width;
    height_ = height;

    // Initialize LCD themes
    InitializeLcdThemes();

    // Load theme from settings
    Settings settings("display", false);
    std::string theme_name = settings.GetString("theme", "light");
    current_theme_ = LvglThemeManager::GetInstance().GetTheme(theme_name);

    // Create a timer to hide the preview image
    esp_timer_create_args_t preview_timer_args = {
        .callback = [](void* arg) {
            LcdDisplay* display = static_cast<LcdDisplay*>(arg);
            // The LVGL mutex is recursive. Check mode and hide atomically so
            // an already-dispatched photo timeout cannot hide live monitoring.
            DisplayLockGuard lock(display, 1);
            if (!lock) {
                esp_timer_start_once(display->preview_timer_, 100000);
                return;
            }
            if (display->vision_dashboard_ != nullptr && display->preview_image_ != nullptr &&
                lv_obj_get_parent(display->preview_image_) == display->vision_dashboard_) {
                return;
            }
            display->SetPreviewImage(nullptr);
        },
        .arg = this,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "preview_timer",
        .skip_unhandled_events = false,
    };
    esp_timer_create(&preview_timer_args, &preview_timer_);
}

SpiLcdDisplay::SpiLcdDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel,
                           int width, int height, int offset_x, int offset_y, bool mirror_x, bool mirror_y, bool swap_xy)
    : LcdDisplay(panel_io, panel, width, height) {

    // draw white
    std::vector<uint16_t> buffer(width_, 0xFFFF);
    for (int y = 0; y < height_; y++) {
        esp_lcd_panel_draw_bitmap(panel_, 0, y, width_, y + 1, buffer.data());
    }

    // Set the display to on
    ESP_LOGI(TAG, "Turning display on");
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_, true));

    ESP_LOGI(TAG, "Initialize LVGL library");
    lv_init();

#if CONFIG_SPIRAM
    // lv image cache, currently only PNG is supported
    size_t psram_size_mb = esp_psram_get_size() / 1024 / 1024;
    if (psram_size_mb >= 8) {
        lv_image_cache_resize(2 * 1024 * 1024, true);
        ESP_LOGI(TAG, "Use 2MB of PSRAM for image cache");
    } else if (psram_size_mb >= 2) {
        lv_image_cache_resize(512 * 1024, true);
        ESP_LOGI(TAG, "Use 512KB of PSRAM for image cache");
    }
#endif

    ESP_LOGI(TAG, "Initialize LVGL port");
    lvgl_port_cfg_t port_cfg = ESP_LVGL_PORT_INIT_CONFIG();
    port_cfg.task_priority = 1;
#if CONFIG_SOC_CPU_CORES_NUM > 1
    port_cfg.task_affinity = 1;
#endif
    lvgl_port_init(&port_cfg);

    ESP_LOGI(TAG, "Adding LCD display");
#if CONFIG_BOARD_TYPE_ATK_DNESP32S3
    // LCD initializes before camera/audio/vision threads. Prefer one wider DMA
    // band over two narrow bands: both use at most 12.8 KB, but the wider band
    // halves the number of visibly progressive partial-refresh boundaries.
    // Larger/double buffers previously starved the detection thread's stack.
    constexpr size_t kInternalReserve = 48 * 1024;
    const size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const size_t largest_dma = heap_caps_get_largest_free_block(MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    uint32_t dma_lines = 8;
    for (const uint32_t lines : {20U, 16U, 12U, 10U, 8U}) {
        const size_t buffer_bytes = size_t(width_) * lines * sizeof(uint16_t);
        if (largest_dma >= buffer_bytes &&
            internal_free >= buffer_bytes + kInternalReserve) {
            dma_lines = lines;
            break;
        }
    }
    ESP_LOGI(TAG, "LCD DMA budget: internal_free=%u largest_dma=%u reserve=%u lines=%u single",
             (unsigned)internal_free, (unsigned)largest_dma, (unsigned)kInternalReserve,
             (unsigned)dma_lines);
#endif
    lvgl_port_display_cfg_t display_cfg = {
        .io_handle = panel_io_,
        .panel_handle = panel_,
        .control_handle = nullptr,
#if CONFIG_BOARD_TYPE_ATK_DNESP32S3
        .buffer_size = static_cast<uint32_t>(width_ * dma_lines),
        .double_buffer = false,
#else
        .buffer_size = static_cast<uint32_t>(width_ * 40),
        .double_buffer = false,
#endif
        .trans_size = 0,
        .hres = static_cast<uint32_t>(width_),
        .vres = static_cast<uint32_t>(height_),
        .monochrome = false,
        .rotation = {
            .swap_xy = swap_xy,
            .mirror_x = mirror_x,
            .mirror_y = mirror_y,
        },
        .color_format = LV_COLOR_FORMAT_RGB565,
        .flags = {
            .buff_dma = 1,
            .buff_spiram = 0,
            .sw_rotate = 0,
            .swap_bytes = 1,
            .full_refresh = 0,
            .direct_mode = 0,
        },
    };

    DisplayLockGuard display_setup_lock(this);
    if (!display_setup_lock) return;
    display_ = lvgl_port_add_disp(&display_cfg);
#if CONFIG_BOARD_TYPE_ATK_DNESP32S3
    if (display_) {
        ESP_LOGI(TAG, "LCD DMA: single buffer, %u bytes, SPI clock unchanged",
                 (unsigned)(display_cfg.buffer_size * sizeof(uint16_t)));
    }
#endif
    if (display_ == nullptr) {
        ESP_LOGE(TAG, "Failed to add display");
        return;
    }

    if (offset_x != 0 || offset_y != 0) {
        lv_display_set_offset(display_, offset_x, offset_y);
    }

    SetupUI();
}

// RGB LCD实现
RgbLcdDisplay::RgbLcdDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel,
                           int width, int height, int offset_x, int offset_y,
                           bool mirror_x, bool mirror_y, bool swap_xy)
    : LcdDisplay(panel_io, panel, width, height) {

    // draw white
    std::vector<uint16_t> buffer(width_, 0xFFFF);
    for (int y = 0; y < height_; y++) {
        esp_lcd_panel_draw_bitmap(panel_, 0, y, width_, y + 1, buffer.data());
    }

    ESP_LOGI(TAG, "Initialize LVGL library");
    lv_init();

    ESP_LOGI(TAG, "Initialize LVGL port");
    lvgl_port_cfg_t port_cfg = ESP_LVGL_PORT_INIT_CONFIG();
    port_cfg.task_priority = 1;
    port_cfg.timer_period_ms = 50;
    lvgl_port_init(&port_cfg);

    ESP_LOGI(TAG, "Adding LCD display");
    const lvgl_port_display_cfg_t display_cfg = {
        .io_handle = panel_io_,
        .panel_handle = panel_,
        .buffer_size = static_cast<uint32_t>(width_ * 20),
        .double_buffer = true,
        .hres = static_cast<uint32_t>(width_),
        .vres = static_cast<uint32_t>(height_),
        .rotation = {
            .swap_xy = swap_xy,
            .mirror_x = mirror_x,
            .mirror_y = mirror_y,
        },
        .flags = {
            .buff_dma = 1,
            .swap_bytes = 0,
            .full_refresh = 1,
            .direct_mode = 1,
        },
    };

    const lvgl_port_display_rgb_cfg_t rgb_cfg = {
        .flags = {
            .bb_mode = true,
            .avoid_tearing = true,
        }
    };
    
    display_ = lvgl_port_add_disp_rgb(&display_cfg, &rgb_cfg);
    if (display_ == nullptr) {
        ESP_LOGE(TAG, "Failed to add RGB display");
        return;
    }
    
    if (offset_x != 0 || offset_y != 0) {
        lv_display_set_offset(display_, offset_x, offset_y);
    }

    SetupUI();
}

MipiLcdDisplay::MipiLcdDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel,
                            int width, int height,  int offset_x, int offset_y,
                            bool mirror_x, bool mirror_y, bool swap_xy)
    : LcdDisplay(panel_io, panel, width, height) {

    // Set the display to on
    ESP_LOGI(TAG, "Turning display on");
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_, true));

    ESP_LOGI(TAG, "Initialize LVGL library");
    lv_init();

    ESP_LOGI(TAG, "Initialize LVGL port");
    lvgl_port_cfg_t port_cfg = ESP_LVGL_PORT_INIT_CONFIG();
    lvgl_port_init(&port_cfg);

    ESP_LOGI(TAG, "Adding LCD display");
    const lvgl_port_display_cfg_t disp_cfg = {
        .io_handle = panel_io,
        .panel_handle = panel,
        .control_handle = nullptr,
        .buffer_size = static_cast<uint32_t>(width_ * 50),
        .double_buffer = false,
        .hres = static_cast<uint32_t>(width_),
        .vres = static_cast<uint32_t>(height_),
        .monochrome = false,
        /* Rotation values must be same as used in esp_lcd for initial settings of the screen */
        .rotation = {
            .swap_xy = swap_xy,
            .mirror_x = mirror_x,
            .mirror_y = mirror_y,
        },
        .flags = {
            .buff_dma = true,
            .buff_spiram =false,
            .sw_rotate = false,
        },
    };

    const lvgl_port_display_dsi_cfg_t dpi_cfg = {
        .flags = {
            .avoid_tearing = false,
        }
    };
    display_ = lvgl_port_add_disp_dsi(&disp_cfg, &dpi_cfg);
    if (display_ == nullptr) {
        ESP_LOGE(TAG, "Failed to add display");
        return;
    }

    if (offset_x != 0 || offset_y != 0) {
        lv_display_set_offset(display_, offset_x, offset_y);
    }

    SetupUI();
}

LcdDisplay::~LcdDisplay() {
    DisplayLockGuard display_lock(this);
    SetPreviewImage(nullptr);
    
    // Clean up GIF controller
    if (gif_controller_) {
        gif_controller_->Stop();
        gif_controller_.reset();
    }
    
    if (preview_timer_ != nullptr) {
        esp_timer_stop(preview_timer_);
        esp_timer_delete(preview_timer_);
    }

    if (preview_image_ != nullptr) {
        lv_obj_del(preview_image_);
    }
    if (chat_message_label_ != nullptr) {
        lv_obj_del(chat_message_label_);
    }
    if (emoji_label_ != nullptr) {
        lv_obj_del(emoji_label_);
    }
    if (emoji_image_ != nullptr) {
        lv_obj_del(emoji_image_);
    }
    if (emoji_box_ != nullptr) {
        lv_obj_del(emoji_box_);
    }
    if (content_ != nullptr) {
        lv_obj_del(content_);
    }
    if (status_bar_ != nullptr) {
        lv_obj_del(status_bar_);
    }
    if (side_bar_ != nullptr) {
        lv_obj_del(side_bar_);
    }
    if (container_ != nullptr) {
        lv_obj_del(container_);
    }
    if (display_ != nullptr) {
        lv_display_delete(display_);
    }

    if (panel_ != nullptr) {
        esp_lcd_panel_del(panel_);
    }
    if (panel_io_ != nullptr) {
        esp_lcd_panel_io_del(panel_io_);
    }
}

bool LcdDisplay::Lock(int timeout_ms) {
    return lvgl_port_lock(timeout_ms);
}

void LcdDisplay::Unlock() {
    lvgl_port_unlock();
}

#if CONFIG_USE_WECHAT_MESSAGE_STYLE
void LcdDisplay::SetupUI() {
    DisplayLockGuard lock(this);

    auto lvgl_theme = static_cast<LvglTheme*>(current_theme_);
    auto text_font = lvgl_theme->text_font()->font();
    auto icon_font = lvgl_theme->icon_font()->font();
    auto large_icon_font = lvgl_theme->large_icon_font()->font();

    auto screen = lv_screen_active();
    lv_obj_set_style_text_font(screen, text_font, 0);
    lv_obj_set_style_text_color(screen, lvgl_theme->text_color(), 0);
    lv_obj_set_style_bg_color(screen, lvgl_theme->background_color(), 0);

    /* Container */
    container_ = lv_obj_create(screen);
    lv_obj_set_size(container_, LV_HOR_RES, LV_VER_RES);
    lv_obj_set_style_radius(container_, 0, 0);
    lv_obj_set_flex_flow(container_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_all(container_, 0, 0);
    lv_obj_set_style_border_width(container_, 0, 0);
    lv_obj_set_style_pad_row(container_, 0, 0);
    lv_obj_set_style_bg_color(container_, lvgl_theme->background_color(), 0);
    lv_obj_set_style_border_color(container_, lvgl_theme->border_color(), 0);

    /* Status bar */
    status_bar_ = lv_obj_create(container_);
    lv_obj_set_size(status_bar_, LV_HOR_RES, LV_SIZE_CONTENT);
    lv_obj_set_style_radius(status_bar_, 0, 0);
    lv_obj_set_style_bg_color(status_bar_, lvgl_theme->background_color(), 0);
    lv_obj_set_style_text_color(status_bar_, lvgl_theme->text_color(), 0);
    
    /* Content - Chat area */
    content_ = lv_obj_create(container_);
    lv_obj_set_style_radius(content_, 0, 0);
    lv_obj_set_width(content_, LV_HOR_RES);
    lv_obj_set_flex_grow(content_, 1);
    lv_obj_set_style_pad_all(content_, lvgl_theme->spacing(4), 0);
    lv_obj_set_style_border_width(content_, 0, 0);
    lv_obj_set_style_bg_color(content_, lvgl_theme->chat_background_color(), 0); // Background for chat area

    // Enable scrolling for chat content
    lv_obj_set_scrollbar_mode(content_, LV_SCROLLBAR_MODE_OFF);
    lv_obj_set_scroll_dir(content_, LV_DIR_VER);
    
    // Create a flex container for chat messages
    lv_obj_set_flex_flow(content_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(content_, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_START);
    lv_obj_set_style_pad_row(content_, lvgl_theme->spacing(4), 0); // Space between messages

    // We'll create chat messages dynamically in SetChatMessage
    chat_message_label_ = nullptr;

    /* Status bar */
    lv_obj_set_flex_flow(status_bar_, LV_FLEX_FLOW_ROW);
    lv_obj_set_style_pad_all(status_bar_, 0, 0);
    lv_obj_set_style_border_width(status_bar_, 0, 0);
    lv_obj_set_style_pad_column(status_bar_, 0, 0);
    lv_obj_set_style_pad_top(status_bar_, lvgl_theme->spacing(2), 0);
    lv_obj_set_style_pad_bottom(status_bar_, lvgl_theme->spacing(2), 0);
    lv_obj_set_style_pad_left(status_bar_, lvgl_theme->spacing(4), 0);
    lv_obj_set_style_pad_right(status_bar_, lvgl_theme->spacing(4), 0);
    lv_obj_set_scrollbar_mode(status_bar_, LV_SCROLLBAR_MODE_OFF);
    // 设置状态栏的内容垂直居中
    lv_obj_set_flex_align(status_bar_, LV_FLEX_ALIGN_SPACE_BETWEEN, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);

    network_label_ = lv_label_create(status_bar_);
    lv_label_set_text(network_label_, "");
    lv_obj_set_style_text_font(network_label_, icon_font, 0);
    lv_obj_set_style_text_color(network_label_, lvgl_theme->text_color(), 0);

    notification_label_ = lv_label_create(status_bar_);
    lv_obj_set_flex_grow(notification_label_, 1);
    lv_obj_set_style_text_align(notification_label_, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(notification_label_, lvgl_theme->text_color(), 0);
    lv_label_set_text(notification_label_, "");
    lv_obj_add_flag(notification_label_, LV_OBJ_FLAG_HIDDEN);

    status_label_ = lv_label_create(status_bar_);
    lv_obj_set_flex_grow(status_label_, 1);
    lv_label_set_long_mode(status_label_, LV_LABEL_LONG_SCROLL_CIRCULAR);
    lv_obj_set_style_text_align(status_label_, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(status_label_, lvgl_theme->text_color(), 0);
    lv_label_set_text(status_label_, Lang::Strings::INITIALIZING);
    
    mute_label_ = lv_label_create(status_bar_);
    lv_label_set_text(mute_label_, "");
    lv_obj_set_style_text_font(mute_label_, icon_font, 0);
    lv_obj_set_style_text_color(mute_label_, lvgl_theme->text_color(), 0);

    battery_label_ = lv_label_create(status_bar_);
    lv_label_set_text(battery_label_, "");
    lv_obj_set_style_text_font(battery_label_, icon_font, 0);
    lv_obj_set_style_text_color(battery_label_, lvgl_theme->text_color(), 0);
    lv_obj_set_style_margin_left(battery_label_, lvgl_theme->spacing(2), 0); // 添加左边距，与前面的元素分隔

    low_battery_popup_ = lv_obj_create(screen);
    lv_obj_set_scrollbar_mode(low_battery_popup_, LV_SCROLLBAR_MODE_OFF);
    lv_obj_set_size(low_battery_popup_, LV_HOR_RES * 0.9, text_font->line_height * 2);
    lv_obj_align(low_battery_popup_, LV_ALIGN_BOTTOM_MID, 0, -lvgl_theme->spacing(4));
    lv_obj_set_style_bg_color(low_battery_popup_, lvgl_theme->low_battery_color(), 0);
    lv_obj_set_style_radius(low_battery_popup_, lvgl_theme->spacing(4), 0);
    low_battery_label_ = lv_label_create(low_battery_popup_);
    lv_label_set_text(low_battery_label_, Lang::Strings::BATTERY_NEED_CHARGE);
    lv_obj_set_style_text_color(low_battery_label_, lv_color_white(), 0);
    lv_obj_center(low_battery_label_);
    lv_obj_add_flag(low_battery_popup_, LV_OBJ_FLAG_HIDDEN);

    emoji_image_ = lv_img_create(screen);
    lv_obj_align(emoji_image_, LV_ALIGN_TOP_MID, 0, text_font->line_height + lvgl_theme->spacing(8));

    // Display AI logo while booting
    emoji_label_ = lv_label_create(screen);
    lv_obj_center(emoji_label_);
    lv_obj_set_style_text_font(emoji_label_, large_icon_font, 0);
    lv_obj_set_style_text_color(emoji_label_, lvgl_theme->text_color(), 0);
    lv_label_set_text(emoji_label_, FONT_AWESOME_MICROCHIP_AI);
}
#if CONFIG_IDF_TARGET_ESP32P4
#define  MAX_MESSAGES 40
#else
#define  MAX_MESSAGES 20
#endif
void LcdDisplay::SetChatMessage(const char* role, const char* content) {
    DisplayLockGuard lock(this);
    if (content_ == nullptr) {
        return;
    }
    
    // 检查消息数量是否超过限制
    uint32_t child_count = lv_obj_get_child_cnt(content_);
    if (child_count >= MAX_MESSAGES) {
        // 删除最早的消息（第一个子对象）
        lv_obj_t* first_child = lv_obj_get_child(content_, 0);
        lv_obj_t* last_child = lv_obj_get_child(content_, child_count - 1);
        if (first_child != nullptr) {
            lv_obj_del(first_child);
        }
        // Scroll to the last message immediately
        if (last_child != nullptr) {
            lv_obj_scroll_to_view_recursive(last_child, LV_ANIM_OFF);
        }
    }
    
    // 折叠系统消息（如果是系统消息，检查最后一个消息是否也是系统消息）
    if (strcmp(role, "system") == 0) {
        if (child_count > 0) {
            // 获取最后一个消息容器
            lv_obj_t* last_container = lv_obj_get_child(content_, child_count - 1);
            if (last_container != nullptr && lv_obj_get_child_cnt(last_container) > 0) {
                // 获取容器内的气泡
                lv_obj_t* last_bubble = lv_obj_get_child(last_container, 0);
                if (last_bubble != nullptr) {
                    // 检查气泡类型是否为系统消息
                    void* bubble_type_ptr = lv_obj_get_user_data(last_bubble);
                    if (bubble_type_ptr != nullptr && strcmp((const char*)bubble_type_ptr, "system") == 0) {
                        // 如果最后一个消息也是系统消息，则删除它
                        lv_obj_del(last_container);
                    }
                }
            }
        }
    } else {
        // 隐藏居中显示的 AI logo
        lv_obj_add_flag(emoji_label_, LV_OBJ_FLAG_HIDDEN);
    }

    //避免出现空的消息框
    if(strlen(content) == 0) {
        return;
    }

    auto lvgl_theme = static_cast<LvglTheme*>(current_theme_);
    auto text_font = lvgl_theme->text_font()->font();

    // Create a message bubble
    lv_obj_t* msg_bubble = lv_obj_create(content_);
    lv_obj_set_style_radius(msg_bubble, 8, 0);
    lv_obj_set_scrollbar_mode(msg_bubble, LV_SCROLLBAR_MODE_OFF);
    lv_obj_set_style_border_width(msg_bubble, 0, 0);
    lv_obj_set_style_pad_all(msg_bubble, lvgl_theme->spacing(4), 0);

    // Create the message text
    lv_obj_t* msg_text = lv_label_create(msg_bubble);
    lv_label_set_text(msg_text, content);
    
    // 计算文本实际宽度
    lv_coord_t text_width = lv_txt_get_width(content, strlen(content), text_font, 0);

    // 计算气泡宽度
    lv_coord_t max_width = LV_HOR_RES * 85 / 100 - 16;  // 屏幕宽度的85%
    lv_coord_t min_width = 20;  
    lv_coord_t bubble_width;
    
    // 确保文本宽度不小于最小宽度
    if (text_width < min_width) {
        text_width = min_width;
    }

    // 如果文本宽度小于最大宽度，使用文本宽度
    if (text_width < max_width) {
        bubble_width = text_width; 
    } else {
        bubble_width = max_width;
    }
    
    // 设置消息文本的宽度
    lv_obj_set_width(msg_text, bubble_width);  // 减去padding
    lv_label_set_long_mode(msg_text, LV_LABEL_LONG_WRAP);

    // 设置气泡宽度
    lv_obj_set_width(msg_bubble, bubble_width);
    lv_obj_set_height(msg_bubble, LV_SIZE_CONTENT);

    // Set alignment and style based on message role
    if (strcmp(role, "user") == 0) {
        // User messages are right-aligned with green background
        lv_obj_set_style_bg_color(msg_bubble, lvgl_theme->user_bubble_color(), 0);
        lv_obj_set_style_bg_opa(msg_bubble, LV_OPA_70, 0);
        // Set text color for contrast
        lv_obj_set_style_text_color(msg_text, lvgl_theme->text_color(), 0);
        
        // 设置自定义属性标记气泡类型
        lv_obj_set_user_data(msg_bubble, (void*)"user");
        
        // Set appropriate width for content
        lv_obj_set_width(msg_bubble, LV_SIZE_CONTENT);
        lv_obj_set_height(msg_bubble, LV_SIZE_CONTENT);
        
        // Don't grow
        lv_obj_set_style_flex_grow(msg_bubble, 0, 0);
    } else if (strcmp(role, "assistant") == 0) {
        // Assistant messages are left-aligned with white background
        lv_obj_set_style_bg_color(msg_bubble, lvgl_theme->assistant_bubble_color(), 0);
        lv_obj_set_style_bg_opa(msg_bubble, LV_OPA_70, 0);
        // Set text color for contrast
        lv_obj_set_style_text_color(msg_text, lvgl_theme->text_color(), 0);
        
        // 设置自定义属性标记气泡类型
        lv_obj_set_user_data(msg_bubble, (void*)"assistant");
        
        // Set appropriate width for content
        lv_obj_set_width(msg_bubble, LV_SIZE_CONTENT);
        lv_obj_set_height(msg_bubble, LV_SIZE_CONTENT);
        
        // Don't grow
        lv_obj_set_style_flex_grow(msg_bubble, 0, 0);
    } else if (strcmp(role, "system") == 0) {
        // System messages are center-aligned with light gray background
        lv_obj_set_style_bg_color(msg_bubble, lvgl_theme->system_bubble_color(), 0);
        lv_obj_set_style_bg_opa(msg_bubble, LV_OPA_70, 0);
        // Set text color for contrast
        lv_obj_set_style_text_color(msg_text, lvgl_theme->system_text_color(), 0);
        
        // 设置自定义属性标记气泡类型
        lv_obj_set_user_data(msg_bubble, (void*)"system");
        
        // Set appropriate width for content
        lv_obj_set_width(msg_bubble, LV_SIZE_CONTENT);
        lv_obj_set_height(msg_bubble, LV_SIZE_CONTENT);
        
        // Don't grow
        lv_obj_set_style_flex_grow(msg_bubble, 0, 0);
    }
    
    // Create a full-width container for user messages to ensure right alignment
    if (strcmp(role, "user") == 0) {
        // Create a full-width container
        lv_obj_t* container = lv_obj_create(content_);
        lv_obj_set_width(container, LV_HOR_RES);
        lv_obj_set_height(container, LV_SIZE_CONTENT);
        
        // Make container transparent and borderless
        lv_obj_set_style_bg_opa(container, LV_OPA_TRANSP, 0);
        lv_obj_set_style_border_width(container, 0, 0);
        lv_obj_set_style_pad_all(container, 0, 0);
        
        // Move the message bubble into this container
        lv_obj_set_parent(msg_bubble, container);
        
        // Right align the bubble in the container
        lv_obj_align(msg_bubble, LV_ALIGN_RIGHT_MID, -25, 0);
        
        // Auto-scroll to this container
        lv_obj_scroll_to_view_recursive(container, LV_ANIM_ON);
    } else if (strcmp(role, "system") == 0) {
        // 为系统消息创建全宽容器以确保居中对齐
        lv_obj_t* container = lv_obj_create(content_);
        lv_obj_set_width(container, LV_HOR_RES);
        lv_obj_set_height(container, LV_SIZE_CONTENT);
        
        // 使容器透明且无边框
        lv_obj_set_style_bg_opa(container, LV_OPA_TRANSP, 0);
        lv_obj_set_style_border_width(container, 0, 0);
        lv_obj_set_style_pad_all(container, 0, 0);
        
        // 将消息气泡移入此容器
        lv_obj_set_parent(msg_bubble, container);
        
        // 将气泡居中对齐在容器中
        lv_obj_align(msg_bubble, LV_ALIGN_CENTER, 0, 0);
        
        // 自动滚动底部
        lv_obj_scroll_to_view_recursive(container, LV_ANIM_ON);
    } else {
        // For assistant messages
        // Left align assistant messages
        lv_obj_align(msg_bubble, LV_ALIGN_LEFT_MID, 0, 0);

        // Auto-scroll to the message bubble
        lv_obj_scroll_to_view_recursive(msg_bubble, LV_ANIM_ON);
    }
    
    // Store reference to the latest message label
    chat_message_label_ = msg_text;
}

void LcdDisplay::SetPreviewImage(std::unique_ptr<LvglImage> image) {
    DisplayLockGuard lock(this);
    if (content_ == nullptr) {
        return;
    }

    // Leave the vision dashboard before showing a normal chat image.  The
    // preview object is temporarily parented to the video tile while sensing.
    if (vision_dashboard_ != nullptr) {
        lv_obj_add_flag(vision_dashboard_, LV_OBJ_FLAG_HIDDEN);
        if (preview_image_ != nullptr) {
            lv_obj_set_parent(preview_image_, content_);
            lv_obj_add_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);
        }
        if (status_bar_ != nullptr) {
            lv_obj_remove_flag(status_bar_, LV_OBJ_FLAG_HIDDEN);
        }
        lv_obj_set_flex_flow(content_, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_grow(content_, 1);
        lv_obj_set_size(content_, LV_HOR_RES, LV_SIZE_CONTENT);
    }

    if (image == nullptr) {
        preview_image_cached_.reset();
        return;
    }
    
    auto lvgl_theme = static_cast<LvglTheme*>(current_theme_);
    // Create a message bubble for image preview
    lv_obj_t* img_bubble = lv_obj_create(content_);
    lv_obj_set_style_radius(img_bubble, 8, 0);
    lv_obj_set_scrollbar_mode(img_bubble, LV_SCROLLBAR_MODE_OFF);
    lv_obj_set_style_border_width(img_bubble, 0, 0);
    lv_obj_set_style_pad_all(img_bubble, lvgl_theme->spacing(4), 0);
    
    // Set image bubble background color (similar to system message)
    lv_obj_set_style_bg_color(img_bubble, lvgl_theme->assistant_bubble_color(), 0);
    lv_obj_set_style_bg_opa(img_bubble, LV_OPA_70, 0);
    
    // 设置自定义属性标记气泡类型
    lv_obj_set_user_data(img_bubble, (void*)"image");

    // Create the image object inside the bubble
    lv_obj_t* preview_image = lv_image_create(img_bubble);
    
    // Calculate appropriate size for the image
    lv_coord_t max_width = LV_HOR_RES * 70 / 100;  // 70% of screen width
    lv_coord_t max_height = LV_VER_RES * 50 / 100; // 50% of screen height
    
    // Calculate zoom factor to fit within maximum dimensions
    auto img_dsc = image->image_dsc();
    lv_coord_t img_width = img_dsc->header.w;
    lv_coord_t img_height = img_dsc->header.h;
    if (img_width == 0 || img_height == 0) {
        img_width = max_width;
        img_height = max_height;
        ESP_LOGW(TAG, "Invalid image dimensions: %ld x %ld, using default dimensions: %ld x %ld", img_width, img_height, max_width, max_height);
    }
    
    lv_coord_t zoom_w = (max_width * 256) / img_width;
    lv_coord_t zoom_h = (max_height * 256) / img_height;
    lv_coord_t zoom = (zoom_w < zoom_h) ? zoom_w : zoom_h;
    
    // Ensure zoom doesn't exceed 256 (100%)
    if (zoom > 256) zoom = 256;
    
    // Set image properties
    lv_image_set_src(preview_image, img_dsc);
    lv_image_set_scale(preview_image, zoom);
    
    // Add event handler to clean up LvglImage when image is deleted
    // We need to transfer ownership of the unique_ptr to the event callback
    LvglImage* raw_image = image.release(); // 释放智能指针的所有权
    lv_obj_add_event_cb(preview_image, [](lv_event_t* e) {
        LvglImage* img = (LvglImage*)lv_event_get_user_data(e);
        if (img != nullptr) {
            delete img; // 通过删除 LvglImage 对象来正确释放内存
        }
    }, LV_EVENT_DELETE, (void*)raw_image);
    
    // Calculate actual scaled image dimensions
    lv_coord_t scaled_width = (img_width * zoom) / 256;
    lv_coord_t scaled_height = (img_height * zoom) / 256;
    
    // Set bubble size to be 16 pixels larger than the image (8 pixels on each side)
    lv_obj_set_width(img_bubble, scaled_width + 16);
    lv_obj_set_height(img_bubble, scaled_height + 16);
    
    // Don't grow in flex layout
    lv_obj_set_style_flex_grow(img_bubble, 0, 0);
    
    // Center the image within the bubble
    lv_obj_center(preview_image);
    
    // Left align the image bubble like assistant messages
    lv_obj_align(img_bubble, LV_ALIGN_LEFT_MID, 0, 0);

    // Auto-scroll to the image bubble
    lv_obj_scroll_to_view_recursive(img_bubble, LV_ANIM_ON);
}

void LcdDisplay::SetVisionPreviewImage(std::unique_ptr<LvglImage> image,
                                       const DetectionResult& result) {
    // Wait for the preceding LVGL flush instead of repeatedly dropping frames
    // after a 5 ms try-lock. The short timeout caused most preview submissions
    // to be discarded (and logged), reducing the visible rate to 3-5 FPS.
    DisplayLockGuard lock(this, 30);
    if (!lock) return;
    if (content_ == nullptr || preview_image_ == nullptr || image == nullptr) return;
    esp_timer_stop(preview_timer_);

    // Reset transient artwork before applying this frame.  The image objects
    // are reused, so leaving one visible causes stale cards to flash over the
    // next frame.

    // The detection dashboard is a full-screen mode. Hide the normal status
    // bar so Wi-Fi, battery and clock do not consume layout height.
    if (status_bar_ != nullptr) {
        lv_obj_add_flag(status_bar_, LV_OBJ_FLAG_HIDDEN);
    }

    auto theme = static_cast<LvglTheme*>(current_theme_);
    const lv_coord_t w = LV_HOR_RES;
    const lv_coord_t h = LV_VER_RES;
    const lv_coord_t left_w = std::max<lv_coord_t>(52, w / 6);
    const lv_coord_t center_w = w - left_w * 2;
    const lv_coord_t preview_w = (center_w - 8) * 9 / 10;
    const lv_coord_t side_w = left_w + (center_w - preview_w) / 2;
    const lv_coord_t card_icon_w = std::max<lv_coord_t>(48, side_w - 6);
    const lv_coord_t bottom_h = 30;
    // The target panel is 320x240; keep the header compact.
        const lv_coord_t title_h = 20;

    lv_obj_set_layout(content_, LV_LAYOUT_NONE);
    lv_obj_set_size(content_, w, h);

    if (vision_dashboard_ == nullptr) {
        vision_dashboard_ = lv_obj_create(content_);
        lv_obj_set_scrollbar_mode(vision_dashboard_, LV_SCROLLBAR_MODE_OFF);
        lv_obj_clear_flag(vision_dashboard_, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_size(vision_dashboard_, w, h);
        lv_obj_set_pos(vision_dashboard_, 0, 0);
        lv_obj_set_style_pad_all(vision_dashboard_, 0, 0);
        lv_obj_set_style_border_width(vision_dashboard_, 0, 0);
        // Cartoon dashboard palette adapted from the reference image.
        lv_obj_set_style_bg_color(vision_dashboard_, lv_color_white(), 0);
        void* dashboard_data = nullptr;
        size_t dashboard_size = 0;
        if (LoadVisionDashboard(dashboard_data, dashboard_size)) {
            vision_background_image_ = std::make_unique<LvglRawImage>(dashboard_data, dashboard_size);
            vision_background_view_ = lv_image_create(vision_dashboard_);
            lv_obj_clear_flag(vision_background_view_, LV_OBJ_FLAG_SCROLLABLE);
            lv_image_set_src(vision_background_view_, vision_background_image_->image_dsc());
            lv_obj_set_size(vision_background_view_, w, h);
            lv_obj_set_pos(vision_background_view_, 0, 0);
        }
        lv_obj_t* bottom_mask = lv_obj_create(vision_dashboard_);
        lv_obj_set_size(bottom_mask, w, bottom_h);
        lv_obj_set_pos(bottom_mask, 0, h - bottom_h);
        lv_obj_set_style_bg_color(bottom_mask, lv_color_white(), 0);
        lv_obj_set_style_bg_opa(bottom_mask, LV_OPA_COVER, 0);
        lv_obj_set_style_border_width(bottom_mask, 0, 0);
        lv_obj_set_style_pad_all(bottom_mask, 0, 0);

        auto panel = [&](lv_coord_t x, lv_coord_t y, lv_coord_t pw, lv_coord_t ph) {
            lv_obj_t* p = lv_obj_create(vision_dashboard_);
            lv_obj_set_size(p, pw, ph); lv_obj_set_pos(p, x, y);
            lv_obj_set_scrollbar_mode(p, LV_SCROLLBAR_MODE_OFF);
            lv_obj_clear_flag(p, LV_OBJ_FLAG_SCROLLABLE);
            lv_obj_set_style_pad_all(p, 0, 0);
            lv_obj_set_style_bg_color(p, lv_color_hex(0xFFFFFF), 0);
            lv_obj_set_style_border_width(p, 0, 0);
            return p;
        };
        // Expand side cards into the existing preview margins.
        const lv_coord_t video_w = (center_w - 8) * 9 / 10;
        const lv_coord_t side_w = left_w + (center_w - video_w) / 2;
        const lv_coord_t card_icon_w = std::max<lv_coord_t>(48, side_w - 6);
        vision_heart_panel_ = panel(0, 0, side_w, h / 2);
        vision_pressure_panel_ = panel(0, h / 2, side_w, h - h / 2);
        vision_posture_panel_ = panel(w - side_w, 0, side_w, h / 2);
        vision_emotion_panel_ = panel(w - side_w, h / 2, side_w, h - h / 2);
        StyleVisionCard(vision_heart_panel_, lv_color_hex(0xFFF5F7), lv_color_hex(0xFF8FA3));
        StyleVisionCard(vision_pressure_panel_, lv_color_hex(0xF1F7FF), lv_color_hex(0x6EA7FF));
        StyleVisionCard(vision_posture_panel_, lv_color_hex(0xF2FFF3), lv_color_hex(0x80D780));
        StyleVisionCard(vision_emotion_panel_, lv_color_hex(0xFFF9E9), lv_color_hex(0xFFD16A));
        for (auto* panel : {vision_heart_panel_, vision_pressure_panel_, vision_posture_panel_, vision_emotion_panel_}) {
            lv_obj_set_style_bg_opa(panel, LV_OPA_TRANSP, 0);
            lv_obj_set_style_border_width(panel, 0, 0);
            lv_obj_set_style_radius(panel, 0, 0);
        }
        vision_heart_icon_ = lv_label_create(vision_heart_panel_);
        vision_pressure_icon_ = lv_label_create(vision_pressure_panel_);
        vision_posture_cartoon_ = CreateVisionPostureCartoon(vision_posture_panel_);
        vision_posture_icon_ = lv_line_create(vision_posture_panel_);
        vision_posture_image_ = lv_image_create(vision_posture_panel_);
        vision_emotion_image_ = lv_image_create(vision_emotion_panel_);
        vision_heart_label_ = lv_label_create(vision_heart_panel_);
        vision_pressure_label_ = lv_label_create(vision_pressure_panel_);
        vision_posture_label_ = lv_label_create(vision_posture_panel_);
        vision_emotion_label_ = lv_label_create(vision_emotion_panel_);
        lv_obj_add_flag(vision_heart_icon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_pressure_icon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_cartoon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_icon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
        for (auto* label : {vision_heart_label_, vision_pressure_label_, vision_posture_label_, vision_emotion_label_}) {
            lv_obj_set_width(label, side_w - 4);
            lv_obj_set_height(label, LV_SIZE_CONTENT);
            lv_obj_align(label, LV_ALIGN_BOTTOM_MID, 0, -5);
            lv_obj_clear_flag(label, LV_OBJ_FLAG_HIDDEN);
            lv_label_set_long_mode(label, LV_LABEL_LONG_WRAP);
            lv_obj_set_style_text_align(label, LV_TEXT_ALIGN_CENTER, 0);
            lv_obj_set_style_text_font(label, theme->text_font()->font(), 0);
            lv_obj_set_style_text_color(label, lv_color_hex(0x243447), 0);
            lv_obj_set_style_transform_zoom(label, 220, 0);
            lv_obj_set_style_text_line_space(label, 2, 0);
        }
        // The common label setup above aligns labels to the bottom. Restore
        // the intended fixed positions for the two numeric cards afterwards.
        lv_obj_set_align(vision_heart_label_, LV_ALIGN_DEFAULT);
        lv_obj_set_pos(vision_heart_label_, 14, 47);
        lv_obj_set_width(vision_heart_label_, 44);
        lv_obj_set_align(vision_pressure_label_, LV_ALIGN_DEFAULT);
        lv_obj_set_pos(vision_pressure_label_, 2, 31);
        lv_obj_set_width(vision_pressure_label_, 64);
        lv_obj_set_style_text_color(vision_heart_label_, lv_color_hex(0xE53935), 0);
        auto setup_icon = [&](lv_obj_t* icon, const char* glyph, lv_color_t color) {
            lv_obj_set_width(icon, side_w);
            lv_obj_set_height(icon, LV_SIZE_CONTENT);
            lv_obj_set_style_text_font(icon, theme->large_icon_font()->font(), 0);
            lv_obj_set_style_text_color(icon, color, 0);
            lv_label_set_text(icon, glyph);
            lv_obj_align(icon, LV_ALIGN_TOP_MID, 0, 8);
        };
        setup_icon(vision_heart_icon_, FONT_AWESOME_HEART, lv_color_hex(0xE53935));
        // A static monitor pictogram keeps the blood-pressure card aligned
        // with the reference design without competing with its value.
        setup_icon(vision_pressure_icon_, FONT_AWESOME_CALCULATOR, lv_color_hex(0x4C8DFF));
        lv_obj_set_style_text_font(vision_pressure_icon_, theme->large_icon_font()->font(), 0);
        lv_obj_set_size(vision_posture_icon_, 52, 56);
        lv_obj_align(vision_posture_icon_, LV_ALIGN_TOP_MID, 0, 5);
        SetVisionPostureGraphic(vision_posture_icon_, false);
        lv_obj_add_flag(vision_posture_label_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_set_size(vision_emotion_image_, 52, 52);
        lv_obj_align(vision_emotion_image_, LV_ALIGN_TOP_MID, 0, 5);
        lv_obj_add_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_t* title_panel = panel(left_w, 0, center_w, title_h);
        StyleVisionCard(title_panel, lv_color_hex(0xEAF4FF), lv_color_hex(0x3F8FE8), 7);
        CreateVisionBear(title_panel);
        lv_obj_t* title_label = lv_label_create(title_panel);
        // Leave a clear gap after the bear so the title never overlaps it.
        lv_obj_set_width(title_label, center_w - 36);
        lv_label_set_long_mode(title_label, LV_LABEL_LONG_CLIP);
        lv_obj_set_style_text_align(title_label, LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_set_style_text_color(title_label, lv_color_hex(0x24507A), 0);
        lv_obj_set_style_transform_zoom(title_label, 180, 0);
        lv_label_set_text(title_label, "坐姿健康检测系统");
        lv_obj_align(title_label, LV_ALIGN_RIGHT_MID, -3, 0);
        // The supplied background already contains the final bear and title.
        lv_obj_add_flag(title_panel, LV_OBJ_FLAG_HIDDEN);
        lv_obj_t* video_area = panel(left_w, title_h, center_w, h - title_h - bottom_h);
        lv_obj_set_style_bg_opa(video_area, LV_OPA_TRANSP, 0);
        lv_obj_set_style_border_width(video_area, 0, 0);
        lv_obj_set_parent(preview_image_, vision_dashboard_);
        // Exact inner video window of the 320x240 artwork.
        const lv_coord_t video_h = std::min<lv_coord_t>(h - title_h - bottom_h - 8, video_w * 3 / 4);
        lv_obj_set_size(preview_image_, video_w, video_h);
        lv_obj_set_align(preview_image_, LV_ALIGN_DEFAULT);
        lv_obj_set_pos(preview_image_, left_w + (center_w - video_w) / 2,
                       title_h + (h - title_h - bottom_h - video_h) / 2);
        lv_image_set_inner_align(preview_image_, LV_IMAGE_ALIGN_COVER);
        lv_obj_set_style_radius(preview_image_, 0, 0);
        if (chat_message_label_ != nullptr) {
            lv_obj_set_parent(chat_message_label_, vision_dashboard_);
            lv_obj_set_size(chat_message_label_, w - 8, bottom_h - 4);
            lv_obj_set_pos(chat_message_label_, 4, h - bottom_h + 2);
            lv_obj_set_style_text_align(chat_message_label_, LV_TEXT_ALIGN_CENTER, 0);
            lv_obj_set_style_text_color(chat_message_label_, lv_color_hex(0x24507A), 0);
            lv_obj_set_style_bg_color(chat_message_label_, lv_color_white(), 0);
            lv_obj_set_style_bg_opa(chat_message_label_, LV_OPA_COVER, 0);
            lv_obj_set_style_pad_all(chat_message_label_, 2, 0);
            lv_label_set_long_mode(chat_message_label_, LV_LABEL_LONG_SCROLL_CIRCULAR);
            lv_obj_remove_flag(chat_message_label_, LV_OBJ_FLAG_HIDDEN);
        }
    }

    PrepareVisionFrame(h - bottom_h);
    char text[128];
    // This label is refreshed on every preview frame; only the heart glyph
    // is static.  Keep the live value inside the red heart as in the mockup.
    if (result.heart_rate.available) {
        snprintf(text, sizeof(text), "%.0f", result.heart_rate.bpm);
        lv_label_set_text(vision_heart_label_, text);
    } else {
        lv_label_set_text(vision_heart_label_, "");
    }
    if (result.blood_pressure.available) {
        snprintf(text, sizeof(text), "%.0f\n%.0f",
                 result.blood_pressure.sbp_mmHg, result.blood_pressure.dbp_mmHg);
        lv_label_set_text(vision_pressure_label_, text);
    } else {
        lv_label_set_text(vision_pressure_label_, "");
    }
    if (result.posture.available) {
        bool posture_bad = result.posture.state == "bad_posture";
        lv_obj_remove_flag(vision_posture_label_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_align(vision_posture_label_, LV_ALIGN_TOP_MID, 0, 101);
        lv_label_set_text(vision_posture_label_, posture_bad ? "坐姿不正" : "坐姿端正");
        const char* posture_asset = posture_bad ? "vision_posture_incorrect.png" : "vision_posture_correct.png";
        if (vision_posture_asset_name_ != posture_asset) {
            auto posture_image = LoadVisionAsset(posture_asset);
            if (posture_image != nullptr) {
                lv_image_set_src(vision_posture_image_, posture_image->image_dsc());
                vision_posture_asset_image_ = std::move(posture_image);
                vision_posture_asset_name_ = posture_asset;
        lv_obj_set_size(vision_posture_image_, card_icon_w, 60);
        lv_obj_set_align(vision_posture_image_, LV_ALIGN_TOP_MID);
        lv_obj_set_pos(vision_posture_image_, 0, 25);
                lv_image_set_inner_align(vision_posture_image_, LV_IMAGE_ALIGN_CONTAIN);
                lv_obj_remove_flag(vision_posture_image_, LV_OBJ_FLAG_HIDDEN);
            }
        }
        lv_obj_remove_flag(vision_posture_image_, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(vision_posture_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_label_, LV_OBJ_FLAG_HIDDEN);
    }
    if (result.emotion.available) {
        std::string emotion_asset_name = "vision_emotion_";
        emotion_asset_name += EmotionAssetKey(result.emotion.label);
        emotion_asset_name += ".png";
        if (vision_emotion_asset_name_ != emotion_asset_name) {
            auto emotion_image = LoadVisionAsset(emotion_asset_name.c_str());
            if (emotion_image != nullptr) {
                lv_image_set_src(vision_emotion_image_, emotion_image->image_dsc());
                vision_emotion_asset_image_ = std::move(emotion_image);
                vision_emotion_asset_name_ = emotion_asset_name;
        lv_obj_set_size(vision_emotion_image_, card_icon_w, 60);
        lv_obj_set_align(vision_emotion_image_, LV_ALIGN_TOP_MID);
        lv_obj_set_pos(vision_emotion_image_, 0, 25);
                lv_image_set_inner_align(vision_emotion_image_, LV_IMAGE_ALIGN_CONTAIN);
                lv_obj_remove_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
            }
        }
        lv_obj_remove_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
        lv_label_set_text(vision_emotion_label_, "");
    }
    lv_obj_add_flag(vision_emotion_label_, LV_OBJ_FLAG_HIDDEN);
    bool hr_alert = result.heart_rate.available && result.heart_rate.bpm > 120.0f;
    bool bp_alert = result.blood_pressure.ready &&
                    (result.blood_pressure.sbp_mmHg >= 140.0f ||
                     result.blood_pressure.dbp_mmHg >= 90.0f);
    bool posture_alert = result.posture.available && result.posture.state == "bad_posture";
    if (false && hr_alert != vision_hr_alert_active_) {
        vision_hr_alert_active_ = hr_alert;
        lv_obj_set_style_bg_color(vision_heart_panel_, hr_alert ? lv_color_hex(0xE85D75) : lv_color_hex(0xFFF5F7), 0);
        if (hr_alert) StartVisionAlertAnimation(vision_heart_panel_);
        else StopVisionAlertAnimation(vision_heart_panel_);
    }
    if (false && bp_alert != vision_bp_alert_active_) {
        vision_bp_alert_active_ = bp_alert;
        lv_obj_set_style_bg_color(vision_pressure_panel_, bp_alert ? lv_color_hex(0xE85D75) : lv_color_hex(0xF1F7FF), 0);
        if (bp_alert) StartVisionAlertAnimation(vision_pressure_panel_);
        else StopVisionAlertAnimation(vision_pressure_panel_);
    }
    if (false && posture_alert != vision_posture_alert_active_) {
        vision_posture_alert_active_ = posture_alert;
        lv_obj_set_style_bg_color(vision_posture_panel_, posture_alert ? lv_color_hex(0xE85D75) : lv_color_hex(0xF2FFF3), 0);
        if (posture_alert) StartVisionAlertAnimation(vision_posture_panel_);
        else StopVisionAlertAnimation(vision_posture_panel_);
    }
    lv_color_t emotion_color = lv_color_hex(0x808080);
    if (result.emotion.label == "happy") emotion_color = lv_color_hex(0x22AA55);
    else if (result.emotion.label == "sad") emotion_color = lv_color_hex(0x3388CC);
    else if (result.emotion.label == "angry") emotion_color = lv_color_hex(0xDD3333);
    else if (result.emotion.label == "fear") emotion_color = lv_color_hex(0x9944CC);
    else if (result.emotion.label == "surprise") emotion_color = lv_color_hex(0xEE9922);
    lv_obj_set_style_text_color(vision_emotion_label_, emotion_color, 0);
    // Re-assert the full-screen dashboard viewport on every camera frame.
    lv_obj_set_parent(preview_image_, vision_dashboard_);
    lv_obj_set_align(preview_image_, LV_ALIGN_DEFAULT);
    const lv_coord_t video_w = preview_w;
    const lv_coord_t video_h = std::min<lv_coord_t>(h - title_h - bottom_h - 8, video_w * 3 / 4);
    lv_obj_set_size(preview_image_, video_w, video_h);
    lv_obj_set_pos(preview_image_, side_w,
                   title_h + (h - title_h - bottom_h - video_h) / 2);
    lv_obj_remove_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);
    PublishVisionFrame(std::move(image));
    // Keep the preview below the four information cards, but above the
    // dashboard background image.
    lv_obj_move_to_index(preview_image_, 1);
    lv_obj_add_flag(emoji_box_, LV_OBJ_FLAG_HIDDEN);
    if (chat_message_label_ != nullptr) lv_obj_remove_flag(chat_message_label_, LV_OBJ_FLAG_HIDDEN);
    lv_obj_remove_flag(vision_dashboard_, LV_OBJ_FLAG_HIDDEN);
    lv_obj_remove_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);
}
#else
void LcdDisplay::SetVisionPreviewImage(std::unique_ptr<LvglImage> image,
                                       const DetectionResult& result) {
    // Wait for the preceding LVGL flush instead of repeatedly dropping frames
    // after a 5 ms try-lock. This remains bounded below the 50 ms frame period.
    DisplayLockGuard lock(this, 30);
    if (!lock) return;
    if (!image || !content_ || !preview_image_) return;
    esp_timer_stop(preview_timer_);
    if (status_bar_ != nullptr) {
        lv_obj_add_flag(status_bar_, LV_OBJ_FLAG_HIDDEN);
    }
    const lv_coord_t w = LV_HOR_RES, h = LV_VER_RES;
    // Coordinates mirror the 320x240 dashboard_5 artwork. The lower strip is
    // a dedicated speech-text channel and must never be covered by cards.
    const lv_coord_t th = std::max<lv_coord_t>(20, h * 28 / 240);
    const lv_coord_t caption_y = h * 199 / 240;
    const lv_coord_t caption_h = std::max<lv_coord_t>(18, h * 21 / 240);
    const lv_coord_t split_y = h * 107 / 240;
    const lv_coord_t side_w = std::max<lv_coord_t>(64, w * 70 / 320);
    const lv_coord_t cw = w - side_w * 2;
    const auto viewport = VisionVideoViewport(w, h);
    const lv_coord_t preview_x = viewport.x;
    const lv_coord_t preview_y = viewport.y;
    const lv_coord_t preview_w = viewport.width;
    const lv_coord_t preview_h = viewport.height;
    const bool entering_vision = vision_dashboard_ == nullptr ||
                                 lv_obj_get_parent(preview_image_) != vision_dashboard_;
    if (entering_vision) {
        lv_obj_set_layout(content_, LV_LAYOUT_NONE);
        lv_obj_set_size(content_, w, h);
        lv_obj_scroll_to(content_, 0, 0, LV_ANIM_OFF);
    }
    if (!vision_dashboard_) {
        vision_dashboard_ = lv_obj_create(content_);
        lv_obj_set_style_border_width(vision_dashboard_, 0, 0);
        lv_obj_set_style_radius(vision_dashboard_, 0, 0);
        lv_obj_set_pos(vision_dashboard_, 0, 0);
        lv_obj_set_scrollbar_mode(vision_dashboard_, LV_SCROLLBAR_MODE_OFF);
        lv_obj_clear_flag(vision_dashboard_, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_size(vision_dashboard_, w, h); lv_obj_set_style_pad_all(vision_dashboard_, 0, 0);
        lv_obj_set_style_bg_color(vision_dashboard_, lv_color_white(), 0);
        void* dashboard_data = nullptr;
        size_t dashboard_size = 0;
        if (LoadVisionDashboard(dashboard_data, dashboard_size)) {
            vision_background_image_ = std::make_unique<LvglRawImage>(dashboard_data, dashboard_size);
            vision_background_view_ = lv_image_create(vision_dashboard_);
            lv_obj_clear_flag(vision_background_view_, LV_OBJ_FLAG_SCROLLABLE);
            lv_image_set_src(vision_background_view_, vision_background_image_->image_dsc());
            lv_obj_set_size(vision_background_view_, w, h);
            lv_obj_set_pos(vision_background_view_, 0, 0);
        }
        auto p = [&](lv_coord_t x, lv_coord_t y, lv_coord_t pw, lv_coord_t ph) { lv_obj_t* o = lv_obj_create(vision_dashboard_); lv_obj_set_size(o, pw, ph); lv_obj_set_pos(o, x, y); lv_obj_set_scrollbar_mode(o, LV_SCROLLBAR_MODE_OFF); lv_obj_clear_flag(o, LV_OBJ_FLAG_SCROLLABLE); lv_obj_set_style_pad_all(o, 0, 0); lv_obj_set_style_bg_color(o, lv_color_hex(0xFFFFFF), 0); lv_obj_set_style_border_width(o, 0, 0); return o; };
        auto l = [&](lv_obj_t* o) { lv_obj_t* x = lv_label_create(o); lv_obj_center(x); lv_obj_set_style_text_align(x, LV_TEXT_ALIGN_CENTER, 0); return x; };
        vision_heart_panel_ = p(0, 0, side_w, split_y);
        vision_pressure_panel_ = p(0, split_y, side_w, caption_y - split_y);
        vision_heart_label_ = l(vision_heart_panel_); vision_pressure_label_ = l(vision_pressure_panel_);
        vision_posture_panel_ = p(w - side_w, 0, side_w, split_y);
        vision_posture_label_ = l(vision_posture_panel_);
        vision_emotion_panel_ = p(w - side_w, split_y, side_w, caption_y - split_y);
        vision_emotion_label_ = l(vision_emotion_panel_);
        vision_heart_icon_ = lv_label_create(vision_heart_panel_);
        vision_pressure_icon_ = lv_label_create(vision_pressure_panel_);
        vision_posture_cartoon_ = CreateVisionPostureCartoon(vision_posture_panel_);
        vision_posture_icon_ = lv_line_create(vision_posture_panel_);
        vision_posture_image_ = lv_image_create(vision_posture_panel_);
        vision_emotion_image_ = lv_image_create(vision_emotion_panel_);
        StyleVisionCard(vision_heart_panel_, lv_color_hex(0xFFF5F7), lv_color_hex(0xFF8FA3));
        StyleVisionCard(vision_pressure_panel_, lv_color_hex(0xF1F7FF), lv_color_hex(0x6EA7FF));
        StyleVisionCard(vision_posture_panel_, lv_color_hex(0xF2FFF3), lv_color_hex(0x80D780));
        StyleVisionCard(vision_emotion_panel_, lv_color_hex(0xFFF9E9), lv_color_hex(0xFFD16A));
        for (auto* panel : {vision_heart_panel_, vision_pressure_panel_, vision_posture_panel_, vision_emotion_panel_}) {
            lv_obj_set_style_bg_opa(panel, LV_OPA_TRANSP, 0);
            lv_obj_set_style_border_width(panel, 0, 0);
            lv_obj_set_style_radius(panel, 0, 0);
        }
        lv_obj_add_flag(vision_heart_icon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_pressure_icon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_cartoon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_icon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
        for (auto* label : {vision_heart_label_, vision_pressure_label_, vision_posture_label_, vision_emotion_label_}) {
            lv_obj_set_width(label, side_w - 8);
            lv_obj_set_height(label, LV_SIZE_CONTENT);
            lv_obj_align(label, LV_ALIGN_CENTER, 0, 13);
            lv_obj_clear_flag(label, LV_OBJ_FLAG_HIDDEN);
            lv_label_set_long_mode(label, LV_LABEL_LONG_WRAP);
            lv_obj_set_style_text_align(label, LV_TEXT_ALIGN_CENTER, 0);
            lv_obj_set_style_text_font(label, static_cast<LvglTheme*>(current_theme_)->text_font()->font(), 0);
            lv_obj_set_style_text_color(label, lv_color_hex(0x243447), 0);
            lv_obj_set_style_transform_zoom(label, 280, 0);
            lv_obj_set_style_text_line_space(label, 0, 0);
        }
        lv_obj_set_style_text_color(vision_heart_label_, lv_color_hex(0xE53935), 0);
        lv_obj_set_style_text_color(vision_pressure_label_, lv_color_hex(0x1267D6), 0);
        lv_obj_set_style_text_color(vision_posture_label_, lv_color_hex(0x1E9E49), 0);
        auto setup_icon = [&](lv_obj_t* icon, const char* glyph, lv_color_t color) {
            lv_obj_set_width(icon, side_w);
            lv_obj_set_height(icon, LV_SIZE_CONTENT);
            lv_obj_set_style_text_font(icon, static_cast<LvglTheme*>(current_theme_)->large_icon_font()->font(), 0);
            lv_obj_set_style_text_color(icon, color, 0);
            lv_label_set_text(icon, glyph);
            lv_obj_align(icon, LV_ALIGN_TOP_MID, 0, 8);
        };
        setup_icon(vision_heart_icon_, FONT_AWESOME_HEART, lv_color_hex(0xE53935));
        setup_icon(vision_pressure_icon_, FONT_AWESOME_CALCULATOR, lv_color_hex(0x4C8DFF));
        lv_obj_set_style_text_font(vision_pressure_icon_, static_cast<LvglTheme*>(current_theme_)->large_icon_font()->font(), 0);
        lv_obj_set_size(vision_posture_icon_, 52, 56);
        lv_obj_align(vision_posture_icon_, LV_ALIGN_TOP_MID, 0, 5);
        SetVisionPostureGraphic(vision_posture_icon_, false);
        lv_obj_set_size(vision_emotion_image_, 58, 58);
        lv_obj_align(vision_emotion_image_, LV_ALIGN_CENTER, 0, 13);
        lv_image_set_inner_align(vision_emotion_image_, LV_IMAGE_ALIGN_CONTAIN);
        lv_obj_add_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_t* title = l(p(side_w, 0, cw, th));
        StyleVisionCard(lv_obj_get_parent(title), lv_color_hex(0xEAF4FF), lv_color_hex(0x3F8FE8), 7);
        CreateVisionBear(lv_obj_get_parent(title));
        // Leave a clear gap after the bear so the title never overlaps it.
        lv_obj_set_width(title, cw - 36);
        lv_label_set_long_mode(title, LV_LABEL_LONG_CLIP);
        lv_obj_set_style_text_color(title, lv_color_hex(0x24507A), 0);
        lv_obj_set_style_transform_zoom(title, 180, 0);
        lv_label_set_text(title, "坐姿健康检测系统");
        lv_obj_align(title, LV_ALIGN_RIGHT_MID, -3, 0);
        lv_obj_add_flag(lv_obj_get_parent(title), LV_OBJ_FLAG_HIDDEN);
        lv_obj_t* video = p(side_w, th, cw, caption_y - th);
        lv_obj_set_style_bg_opa(video, LV_OPA_TRANSP, 0);
        lv_obj_set_style_border_width(video, 0, 0);
        lv_obj_set_parent(preview_image_, vision_dashboard_);
        lv_obj_set_size(preview_image_, preview_w, preview_h);
        lv_obj_set_align(preview_image_, LV_ALIGN_DEFAULT);
        lv_obj_set_pos(preview_image_, preview_x, preview_y);
        lv_image_set_inner_align(preview_image_, LV_IMAGE_ALIGN_COVER);
        lv_obj_set_style_radius(preview_image_, 7, 0);
    }
    if (vision_fps_label_ == nullptr) {
        vision_fps_label_ = lv_label_create(vision_dashboard_);
        lv_obj_set_size(vision_fps_label_, 66, 18);
        lv_obj_set_pos(vision_fps_label_, preview_x + 3, preview_y + 3);
        lv_label_set_long_mode(vision_fps_label_, LV_LABEL_LONG_CLIP);
        lv_label_set_text(vision_fps_label_, "FPS --");
        lv_obj_set_style_text_font(vision_fps_label_, &lv_font_montserrat_14, 0);
        lv_obj_set_style_text_align(vision_fps_label_, LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_set_style_text_color(vision_fps_label_, lv_color_white(), 0);
        lv_obj_set_style_bg_color(vision_fps_label_, lv_color_hex(0x17324D), 0);
        lv_obj_set_style_bg_opa(vision_fps_label_, LV_OPA_70, 0);
        lv_obj_set_style_border_width(vision_fps_label_, 0, 0);
        lv_obj_set_style_radius(vision_fps_label_, 3, 0);
        lv_obj_set_style_pad_hor(vision_fps_label_, 2, 0);
        lv_obj_set_style_pad_ver(vision_fps_label_, 1, 0);
        vision_fps_window_start_ms_ = 0;
        vision_fps_frame_count_ = 0;
    }
    if (vision_caption_ == nullptr) {
        vision_caption_ = lv_obj_create(vision_dashboard_);
        lv_obj_remove_style_all(vision_caption_);
        lv_obj_set_size(vision_caption_, w - 12, caption_h);
        lv_obj_set_pos(vision_caption_, 6, caption_y);
        lv_obj_clear_flag(vision_caption_, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_clear_flag(vision_caption_, LV_OBJ_FLAG_OVERFLOW_VISIBLE);
    }
    lv_obj_remove_flag(vision_dashboard_, LV_OBJ_FLAG_HIDDEN);
    // Re-apply the transcript-channel layout on every entry. The label is
    // reparented to content_ when leaving vision mode, while the dashboard
    // object itself is intentionally retained for reuse.
    if (chat_message_label_ != nullptr) {
        if (lv_obj_get_parent(chat_message_label_) != vision_caption_) {
            lv_obj_set_parent(chat_message_label_, vision_caption_);
        }
        if (entering_vision) {
            lv_obj_set_style_bg_opa(chat_message_label_, LV_OPA_TRANSP, 0);
            lv_obj_set_style_pad_all(chat_message_label_, 0, 0);
            lv_obj_set_style_text_color(chat_message_label_, lv_color_hex(0x24507A), 0);
            StartVisionTranscriptScroll(chat_message_label_, w - 12, 0, caption_h);
        }
        lv_obj_remove_flag(chat_message_label_, LV_OBJ_FLAG_HIDDEN);
    }
    // Keep the artwork's own transcript slot visible; it is already sized and
    // colored for this dashboard.
    PrepareVisionFrame(caption_y, false);
    char buf[96];
    if (result.heart_rate.available) {
        snprintf(buf, sizeof(buf), "%.0f", result.heart_rate.bpm);
        SetLabelTextIfChanged(vision_heart_label_, buf);
    } else {
        SetLabelTextIfChanged(vision_heart_label_, "");
    }
    if (result.blood_pressure.ready) {
        snprintf(buf, sizeof(buf), "%.0f\n%.0f",
                 result.blood_pressure.sbp_mmHg, result.blood_pressure.dbp_mmHg);
        SetLabelTextIfChanged(vision_pressure_label_, buf);
    } else {
        SetLabelTextIfChanged(vision_pressure_label_, "");
    }
    if (result.posture.available) {
        bool posture_bad = result.posture.state == "bad_posture";
        lv_obj_remove_flag(vision_posture_label_, LV_OBJ_FLAG_HIDDEN);
        const char* posture_text = posture_bad ? "不端正" : "端正";
        if (std::strcmp(lv_label_get_text(vision_posture_label_), posture_text) != 0) {
            lv_label_set_text(vision_posture_label_, posture_text);
            lv_obj_set_style_text_color(vision_posture_label_,
                                       posture_bad ? lv_color_hex(0xD93025)
                                                   : lv_color_hex(0x1E9E49), 0);
        }
        lv_obj_add_flag(vision_posture_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_icon_, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(vision_posture_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_label_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_icon_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_posture_cartoon_, LV_OBJ_FLAG_HIDDEN);
    }
    if (result.emotion.available) {
        std::string emotion_asset_name = "vision_emotion_";
        emotion_asset_name += EmotionAssetKey(result.emotion.label);
        emotion_asset_name += ".png";
        if (vision_emotion_asset_name_ != emotion_asset_name) {
            auto emotion_image = LoadVisionAsset(emotion_asset_name.c_str());
            if (emotion_image != nullptr) {
                lv_image_set_src(vision_emotion_image_, emotion_image->image_dsc());
                vision_emotion_asset_image_ = std::move(emotion_image);
                vision_emotion_asset_name_ = emotion_asset_name;
            }
        }
        lv_obj_add_flag(vision_emotion_label_, LV_OBJ_FLAG_HIDDEN);
        if (vision_emotion_asset_image_ && vision_emotion_asset_name_ == emotion_asset_name) {
            lv_obj_remove_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
        } else {
            lv_obj_add_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
        }
    } else {
        lv_obj_add_flag(vision_emotion_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(vision_emotion_label_, LV_OBJ_FLAG_HIDDEN);
    }
    // Keep visual cards stable. Health warnings are spoken by main's fixed
    // voice-alert path; the dashboard must not flash or change layers.
    if (lv_obj_get_parent(preview_image_) != vision_dashboard_) {
        lv_obj_set_parent(preview_image_, vision_dashboard_);
        lv_obj_set_align(preview_image_, LV_ALIGN_DEFAULT);
        lv_obj_set_size(preview_image_, preview_w, preview_h);
        lv_obj_set_pos(preview_image_, preview_x, preview_y);
        lv_image_set_inner_align(preview_image_, LV_IMAGE_ALIGN_COVER);
        lv_obj_set_style_radius(preview_image_, 7, 0);
        lv_obj_move_to_index(preview_image_, 1);
    }
    PublishVisionFrame(std::move(image));
    lv_obj_add_flag(emoji_box_, LV_OBJ_FLAG_HIDDEN);
    lv_obj_remove_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);
}

void LcdDisplay::SetupUI() {
    DisplayLockGuard lock(this);
    LvglTheme* lvgl_theme = static_cast<LvglTheme*>(current_theme_);
    auto text_font = lvgl_theme->text_font()->font();
    auto icon_font = lvgl_theme->icon_font()->font();
    auto large_icon_font = lvgl_theme->large_icon_font()->font();

    auto screen = lv_screen_active();
    lv_obj_set_style_text_font(screen, text_font, 0);
    lv_obj_set_style_text_color(screen, lvgl_theme->text_color(), 0);
    lv_obj_set_style_bg_color(screen, lvgl_theme->background_color(), 0);

    /* Container */
    container_ = lv_obj_create(screen);
    lv_obj_set_size(container_, LV_HOR_RES, LV_VER_RES);
    lv_obj_set_style_radius(container_, 0, 0);
    lv_obj_set_flex_flow(container_, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_style_pad_all(container_, 0, 0);
    lv_obj_set_style_border_width(container_, 0, 0);
    lv_obj_set_style_pad_row(container_, 0, 0);
    lv_obj_set_style_bg_color(container_, lvgl_theme->background_color(), 0);
    lv_obj_set_style_border_color(container_, lvgl_theme->border_color(), 0);

    /* Status bar */
    status_bar_ = lv_obj_create(container_);
    lv_obj_set_size(status_bar_, LV_HOR_RES, LV_SIZE_CONTENT);
    lv_obj_set_style_radius(status_bar_, 0, 0);
    lv_obj_set_style_bg_color(status_bar_, lvgl_theme->background_color(), 0);
    lv_obj_set_style_text_color(status_bar_, lvgl_theme->text_color(), 0);
    lv_obj_set_flex_flow(status_bar_, LV_FLEX_FLOW_ROW);
    lv_obj_set_style_pad_top(status_bar_, lvgl_theme->spacing(2), 0);
    lv_obj_set_style_pad_bottom(status_bar_, lvgl_theme->spacing(2), 0);
    lv_obj_set_style_pad_left(status_bar_, lvgl_theme->spacing(4), 0);
    lv_obj_set_style_pad_right(status_bar_, lvgl_theme->spacing(4), 0);
    lv_obj_set_style_border_width(status_bar_, 0, 0);
    lv_obj_set_style_pad_column(status_bar_, 0, 0);
    
    /* Content */
    content_ = lv_obj_create(container_);
    lv_obj_set_scrollbar_mode(content_, LV_SCROLLBAR_MODE_OFF);
    lv_obj_set_style_radius(content_, 0, 0);
    lv_obj_set_width(content_, LV_HOR_RES);
    lv_obj_set_flex_grow(content_, 1);
    lv_obj_set_style_pad_all(content_, 0, 0);
    lv_obj_set_style_border_width(content_, 0, 0);
    lv_obj_set_style_bg_color(content_, lvgl_theme->chat_background_color(), 0);

    lv_obj_set_flex_flow(content_, LV_FLEX_FLOW_COLUMN); // 垂直布局（从上到下）
    lv_obj_set_flex_align(content_, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_SPACE_EVENLY); // 子对象居中对齐，等距分布

    emoji_box_ = lv_obj_create(content_);
    lv_obj_set_size(emoji_box_, LV_SIZE_CONTENT, LV_SIZE_CONTENT);
    lv_obj_set_style_bg_opa(emoji_box_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_pad_all(emoji_box_, 0, 0);
    lv_obj_set_style_border_width(emoji_box_, 0, 0);

    emoji_label_ = lv_label_create(emoji_box_);
    lv_obj_set_style_text_font(emoji_label_, large_icon_font, 0);
    lv_obj_set_style_text_color(emoji_label_, lvgl_theme->text_color(), 0);
    lv_label_set_text(emoji_label_, FONT_AWESOME_MICROCHIP_AI);

    emoji_image_ = lv_img_create(emoji_box_);
    lv_obj_center(emoji_image_);
    lv_obj_add_flag(emoji_image_, LV_OBJ_FLAG_HIDDEN);

    preview_image_ = lv_image_create(content_);
    lv_obj_set_size(preview_image_, width_ / 2, height_ / 2);
    lv_obj_align(preview_image_, LV_ALIGN_CENTER, 0, 0);
    lv_obj_add_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);

    chat_message_label_ = lv_label_create(content_);
    lv_label_set_text(chat_message_label_, "");
    lv_obj_set_width(chat_message_label_, width_ * 0.9); // 限制宽度为屏幕宽度的 90%
    lv_label_set_long_mode(chat_message_label_, LV_LABEL_LONG_WRAP); // 设置为自动换行模式
    lv_obj_set_style_text_align(chat_message_label_, LV_TEXT_ALIGN_CENTER, 0); // 设置文本居中对齐
    lv_obj_set_style_text_color(chat_message_label_, lvgl_theme->text_color(), 0);

    /* Status bar */
    network_label_ = lv_label_create(status_bar_);
    lv_label_set_text(network_label_, "");
    lv_obj_set_style_text_font(network_label_, icon_font, 0);
    lv_obj_set_style_text_color(network_label_, lvgl_theme->text_color(), 0);

    notification_label_ = lv_label_create(status_bar_);
    lv_obj_set_flex_grow(notification_label_, 1);
    lv_obj_set_style_text_align(notification_label_, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(notification_label_, lvgl_theme->text_color(), 0);
    lv_label_set_text(notification_label_, "");
    lv_obj_add_flag(notification_label_, LV_OBJ_FLAG_HIDDEN);

    status_label_ = lv_label_create(status_bar_);
    lv_obj_set_flex_grow(status_label_, 1);
    lv_label_set_long_mode(status_label_, LV_LABEL_LONG_SCROLL_CIRCULAR);
    lv_obj_set_style_text_align(status_label_, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(status_label_, lvgl_theme->text_color(), 0);
    lv_label_set_text(status_label_, Lang::Strings::INITIALIZING);

    mute_label_ = lv_label_create(status_bar_);
    lv_label_set_text(mute_label_, "");
    lv_obj_set_style_text_font(mute_label_, icon_font, 0);
    lv_obj_set_style_text_color(mute_label_, lvgl_theme->text_color(), 0);

    battery_label_ = lv_label_create(status_bar_);
    lv_label_set_text(battery_label_, "");
    lv_obj_set_style_text_font(battery_label_, icon_font, 0);
    lv_obj_set_style_text_color(battery_label_, lvgl_theme->text_color(), 0);

    low_battery_popup_ = lv_obj_create(screen);
    lv_obj_set_scrollbar_mode(low_battery_popup_, LV_SCROLLBAR_MODE_OFF);
    lv_obj_set_size(low_battery_popup_, LV_HOR_RES * 0.9, text_font->line_height * 2);
    lv_obj_align(low_battery_popup_, LV_ALIGN_BOTTOM_MID, 0, -lvgl_theme->spacing(4));
    lv_obj_set_style_bg_color(low_battery_popup_, lvgl_theme->low_battery_color(), 0);
    lv_obj_set_style_radius(low_battery_popup_, lvgl_theme->spacing(4), 0);
    
    low_battery_label_ = lv_label_create(low_battery_popup_);
    lv_label_set_text(low_battery_label_, Lang::Strings::BATTERY_NEED_CHARGE);
    lv_obj_set_style_text_color(low_battery_label_, lv_color_white(), 0);
    lv_obj_center(low_battery_label_);
    lv_obj_add_flag(low_battery_popup_, LV_OBJ_FLAG_HIDDEN);
}

void LcdDisplay::SetPreviewImage(std::unique_ptr<LvglImage> image) {
    DisplayLockGuard lock(this);
    if (preview_image_ == nullptr) {
        ESP_LOGE(TAG, "Preview image is not initialized");
        return;
    }

    if (image != nullptr && vision_dashboard_ != nullptr &&
        !lv_obj_has_flag(vision_dashboard_, LV_OBJ_FLAG_HIDDEN)) {
        ESP_LOGW(TAG, "Ignoring ordinary image preview while vision dashboard is active");
        return;
    }

    if (image == nullptr) {
        esp_timer_stop(preview_timer_);
        if (status_bar_ != nullptr) {
            lv_obj_remove_flag(status_bar_, LV_OBJ_FLAG_HIDDEN);
        }
        if (vision_dashboard_ != nullptr) {
            lv_obj_add_flag(vision_dashboard_, LV_OBJ_FLAG_HIDDEN);
        }
        lv_obj_set_parent(preview_image_, content_);
        if (chat_message_label_ != nullptr) {
            lv_anim_delete(chat_message_label_, SetVisionTranscriptX);
            lv_obj_set_parent(chat_message_label_, content_);
            lv_obj_set_size(chat_message_label_, width_ * 9 / 10, LV_SIZE_CONTENT);
            lv_obj_set_style_bg_opa(chat_message_label_, LV_OPA_TRANSP, 0);
            lv_obj_set_style_pad_all(chat_message_label_, 0, 0);
            lv_obj_set_style_transform_zoom(chat_message_label_, 256, 0);
            lv_label_set_long_mode(chat_message_label_, LV_LABEL_LONG_WRAP);
            lv_obj_set_style_text_align(chat_message_label_, LV_TEXT_ALIGN_CENTER, 0);
            lv_obj_remove_flag(chat_message_label_, LV_OBJ_FLAG_HIDDEN);
        }
        lv_obj_set_size(preview_image_, width_ / 2, height_ / 2);
        lv_obj_align(preview_image_, LV_ALIGN_CENTER, 0, 0);
        lv_obj_set_flex_flow(content_, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_grow(content_, 1);
        lv_obj_set_size(content_, LV_HOR_RES, LV_SIZE_CONTENT);
        lv_obj_remove_flag(emoji_box_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_remove_flag(chat_message_label_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);
        lv_image_set_src(preview_image_, nullptr);
        if (preview_image_cached_) {
            vision_frame_history_.push_back(std::move(preview_image_cached_));
            while (vision_frame_history_.size() > 8) vision_frame_history_.pop_front();
        }
        if (gif_controller_) {
            gif_controller_->Start();
        }
        return;
    }

    // Keep the old descriptor alive until lv_image_set_src has detached it.
    if (preview_image_cached_) {
        vision_frame_history_.push_back(std::move(preview_image_cached_));
    }
    preview_image_cached_ = std::move(image);
    if (vision_dashboard_ != nullptr) {
        lv_obj_add_flag(vision_dashboard_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_set_parent(preview_image_, content_);
        if (chat_message_label_ != nullptr) {
            lv_anim_delete(chat_message_label_, SetVisionTranscriptX);
            lv_obj_set_parent(chat_message_label_, content_);
            lv_obj_set_size(chat_message_label_, width_ * 9 / 10, LV_SIZE_CONTENT);
            lv_obj_set_style_bg_opa(chat_message_label_, LV_OPA_TRANSP, 0);
            lv_obj_set_style_pad_all(chat_message_label_, 0, 0);
            lv_obj_set_style_transform_zoom(chat_message_label_, 256, 0);
            lv_label_set_long_mode(chat_message_label_, LV_LABEL_LONG_WRAP);
            lv_obj_set_style_text_align(chat_message_label_, LV_TEXT_ALIGN_CENTER, 0);
        }
        if (status_bar_ != nullptr) {
            lv_obj_remove_flag(status_bar_, LV_OBJ_FLAG_HIDDEN);
        }
        lv_obj_set_flex_flow(content_, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_grow(content_, 1);
        lv_obj_set_size(content_, LV_HOR_RES, LV_SIZE_CONTENT);
    }
    auto img_dsc = preview_image_cached_->image_dsc();
    // 设置图片源并显示预览图片
    lv_image_set_src(preview_image_, img_dsc);
    while (vision_frame_history_.size() > 8) vision_frame_history_.pop_front();
    if (img_dsc->header.w > 0 && img_dsc->header.h > 0) {
        // zoom factor 0.5
        lv_image_set_scale(preview_image_, 128 * width_ / img_dsc->header.w);
    }

    // Hide emoji_box_
    if (gif_controller_) {
        gif_controller_->Stop();
    }
    lv_obj_add_flag(emoji_box_, LV_OBJ_FLAG_HIDDEN);
    lv_obj_remove_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);
    esp_timer_stop(preview_timer_);
    ESP_ERROR_CHECK(esp_timer_start_once(preview_timer_, PREVIEW_IMAGE_DURATION_MS * 1000));
}

void LcdDisplay::SetChatMessage(const char* role, const char* content) {
    DisplayLockGuard lock(this);
    if (chat_message_label_ == nullptr) {
        return;
    }
    if (vision_dashboard_ != nullptr &&
        lv_obj_get_parent(chat_message_label_) == vision_caption_ &&
        !lv_obj_has_flag(vision_dashboard_, LV_OBJ_FLAG_HIDDEN)) {
        const auto line = VisionTranscriptSingleLine(content);
        // Application state transitions frequently send an empty system
        // message. In monitoring mode it must not erase the latest transcript.
        if (line.empty()) return;
        if (line == lv_label_get_text(chat_message_label_)) {
            if (lv_anim_get(chat_message_label_, SetVisionTranscriptX) == nullptr) {
                const lv_coord_t caption_h = std::max<lv_coord_t>(18, LV_VER_RES * 21 / 240);
                StartVisionTranscriptScroll(chat_message_label_, LV_HOR_RES - 12, 0, caption_h);
            }
            return;
        }
        lv_label_set_text(chat_message_label_, line.c_str());
        const lv_coord_t caption_h = std::max<lv_coord_t>(18, LV_VER_RES * 21 / 240);
        StartVisionTranscriptScroll(chat_message_label_, LV_HOR_RES - 12, 0, caption_h);
    } else {
        lv_label_set_text(chat_message_label_, content != nullptr ? content : "");
    }
}
#endif

void LcdDisplay::SetEmotion(const char* emotion) {
    // Stop any running GIF animation
    if (gif_controller_) {
        DisplayLockGuard lock(this);
        gif_controller_->Stop();
        gif_controller_.reset();
    }
    
    if (emoji_image_ == nullptr) {
        return;
    }

    auto emoji_collection = static_cast<LvglTheme*>(current_theme_)->emoji_collection();
    auto image = emoji_collection != nullptr ? emoji_collection->GetEmojiImage(emotion) : nullptr;
    if (image == nullptr) {
        const char* utf8 = font_awesome_get_utf8(emotion);
        if (utf8 != nullptr && emoji_label_ != nullptr) {
            DisplayLockGuard lock(this);
            lv_label_set_text(emoji_label_, utf8);
            lv_obj_add_flag(emoji_image_, LV_OBJ_FLAG_HIDDEN);
            lv_obj_remove_flag(emoji_label_, LV_OBJ_FLAG_HIDDEN);
        }
        return;
    }

    DisplayLockGuard lock(this);
    if (image->IsGif()) {
        // Create new GIF controller
        gif_controller_ = std::make_unique<LvglGif>(image->image_dsc());
        
        if (gif_controller_->IsLoaded()) {
            // Set up frame update callback
            gif_controller_->SetFrameCallback([this]() {
                lv_image_set_src(emoji_image_, gif_controller_->image_dsc());
            });
            
            // Set initial frame and start animation
            lv_image_set_src(emoji_image_, gif_controller_->image_dsc());
            gif_controller_->Start();
            
            // Show GIF, hide others
            lv_obj_add_flag(emoji_label_, LV_OBJ_FLAG_HIDDEN);
            lv_obj_remove_flag(emoji_image_, LV_OBJ_FLAG_HIDDEN);
        } else {
            ESP_LOGE(TAG, "Failed to load GIF for emotion: %s", emotion);
            gif_controller_.reset();
        }
    } else {
        lv_image_set_src(emoji_image_, image->image_dsc());
        lv_obj_add_flag(emoji_label_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_remove_flag(emoji_image_, LV_OBJ_FLAG_HIDDEN);
    }

#if CONFIG_USE_WECHAT_MESSAGE_STYLE
    // Wechat message style中，如果emotion是neutral，则不显示
    uint32_t child_count = lv_obj_get_child_cnt(content_);
    if (strcmp(emotion, "neutral") == 0 && child_count > 0) {
        // Stop GIF animation if running
        if (gif_controller_) {
            gif_controller_->Stop();
            gif_controller_.reset();
        }
        
        lv_obj_add_flag(emoji_image_, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(emoji_label_, LV_OBJ_FLAG_HIDDEN);
    }
#endif
}

void LcdDisplay::SetTheme(Theme* theme) {
    DisplayLockGuard lock(this);
    
    auto lvgl_theme = static_cast<LvglTheme*>(theme);
    
    // Get the active screen
    lv_obj_t* screen = lv_screen_active();

    // Set font
    auto text_font = lvgl_theme->text_font()->font();
    auto icon_font = lvgl_theme->icon_font()->font();
    auto large_icon_font = lvgl_theme->large_icon_font()->font();

    if (text_font->line_height >= 40) {
        lv_obj_set_style_text_font(mute_label_, large_icon_font, 0);
        lv_obj_set_style_text_font(battery_label_, large_icon_font, 0);
        lv_obj_set_style_text_font(network_label_, large_icon_font, 0);
    } else {
        lv_obj_set_style_text_font(mute_label_, icon_font, 0);
        lv_obj_set_style_text_font(battery_label_, icon_font, 0);
        lv_obj_set_style_text_font(network_label_, icon_font, 0);
    }

    // Set parent text color
    lv_obj_set_style_text_font(screen, text_font, 0);
    lv_obj_set_style_text_color(screen, lvgl_theme->text_color(), 0);

    // Set background image
    if (lvgl_theme->background_image() != nullptr) {
        lv_obj_set_style_bg_image_src(container_, lvgl_theme->background_image()->image_dsc(), 0);
    } else {
        lv_obj_set_style_bg_image_src(container_, nullptr, 0);
        lv_obj_set_style_bg_color(container_, lvgl_theme->background_color(), 0);
    }
    
    // Update status bar background color with 50% opacity
    lv_obj_set_style_bg_opa(status_bar_, LV_OPA_50, 0);
    lv_obj_set_style_bg_color(status_bar_, lvgl_theme->background_color(), 0);
    
    // Update status bar elements
    lv_obj_set_style_text_color(network_label_, lvgl_theme->text_color(), 0);
    lv_obj_set_style_text_color(status_label_, lvgl_theme->text_color(), 0);
    lv_obj_set_style_text_color(notification_label_, lvgl_theme->text_color(), 0);
    lv_obj_set_style_text_color(mute_label_, lvgl_theme->text_color(), 0);
    lv_obj_set_style_text_color(battery_label_, lvgl_theme->text_color(), 0);
    lv_obj_set_style_text_color(emoji_label_, lvgl_theme->text_color(), 0);

    // Set content background opacity
    lv_obj_set_style_bg_opa(content_, LV_OPA_TRANSP, 0);

    // If we have the chat message style, update all message bubbles
#if CONFIG_USE_WECHAT_MESSAGE_STYLE
    // Iterate through all children of content (message containers or bubbles)
    uint32_t child_count = lv_obj_get_child_cnt(content_);
    for (uint32_t i = 0; i < child_count; i++) {
        lv_obj_t* obj = lv_obj_get_child(content_, i);
        if (obj == nullptr) continue;
        
        lv_obj_t* bubble = nullptr;
        
        // 检查这个对象是容器还是气泡
        // 如果是容器（用户或系统消息），则获取其子对象作为气泡
        // 如果是气泡（助手消息），则直接使用
        if (lv_obj_get_child_cnt(obj) > 0) {
            // 可能是容器，检查它是否为用户或系统消息容器
            // 用户和系统消息容器是透明的
            lv_opa_t bg_opa = lv_obj_get_style_bg_opa(obj, 0);
            if (bg_opa == LV_OPA_TRANSP) {
                // 这是用户或系统消息的容器
                bubble = lv_obj_get_child(obj, 0);
            } else {
                // 这可能是助手消息的气泡自身
                bubble = obj;
            }
        } else {
            // 没有子元素，可能是其他UI元素，跳过
            continue;
        }
        
        if (bubble == nullptr) continue;
        
        // 使用保存的用户数据来识别气泡类型
        void* bubble_type_ptr = lv_obj_get_user_data(bubble);
        if (bubble_type_ptr != nullptr) {
            const char* bubble_type = static_cast<const char*>(bubble_type_ptr);
            
            // 根据气泡类型应用正确的颜色
            if (strcmp(bubble_type, "user") == 0) {
                lv_obj_set_style_bg_color(bubble, lvgl_theme->user_bubble_color(), 0);
            } else if (strcmp(bubble_type, "assistant") == 0) {
                lv_obj_set_style_bg_color(bubble, lvgl_theme->assistant_bubble_color(), 0); 
            } else if (strcmp(bubble_type, "system") == 0) {
                lv_obj_set_style_bg_color(bubble, lvgl_theme->system_bubble_color(), 0);
            } else if (strcmp(bubble_type, "image") == 0) {
                lv_obj_set_style_bg_color(bubble, lvgl_theme->system_bubble_color(), 0);
            }
            
            // Update border color
            lv_obj_set_style_border_color(bubble, lvgl_theme->border_color(), 0);
            
            // Update text color for the message
            if (lv_obj_get_child_cnt(bubble) > 0) {
                lv_obj_t* text = lv_obj_get_child(bubble, 0);
                if (text != nullptr) {
                    // 根据气泡类型设置文本颜色
                    if (strcmp(bubble_type, "system") == 0) {
                        lv_obj_set_style_text_color(text, lvgl_theme->system_text_color(), 0);
                    } else {
                        lv_obj_set_style_text_color(text, lvgl_theme->text_color(), 0);
                    }
                }
            }
        } else {
            ESP_LOGW(TAG, "child[%lu] Bubble type is not found", i);
        }
    }
#else
    // Simple UI mode - just update the main chat message
    if (chat_message_label_ != nullptr) {
        lv_obj_set_style_text_color(chat_message_label_, lvgl_theme->text_color(), 0);
    }
    
    if (emoji_label_ != nullptr) {
        lv_obj_set_style_text_color(emoji_label_, lvgl_theme->text_color(), 0);
    }
#endif
    
    // Update low battery popup
    lv_obj_set_style_bg_color(low_battery_popup_, lvgl_theme->low_battery_color(), 0);

    // No errors occurred. Save theme to settings
    Display::SetTheme(lvgl_theme);
}
