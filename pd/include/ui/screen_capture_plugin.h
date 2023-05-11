#pragma once

#include <memory>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/png/writePNG.h>

namespace ui
{
    class ScreenCapturePlugin: public igl::opengl::glfw::ViewerPlugin
    {
    public:
        ScreenCapturePlugin() { plugin_name = "capture"; }

        bool post_draw() override
        {
            if (sequence_capturing == false && single_image_capturing == false)
            {
                return false;
            }

            if (sequence_capturing == true && viewer->core().is_animating == false)
            {
                return false;
            }

            const int width = viewer->core().viewport(2);
            const int height = viewer->core().viewport(3);

            std::unique_ptr<GLubyte[]> pixels(new GLubyte[width * height * 4]);

            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());
            
            // If Linux system, use `convert -delay 3 -loop 0 *.png output.gif` to generate gif from captured pngs
            const auto zeros_prefix_str = [](int width, std::string& num)
            {
                while (num.size() < width)
                {
                    num = "0" + num;
                }
            };

            std::string n_frame_str = std::to_string(capture_idx++);
            zeros_prefix_str(3, n_frame_str);

            std::string path = path_prefix + n_frame_str + ".png";
            // invoke a file saver thread
            std::thread{ save_png_file, path, std::move(pixels), width, height }.detach();
            if (single_image_capturing == true)
            {
                single_image_capturing = false;
            }

            return false;
        }

        void start_capture(std::string path)
        {
            path_prefix = path;
            capture_idx = 0;
            sequence_capturing = true;
        }

        void stop_capture() { sequence_capturing = false; }

        void capture_current_state(std::string path)
        {
            path_prefix = path;
            capture_idx = 0;
            single_image_capturing = true;
        }

        bool is_capturing_sequence() const { return sequence_capturing; }
        int cur_capture_frame_id() const { return capture_idx; }

    private:
        static void save_png_file(std::string path, std::unique_ptr<GLubyte[]> pixels, int width, int height)
        {
            Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
            Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
            Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
            Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> A(width, height);
            int cnt = 0;
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    R(i, j) = pixels[cnt * 4 + 0];
                    G(i, j) = pixels[cnt * 4 + 1];
                    B(i, j) = pixels[cnt * 4 + 2];
                    A(i, j) = pixels[cnt * 4 + 3];
                    cnt++;
                }
            }

			igl::png::writePNG(R, G, B, A, path.c_str());
        };

        std::string path_prefix;
        int capture_idx;
        bool sequence_capturing{ false };
        bool single_image_capturing{ false };
    };
}
