#include <filesystem>
#include <limits>
#include <fstream>
#include <iterator>

namespace util {
    std::string read_shader_file(const std::filesystem::path::value_type* shader_file)
    {
        std::ifstream ifs;

        auto ex = ifs.exceptions();
        ex |= std::ios_base::badbit | std::ios_base::failbit;
        ifs.exceptions(ex);

        ifs.open(shader_file);
        ifs.ignore(std::numeric_limits<std::streamsize>::max());
        auto size = ifs.gcount();

        if (size > 0x10000) // 64KiB sanity check:
        {
            return ""; // bad
        }

        ifs.clear();
        ifs.seekg(0, std::ios_base::beg);

        return std::string {std::istreambuf_iterator<char> {ifs}, {}};
    }
}

