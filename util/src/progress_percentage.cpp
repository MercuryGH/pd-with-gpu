#include <util/progress_percentage.h>

#include <iostream>
#include <thread>
#include <cmath>

namespace util {
    void ProgressPercentage::start(int total)
    {
        progress = 0.0f;
        cur = 0;
        total_n = total;
    }

    void ProgressPercentage::increment_progress(int inc)
    {
        // update progress bar
        cur += inc;
        progress = ((float)cur / total_n);
    }

    void ProgressPercentage::set_progress_percentage(int percentage)
    {
        progress = (float)percentage / 100;
    }
}