#pragma once

#include <util/helper_cuda.h>

namespace util
{
	// returns the best device index (if only one device is available, always return 0)
	// return-by-reference the total number of available device 
	int select_best_device(int& n_devs);

	// test if the device is available in the application
	void test_device(int dev_id);

	void print_cuda_info();

	int get_n_blocks(int n_elements, int n_threads=WARP_SIZE);
}
