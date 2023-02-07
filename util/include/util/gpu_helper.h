#pragma once

namespace util
{
	// returns the best device index (if only one device is available, always return 0)
	// return-by-reference the total number of available device 
	int select_best_device(int& n_devs);

	// test if the device is available in the application
	void test_device(int dev_id);
}
