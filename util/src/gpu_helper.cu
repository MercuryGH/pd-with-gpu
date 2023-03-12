#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <util/gpu_helper.h>

namespace util
{
	int select_best_device(int& n_devs)
	{
		cudaGetDeviceCount(&n_devs);

		if (n_devs == 1)
		{
			return 0;
		}

		// select the device with maximum #core
		int best_dev_idx = 0;
		int max_n_core = -1;
		for (int i = 0; i < n_devs; i++)
		{
			cudaDeviceProp dev_prop;
			cudaGetDeviceProperties(&dev_prop, i);
			if (max_n_core < dev_prop.multiProcessorCount)
			{
				max_n_core = dev_prop.multiProcessorCount;
				best_dev_idx = i;
			}
		}
		return best_dev_idx;
	}

	void test_device(int dev_id)
	{
		cudaDeviceProp dev_prop;
		cudaGetDeviceProperties(&dev_prop, dev_id);

		// If this is a device emulator, it has major & minor = 9999
		if (dev_prop.major == 9999 && dev_prop.minor == 9999)
		{
			printf("Using an emulator device\n");
		}
		else
		{
			assert(dev_prop.warpSize == 32);
			printf("Using GPU device id = %d", dev_id);
		}
	}

	void print_cuda_info()
	{
		int n_gpu_devs;
		int best_dev_idx = select_best_device(n_gpu_devs);
		printf("Available #dev = %d\n", n_gpu_devs);
		test_device(best_dev_idx);
	}
}
