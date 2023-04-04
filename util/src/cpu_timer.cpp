#include <util/cpu_timer.h>

namespace util
{
	void CpuTimer::start() 
	{
		start_time = std::chrono::steady_clock::now();
		running = true;
	}

	void CpuTimer::stop() 
	{
		end_time = std::chrono::steady_clock::now();
		running = false;
		//last_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	}

	double CpuTimer::elapsed_milliseconds() 
	{
		std::chrono::time_point<std::chrono::steady_clock> cur_time;
		if (running)
		{
			cur_time = std::chrono::steady_clock::now();
		}
		else
		{
			cur_time = end_time;
		}

		return std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time).count();
	}

	double CpuTimer::elapsed_seconds() 
	{
		return elapsed_milliseconds() / 1000.0;
	}
}