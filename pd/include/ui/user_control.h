#pragma once

#include <unordered_set>

namespace ui
{
	struct UserControl
	{
		bool apply_ext_force{ false };
		int ext_forced_vertex_idx{ 0 };

		bool toggle_vertex_fix{ false };
		std::unordered_set<int> toggle_fixed_vertex_idxs;
	};
}