#pragma once

#include <unordered_set>

namespace ui
{
	struct UserControl
	{
		// Depends on user controled mesh_id
		int apply_ext_force_mesh_id { 0 };
		bool apply_ext_force{ false };
		int ext_forced_vertex_idx{ 0 };
		int mouse_x{ 0 };
		int mouse_y{ 0 };

		// User LMC a vertex
		int selected_vertex_idx{ 0 };

		// This does not depend on mesh_id. It depends on the id of the current selected mesh.
		int cur_sel_mesh_id { -1 };
		bool toggle_vertex_fix{ false };
		std::unordered_set<int> toggle_fixed_vertex_idxs;
	};
}