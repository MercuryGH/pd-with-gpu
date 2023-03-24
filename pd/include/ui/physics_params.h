#pragma once

namespace ui
{
	struct PhysicsParams
	{
		bool enable_gravity{ true };
		float mass_per_vertex{ 0.001f };

		float edge_length_constraint_wi{ 100.f }; 
		float positional_constraint_wi{ 100.f };

		float external_force_val{ 1.f };
	};
}