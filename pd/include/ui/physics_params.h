#pragma once

namespace ui
{
	struct PhysicsParams
	{
		bool enable_gravity{ true };
		float mass_per_vertex{ 1.f };

		float edge_length_constraint_wi{ 100000.f };  // maybe unnecessary
		float positional_constraint_wi{ 100'000'000.f };

		float external_force_val{ 4000.f };
	};
}