#include <instancing/instancing.h>

#include <meshgen/mesh_generator.h>

#include <primitive/floor.h>
#include <primitive/block.h>
#include <primitive/sphere.h>

namespace instancing {
    void Instantiator::instance_floor()
    {
		obj_manager.add_rigid_collider(std::make_unique<primitive::Floor>(-1));
    }

    void Instantiator::instance_bending_skirt()
    {
        auto [V, F] = meshgen::generate_hemisphere(1);
		int id = obj_manager.add_model(V, F);
		pd::DeformableMesh& model = models.at(id);

		model.set_edge_strain_constraints(100.f);

		model.set_bending_constraints(5e-7f * 13);

		std::unordered_set<int> toggle_vertices;
		for (int i = 1; i <= 324; i += 17)
		{
			toggle_vertices.insert(i);
		}
		model.toggle_vertices_fixed(toggle_vertices, 100.f);
    }

    void Instantiator::instance_cloth()
    {

    }

    void Instantiator::instance_box()
    {

    }

    void Instantiator::instance_armadillo()
    {

    }
}