#include <instancing/instantiator.h>

#include <meshgen/mesh_generator.h>

#include <primitive/floor.h>
#include <primitive/block.h>
#include <primitive/sphere.h>

namespace instancing {
    void Instantiator::instance_floor()
    {
        obj_manager.add_rigid_collider(std::make_unique<primitive::Floor>(-1));
    }

    void Instantiator::instance_bending_hemisphere()
    {
        auto [V, F] = meshgen::generate_hemisphere(1);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = models.at(id);

        model.set_edge_strain_constraints(100.f);

        // model.set_bending_constraints(5e-7f * 13);

        // small curvature
        model.set_bending_constraints(5e-7f * 9);

        std::unordered_set<int> toggle_vertices;
        for (int i = 1; i <= 324; i += 17)
        {
            toggle_vertices.insert(i);
        }
        model.toggle_vertices_fixed(toggle_vertices, 100.f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_cloth()
    {
        auto [V, F] = meshgen::generate_cloth(20, 20);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = models.at(id);
        model.set_edge_strain_constraints(100.f);

        model.set_bending_constraints(5e-7f);

        // add positional constraint
        model.toggle_vertices_fixed({ 0, 420 }, 100.f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_cylinder()
    {
        const int usub = 20;
        const int vsub = 16;
        auto [V, F] = meshgen::generate_cylinder(0.5f, 3, usub, vsub);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = models.at(id);
        
        model.set_edge_strain_constraints(20.f);

        model.set_bending_constraints(5e-7f * 10);

        std::unordered_set<int> toggle_vertices;
        for (int i = 2; i < usub * (vsub + 1) - vsub + 2; i += vsub + 1)
        {
            toggle_vertices.insert(i);
        }
        for (int i = vsub; i < usub * (vsub + 1); i += vsub + 1)
        {
            toggle_vertices.insert(i);
        }
        model.toggle_vertices_fixed(toggle_vertices, 10.f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_bar()
    {
        // static int w = 2;
        // static int h = 3;
        // static int d = 3;
        // auto [V, T, boundary_facets] = meshgen::generate_bar(w, h, d);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_obj_model(const std::string& file_path)
    {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::read_triangle_mesh(file_path, V, F);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = models.at(id);

        model.set_edge_strain_constraints(100.f);

        model.set_bending_constraints(5e-7f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_armadillo()
    {
        instance_obj_model("/home/xinghai/codes/pd-with-gpu/assets/meshes/armadillo.obj");
    }

    void Instantiator::instance_bunny()
    {

    }

    void Instantiator::instance_cone()
    {
        const int usub = 20;
        const int vsub = 16;
        auto [V, F] = meshgen::generate_cylinder(0.5f, 3, usub, vsub, 4);
        int id = obj_manager.add_model(V, F);
    }

    void Instantiator::instance_test()
    {
        const int usub = 20;
        const int vsub = 16;
        auto [V, F] = meshgen::generate_cylinder(0.5f, 3, usub, vsub);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = models.at(id);

        // apply translation
        model.apply_translation(Eigen::Vector3d(1, 0, 0));
        
        model.set_edge_strain_constraints(20.f);

        // model.set_bending_constraints(5e-7f * 10);

        std::unordered_set<int> toggle_vertices;
        for (int i = 2; i < usub * (vsub + 1) - vsub + 2; i += vsub + 1)
        {
            toggle_vertices.insert(i);
        }
        for (int i = vsub; i < usub * (vsub + 1); i += vsub + 1)
        {
            toggle_vertices.insert(i);
        }
        model.toggle_vertices_fixed(toggle_vertices, 10.f);

		obj_manager.recalc_total_n_constraints();
    }
}