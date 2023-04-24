#include <instancing/instantiator.h>

#include <meshgen/mesh_generator.h>

#include <primitive/floor.h>
#include <primitive/block.h>
#include <primitive/sphere.h>

namespace instancing {
    void Instantiator::reset_all()
    {
        obj_manager.reset_all();
    }

    void Instantiator::instance_floor()
    {
        obj_manager.add_rigid_collider(std::make_unique<primitive::Floor>(-1));
    }

    void Instantiator::instance_bending_hemisphere()
    {
        auto [V, F] = meshgen::generate_hemisphere(1);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = obj_manager.models.at(id);

        model.set_edge_strain_constraints(100.f);

        // maintain curvature
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
        instance_cloth_not_bend();
        instance_cloth_bend();        
    }

    void Instantiator::instance_cloth_not_bend()
    {
        auto [V, F] = meshgen::generate_cloth(20, 20);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = obj_manager.models.at(id);
        // apply translation
        model.apply_translation(Eigen::Vector3d(1, 0, 0));

        model.set_edge_strain_constraints(100.f);

        // add positional constraint
        model.toggle_vertices_fixed({ 0, 420 }, 100.f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_cloth_bend()
    {
        auto [V, F] = meshgen::generate_cloth(20, 20);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = obj_manager.models.at(id);
        model.set_edge_strain_constraints(100.f);

        model.set_bending_constraints(5e-7f);

        // add positional constraint
        model.toggle_vertices_fixed({ 0, 420 }, 100.f);

		obj_manager.recalc_total_n_constraints();
    }
    
    void Instantiator::instance_4hanged_cloth()
    {
        instance_4hanged_cloth_not_bend();
        instance_4hanged_cloth_bend();        
    }

    void Instantiator::instance_4hanged_cloth_not_bend()
    {
        auto [V, F] = meshgen::generate_cloth(20, 20);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = obj_manager.models.at(id);
        model.apply_translation(Eigen::Vector3d(1, 0, 0));

        model.set_edge_strain_constraints(100.f);

        // add positional constraint
        model.toggle_vertices_fixed({ 0, 20, 420, 440 }, 100.f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_4hanged_cloth_bend()
    {
        auto [V, F] = meshgen::generate_cloth(20, 20);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = obj_manager.models.at(id);
        model.set_edge_strain_constraints(100.f);

        model.set_bending_constraints(5e-7f);

        // add positional constraint
        model.toggle_vertices_fixed({ 0, 20, 420, 440 }, 100.f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_cylinder()
    {
        instance_cylinder_not_bend();
        instance_cylinder_bend();
    }

    void Instantiator::instance_cylinder_not_bend()
    {
        constexpr int usub = 20;
        constexpr int vsub = 16;
        auto [V, F] = meshgen::generate_cylinder(0.5f, 3, usub, vsub);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = obj_manager.models.at(id);

        // apply translation
        model.apply_translation(Eigen::Vector3d(1, 0, 0));
        
        model.set_edge_strain_constraints(20.f);

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

    void Instantiator::instance_cylinder_bend()
    {
        constexpr int usub = 20;
        constexpr int vsub = 16;
        auto [V, F] = meshgen::generate_cylinder(0.5f, 3, usub, vsub);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = obj_manager.models.at(id);
        
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
        constexpr int w = 3;
        constexpr int h = 3;
        constexpr int d = 12;
        auto [V, T, boundary_facets] = meshgen::generate_bar(w, h, d);
        int id = obj_manager.add_model(V, T, boundary_facets);
        pd::DeformableMesh& model = obj_manager.models.at(id);
        
        model.set_tet_strain_constraints(1000, pd::SimVector3(0.95, 0.95, 0.95), pd::SimVector3(1.05, 1.05, 1.05));
        // model.set_tet_strain_constraints(100, pd::SimVector3(0.95, 0.95, 0.95), pd::SimVector3(1.05, 1.05, 1.05));

        std::unordered_set<int> toggle_vertices;
        for (int i = 0; i <= (w + 1) * (h + 1) * (d + 1) - (d + 1); i += d + 1)
        {
            toggle_vertices.insert(i);
        }
        model.toggle_vertices_fixed(toggle_vertices, 100.f);
        obj_manager.user_control.vertex_idxs_memory = toggle_vertices;

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_bridge()
    {
        constexpr int w = 2;
        constexpr int h = 2;
        constexpr int d = 20;
        auto [V, T, boundary_facets] = meshgen::generate_bar(w, h, d);
        int id = obj_manager.add_model(V, T, boundary_facets);
        pd::DeformableMesh& model = obj_manager.models.at(id);

        // model.set_tet_strain_constraints(400.f, Eigen::Vector3f(0.95f, 0.95f, 0.95f), Eigen::Vector3f(1.05f, 1.05f, 1.05f));

        model.set_tet_strain_constraints(1000, pd::SimVector3(0.95, 0.95, 0.95), pd::SimVector3(1.05, 1.05, 1.05));

        std::unordered_set<int> toggle_vertices;
        for (int i = 0; i <= (w + 1) * (h + 1) * (d + 1) - (d + 1); i += d + 1)
        {
            toggle_vertices.insert(i);
        }
        for (int i = d; i <= (w + 1) * (h + 1) * (d + 1) - 1; i += d + 1)
        {
            toggle_vertices.insert(i);
        }
        model.toggle_vertices_fixed(toggle_vertices, 100.f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_obj_model(const std::string& file_path)
    {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::read_triangle_mesh(file_path, V, F);
        int id = obj_manager.add_model(V, F);
        pd::DeformableMesh& model = obj_manager.models.at(id);

        model.set_edge_strain_constraints(100.f);

        model.set_bending_constraints(5e-7f);

		obj_manager.recalc_total_n_constraints();
    }

    void Instantiator::instance_armadillo()
    {
        instance_obj_model("../assets/meshes/armadillo.obj");
    }

    void Instantiator::instance_bunny()
    {

    }

    void Instantiator::instance_cone()
    {
        constexpr int usub = 20;
        constexpr int vsub = 16;
        auto [V, F] = meshgen::generate_cone(0.5f, 3, usub, vsub);
        int id = obj_manager.add_model(V, F);
    }

    void Instantiator::instance_test()
    {
        physics_params.mass_per_vertex = 10.0;

        constexpr int w = 1;
        constexpr int h = 1;
        // constexpr int d = 7;
        constexpr int d = 7;

        auto [V, T, boundary_facets] = meshgen::generate_bar(w, h, d);
        int id = obj_manager.add_model(V, T, boundary_facets);
        pd::DeformableMesh& model = obj_manager.models.at(id);

        physics_params.enable_gravity = false;
        
        model.set_tet_strain_constraints(10000000, pd::SimVector3(0.99, 0.99, 0.99), pd::SimVector3(1.01, 1.01, 1.01));

        std::unordered_set<int> toggle_vertices;
        for (int i = d; i <= (w + 1) * (h + 1) * (d + 1) - 1; i += d + 1)
        {
            toggle_vertices.insert(i);
            model.set_vertex_mass(i, 1e10);
        }
        model.toggle_vertices_fixed(toggle_vertices, 1000000000);

		obj_manager.recalc_total_n_constraints();   

        return;

        /*

        physics_params.mass_per_vertex = 10.0;
        physics_params.enable_gravity = false;

        float weight = 10000000.0;
        Eigen::Vector3f min_strain(0.990, 0.990, 0.990);
        Eigen::Vector3f max_strain(1.010, 1.010, 1.010);

        auto [V, T, boundary_facets] = meshgen::generate_bar(3, 3, 12);

        int id = obj_manager.add_model(V, T, boundary_facets);
        pd::DeformableMesh& model = obj_manager.models.at(id);
        
        model.set_tet_strain_constraints(weight, min_strain, max_strain);

        // model.toggle_vertices_fixed({ 3 }, 1000000000.0);

		obj_manager.recalc_total_n_constraints();*/
    }
}