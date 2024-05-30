#pragma once

#include <vector>

#include <Eigen/Core>

#include <io/io_data.h>

#include <util/singleton.h>

namespace io
{
//     template<typename T>
//     struct V3
//     {
//         T x, y, z;
//     };
//
//     template<typename Container_Type, typename Value_Type>
//     // requires (std::convertible_to<Container_Type, std::vector<V3<Value_Type>>>)
//     requires (std::convertible_to<Container_Type, std::vector<Value_Type>>)
//     struct Constraint;

    // if use concept, no easy way to seperate impl and decl
    template<typename Container_Type>
    concept Array = requires {
        typename Container_Type::value_type; // constraint: Container_Type has a ::value_type
    };

    struct MeshIO final: public util::Singleton<MeshIO>
    {
        MeshIO(token) {}

        auto import_triangle_mesh(
            pd::MeshIDType mesh_id,
            const Array auto& positions, // vector<double3>
            const Array auto& elements   // vector<int3>
        ) -> void
        {
            auto [V, F] = import_trimesh_data(mesh_id, positions, elements);

            create_mesh_handle(mesh_id, V, F);
        }

        auto import_tetrahedron_mesh(
            pd::MeshIDType mesh_id,
            const Array auto& positions, // vector<double3>
            const Array auto& elements,   // vector<int3>
            const Array auto& tets
        ) -> void
        {
            auto [V, F, T] = import_tetmesh_data(mesh_id, positions, elements, tets);

            create_mesh_handle(mesh_id, V, F, T);
        }

        auto export_triangle_mesh(pd::MeshIDType mesh_id, Array auto& positions) -> void
        {
            export_data(mesh_id, positions);
        }

        auto get_all_mesh_ids() -> std::vector<pd::MeshIDType>;

    private:
        auto create_mesh_handle(pd::MeshIDType mesh_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) -> void;
        auto create_mesh_handle(pd::MeshIDType mesh_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& T) -> void;

        auto import_trimesh_data(
            pd::MeshIDType mesh_id,
            const Array auto& positions, // vector<double3>
            const Array auto& elements   // vector<int3>
        ) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXi>
        {
            assert(elements.size() % 3 == 0);

            const int n_verts = positions.size();
            const int n_tris = elements.size() / 3;

            Eigen::MatrixXd V(n_verts, 3);
            Eigen::MatrixXi F(n_tris, 3);

            // std::cout << n_verts << ", " << n_tris << "\n";

            for (int i = 0; i < n_verts; i++) {
                const auto& cur_pos = positions[i];

                // type unsafe
                const Eigen::RowVector3d pos = *(Eigen::RowVector3d*) &cur_pos;

                V.row(i) << pos;
            }

            // convert index
            for (int i = 0; i < n_tris; i++) {
                int tri_indices[3];
                for (int j = 0; j < 3; j++)
                {
                    // type unsafe
                    tri_indices[j] = *(int*)&elements[i * 3 + j];
                }

                // type unsafe
                const Eigen::RowVector3i tri(tri_indices[0], tri_indices[1], tri_indices[2]);

                F.row(i) << tri;
            }
            // std::cout << "--- DEBUG END ---\n";

            return {V, F};
        }

        auto import_tetmesh_data(
            pd::MeshIDType mesh_id,
            const Array auto& positions, // vector<double3>
            const Array auto& elements,  // vector<int3>
            const Array auto& tets
        ) -> std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi>
        {
            assert(elements.size() % 3 == 0);
            assert(tets.size() % 4 == 0);

            const int n_verts = positions.size();
            const int n_tris = elements.size() / 3;
            const int n_tets = tets.size() / 4;

            Eigen::MatrixXd V(n_verts, 3);
            Eigen::MatrixXi F(n_tris, 3);
            Eigen::MatrixXi T(n_tets, 4);

            // std::cout << n_verts << ", " << n_tris << "\n";

            for (int i = 0; i < n_verts; i++) {
                const auto& cur_pos = positions[i];

                // type unsafe
                const Eigen::RowVector3d pos = *(Eigen::RowVector3d*) &cur_pos;

                V.row(i) << pos;
            }

            // convert index
            for (int i = 0; i < n_tris; i++) {
                int tri_indices[3];
                for (int j = 0; j < 3; j++)
                {
                    // type unsafe
                    tri_indices[j] = *(int*)&elements[i * 3 + j];
                }

                // type unsafe
                const Eigen::RowVector3i tri(tri_indices[0], tri_indices[1], tri_indices[2]);

                F.row(i) << tri;
            }

            for (int i = 0; i < n_tets; i++) {
                int tet_indices[4];
                for (int j = 0; j < 4; j++)
                {
                    // type unsafe
                    tet_indices[j] = *(int*)&tets[i * 4 + j];
                }

                // type unsafe
                const Eigen::RowVector4i tet(tet_indices[0], tet_indices[1], tet_indices[2], tet_indices[3]);

                T.row(i) << tet;
            }

            return {V, F, T};
        }


        auto export_data(
            pd::MeshIDType mesh_id,
            Array auto& positions // vector<double3>
        ) -> void
        {
            const auto& model = IOData::instance().models.at(mesh_id);
            const int n_verts = model.positions().rows();

            positions.resize(n_verts);

            for (int i = 0; i < n_verts; i++) {
                Eigen::RowVector3d cur_pos = model.positions().row(i);

                // type unsafe
                // memcpy
                memcpy(&positions[i], &cur_pos, sizeof(cur_pos));

                // Compiler does not work
                // using ArrayElementType = decltype(positions)::value_type;
                // positions[i] = *(reinterpret_cast<ArrayElementType *>(&cur_pos));
            }
        }
    };
}
