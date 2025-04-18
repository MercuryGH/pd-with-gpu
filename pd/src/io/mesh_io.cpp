#include <io/mesh_io.h>

#include <igl/boundary_facets.h>

#include <primitive/primitive.h>

namespace io
{
    auto MeshIO::create_mesh_handle(pd::MeshIDType mesh_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) -> void
    {
        // create a new mesh
        IOData::instance().models.emplace(mesh_id, pd::DeformableMesh(V, F));

        // reset f_ext
        IOData::instance().f_exts[mesh_id].resizeLike(V);
        IOData::instance().f_exts.at(mesh_id).setZero();
    }

    auto MeshIO::create_mesh_handle(pd::MeshIDType mesh_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& T) -> void
    {
        // create a new mesh
		Eigen::MatrixXi boundary_facets;
    	igl::boundary_facets(T, boundary_facets);

        // F == boundary_facets
        IOData::instance().models.emplace(mesh_id, pd::DeformableMesh(V, T, boundary_facets));

        // reset f_ext
        IOData::instance().f_exts[mesh_id].resizeLike(V);
        IOData::instance().f_exts.at(mesh_id).setZero();
    }

    auto MeshIO::get_all_mesh_ids() -> std::vector<pd::MeshIDType>
    {
        std::vector<pd::MeshIDType> ret;
        for (const auto& [id, mesh] : IOData::instance().models)
        {
            ret.push_back(id);
        }
        return ret;
    }
}
