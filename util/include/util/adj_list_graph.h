#pragma once
#include <vector>
#include <unordered_set>

#include <pd/types.h>

namespace util
{
    struct AdjListGraph
    {
    public:
        // Graph Constructor
        AdjListGraph(const std::vector<std::pair<pd::VertexIndexType, pd::VertexIndexType>>& edges, int n)
        {
            adj_list.resize(n);

            // add edges
            for (const auto& [src, dst] : edges)
            {
                adj_list[src].insert(dst);
                adj_list[dst].insert(src);
            }
        }

        const std::vector<std::unordered_set<pd::VertexIndexType>>& get_adj_list() { return adj_list; };

        std::vector<pd::VertexIndexType> get_vertex_1_ring_neighbors(pd::VertexIndexType vi)
        {
            std::vector<pd::VertexIndexType> neighbor_indices;
            for (const auto vj : adj_list[vi])
            {
                neighbor_indices.push_back(vj);
            }
            return neighbor_indices;
        };

    private:
        std::vector<std::unordered_set<pd::VertexIndexType>> adj_list;
    };
}
