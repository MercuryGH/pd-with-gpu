#pragma once
#include <vector>
#include <unordered_set>

namespace util
{
    struct AdjListGraph
    {
    public:
        // Graph Constructor
        AdjListGraph(const std::vector<std::pair<int, int>>& edges, int n)
        {
            adj_list.resize(n);

            // add edges
            for (const auto& [src, dst] : edges)
            {
                adj_list[src].insert(dst);
                adj_list[dst].insert(src);
            }
        }

        const std::vector<std::unordered_set<int>>& get_adj_list() { return adj_list; };

    private:
        std::vector<std::unordered_set<int>> adj_list;

    };
}
