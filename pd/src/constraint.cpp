#include <pd/constraint.h>

namespace pd
{
    Constraint::Constraint(const Constraint& rhs): wc(rhs.wc)
    {
        // printf("Call Base cpy constructor\n");

        realloc_vertices(rhs.n_vertices);
        memcpy(vertices, rhs.vertices, sizeof(VertexIndexType) * n_vertices);
    }

    Constraint::Constraint(Constraint&& rhs) noexcept: wc(rhs.wc)
    {
        printf("Call Base move constructor\n");

        n_vertices = rhs.n_vertices;
        vertices = rhs.vertices;

        rhs.n_vertices = 0;
        rhs.vertices = nullptr;
    }

    Constraint& Constraint::operator=(const Constraint& rhs)
    {
        printf("Call Base assignment operator\n");
        if (this != &rhs)
        {
            realloc_vertices(rhs.n_vertices);
            memcpy(vertices, rhs.vertices, sizeof(VertexIndexType) * n_vertices);

            wc = rhs.wc;
        }
        return *this;
    }

    Constraint& Constraint::operator=(Constraint&& rhs) noexcept
    {
        printf("Call Base move assignment operator\n");
        if (this != &rhs)
        {
            cudaFree(vertices);
            n_vertices = rhs.n_vertices;
            vertices = rhs.vertices;

            rhs.n_vertices = 0;
            rhs.vertices = nullptr;

            wc = rhs.wc;
        }
        return *this;
    }

    int Constraint::get_involved_vertices(VertexIndexType** vertices) const
    {
        *vertices = this->vertices;
        return n_vertices;
    }

    void Constraint::set_vertex_offset(int n_vertex_offset)
    {
        for (int i = 0; i < n_vertices; i++)
        {
            vertices[i] += n_vertex_offset;
        }
    }

    void Constraint::realloc_vertices(int size)
    {
        cudaFree(vertices);
        n_vertices = size;
        cudaMallocManaged(&vertices, sizeof(VertexIndexType) * n_vertices);
    }
}