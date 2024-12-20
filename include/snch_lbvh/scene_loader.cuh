#pragma once

#include "lbvh.cuh"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <iostream>

namespace lbvh
{
    template <unsigned int dim>
    class scene_loader;

    template <>
    class scene_loader<2>
    {
    public:
        scene_loader(const std::string &filename)
        {
            load_from_obj(filename);
        }

        const std::vector<float2> &get_vertices() const
        {
            return vertices;
        }

        const std::vector<int2> &get_indices() const
        {
            return indices;
        }

        std::size_t vertices_size() const
        {
            return vertices.size();
        }

        std::size_t primitives_size() const
        {
            return indices.size();
        }

    private:
        std::vector<float2> vertices;
        std::vector<int2> indices;

        void load_from_obj(const std::string &filename)
        {
            std::ifstream obj_file(filename);
            if (!obj_file.is_open())
            {
                throw std::runtime_error("Could not open .obj file.");
            }

            std::string line;
            while (std::getline(obj_file, line))
            {
                std::istringstream line_stream(line);
                std::string prefix;
                line_stream >> prefix;

                if (prefix == "v")
                {
                    float x, y, z;
                    line_stream >> x >> y >> z;
                    vertices.emplace_back(make_float2(x, y));
                }
                else if (prefix == "l")
                {
                    int v1, v2;
                    line_stream >> v1 >> v2;
                    indices.emplace_back(make_int2(v1 - 1, v2 - 1));
                }
                else
                {
                    continue;
                }
            }
            obj_file.close();
        }
    };

    template <>
    class scene_loader<3>
    {
    public:
        scene_loader(const std::string &filename)
        {
            load_from_obj(filename);
        }

        const std::vector<float3> &get_vertices() const
        {
            return vertices;
        }

        const std::vector<int3> &get_indices() const
        {
            return indices;
        }

        std::size_t vertices_size() const
        {
            return vertices.size();
        }

        std::size_t primitives_size() const
        {
            return indices.size();
        }

    private:
        std::vector<float3> vertices;
        std::vector<int3> indices;

        void load_from_obj(const std::string &filename)
        {
            std::ifstream obj_file(filename);
            if (!obj_file.is_open())
            {
                throw std::runtime_error("Could not open .obj file.");
            }

            std::string line;
            while (std::getline(obj_file, line))
            {
                std::istringstream line_stream(line);
                std::string prefix;
                line_stream >> prefix;

                if (prefix == "v")
                {
                    float x, y, z;
                    line_stream >> x >> y >> z;
                    vertices.emplace_back(make_float3(x, y, z));
                }
                else if (prefix == "f")
                {
                    int v1, v2, v3;
                    line_stream >> v1 >> v2 >> v3;
                    indices.emplace_back(make_int3(v1 - 1, v2 - 1, v3 - 1));
                }
                else
                {
                    continue;
                }
            }
            obj_file.close();
        }
    };
}