#include "include/lbvh.cuh"
#include "include/scene.cuh"
#include "include/scene_loader.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <primitive/silhouette> <scene_scale> <color_scale>" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    lbvh::scene_loader<3> loader(filename);
    const auto &vertices = loader.get_vertices();
    const auto &indices = loader.get_indices();
    lbvh::scene<3> scene(vertices.begin(), vertices.end(), indices.begin(), indices.end());
    scene.compute_silhouettes();
    scene.build_bvh();

    const auto bvh_dev = scene.get_bvh_device_ptr();

    int height = 400, width = 400;
    int N = height * width;
    thrust::device_vector<float> result(N);

    std::string mode = argv[2];
    float scale = std::stof(argv[3]);
    float color_scale = std::stof(argv[4]);
    if (mode == "primitive")
    {
        thrust::transform(
            thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(N),
            result.begin(),
            [bvh_dev, width, height, scale] __device__(const unsigned int idx)
            {
                float x = (static_cast<float>(idx % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(idx / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                float3 coord = make_float3(x * scale, y * scale, 0.0f);
                const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(coord), lbvh::scene<3>::distance_calculator());
                return nest.second;
            });
    }
    else if (mode == "silhouette")
    {
        thrust::transform(
            thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(N),
            result.begin(),
            [bvh_dev, width, height, scale] __device__(const unsigned int idx)
            {
                float x = (static_cast<float>(idx % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(idx / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                float3 coord = make_float3(x * scale, y * scale, 0.0f);
                const auto dest = lbvh::query_device(bvh_dev, lbvh::nearest_silhouette(coord, false), lbvh::scene<3>::silhouette_distance_calculator());
                return dest;
            });
    }
    else if (mode == "intersection")
    {
        if (argc < 6)
        {
            std::cerr << "Usage: " << argv[0] << " <filename> intersection <scene_scale> <color_scale> <probe_angle>" << std::endl;
            return 1;
        }
        float angle = std::stof(argv[5]);
        thrust::transform(
            thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(N),
            result.begin(),
            [bvh_dev, width, height, scale, angle] __device__(const unsigned int idx)
            {
                float x = (static_cast<float>(idx % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(idx / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                float3 coord = make_float3(x * scale, y * scale, 0.0f);
                // auto li = lbvh::line_intersect(lbvh::Line<float, 2>(coord, lbvh::normalize(make_float2(1.0f, 1.0f))));
                const auto dest = lbvh::query_device(
                    bvh_dev,
                    lbvh::line_intersect(lbvh::Line<float, 3>(lbvh::vec3_to_vec4(coord), lbvh::vec3_to_vec4(lbvh::normalize(make_float3(std::cos(angle / 180.0f * M_PI), std::sin(angle / 180.0f * M_PI), 0.0f))))),
                    lbvh::scene<3>::intersect_test()
                );
                if (dest.first == false)
                {
                    return 1.0f;
                }
                else
                {
                    return dest.second;
                }
            });
    }
    else
    {
        std::cout << "Invalid mode." << std::endl;
        exit(1);
    }

    thrust::host_vector<float> host_result = result;
    std::ofstream file("output.ppm");
    file << "P3\n"
         << width << " " << height << "\n255\n";
    for (int i = 0; i < N; i++)
    {
        int gray_value = static_cast<int>(host_result[i] / color_scale * 255);
        gray_value = std::max(0, std::min(255, gray_value));
        file << gray_value << " " << gray_value << " " << gray_value << "\n";
    }
    file.close();
    std::cout << "Image savd to output.ppm" << std::endl;

    return 0;
}