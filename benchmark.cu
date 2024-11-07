#include "include/lbvh.cuh"
#include "include/scene.cuh"
#include "include/scene_loader.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <fcpw/fcpw.h>
#include <fcpw/fcpw_gpu.h>
#include <fcpw/utilities/scene_loader.h>
#include <chrono>

#ifdef _WIN32
#define NULL_STREAM_PATH "NUL:"
#else
#define NULL_STREAM_PATH "/dev/null"
#endif

static std::ofstream nullStream(NULL_STREAM_PATH);
static std::streambuf *oldCoutBuffer = nullptr;

#define SUPPRESS_STDOUT() \
    oldCoutBuffer = std::cout.rdbuf(nullStream.rdbuf());

#define RESTORE_STDOUT() \
    std::cout.rdbuf(oldCoutBuffer);

#define BENCHMARK_CLOSEST_PRIMITIVE
#define BENCHMARK_CLOSEST_SILHOUETTE
#define BENCHMARK_RAY_INTERSECTION

void benchmark_2d()
{
    std::vector<std::string> paths = {
        "example/2d/circle_in_square.obj",
        "example/2d/waker.obj"};

    for (const auto &path : paths)
    {
        std::cout << "Case: " << path << std::endl;
        int height = 400, width = 400;
        int N = height * width;
        double query_time;
        constexpr int repetitions = 32;
#ifdef BENCHMARK_CLOSEST_PRIMITIVE
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            fcpw::Scene<2> scene;
            fcpw::SceneLoader<2> loader;
            fcpw::files.clear();
            fcpw::files.push_back({path, fcpw::LoadingOption::ObjLineSegments});
            loader.loadFiles(scene, false);
            scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false, false);
            fcpw::GPUScene<2> scene_d((std::filesystem::current_path().parent_path() / "ext" / "fcpw").string());
            scene_d.transferToGPU(scene);

            std::vector<fcpw::GPUBoundingSphere> boundingSpheres;
            // Prepare fcpw query spheres.
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                boundingSpheres.push_back(fcpw::GPUBoundingSphere({x * 2.0f, y * 2.0f, 0.0f}, INFINITY));
            }

            std::vector<fcpw::GPUInteraction> interactions;
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                scene_d.findClosestPoints(boundingSpheres, interactions);
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                // std::cout << "FCPW: " << query_duration.count() << std::endl;
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }
            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "FCPW closest primitive: " << query_time << "us" << std::endl;

        query_time = 0;
        {
            SUPPRESS_STDOUT();
            lbvh::scene_loader<2> loader(path);
            const auto &vertices = loader.get_vertices();
            const auto &indices = loader.get_indices();
            lbvh::scene<2> scene(vertices.begin(), vertices.end(), indices.begin(), indices.end());
            scene.compute_silhouettes();
            scene.build_bvh();
            const auto bvh_dev = scene.get_bvh_device_ptr();

            thrust::host_vector<float2> queries;
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                queries.push_back({x * 2.0f, y * 2.0f});
            }

            std::vector<float> result_h;
            result_h.resize(N);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                // thrust::device_vector<float2> queries_d = queries;
                thrust::device_vector<float2> queries_d(N);
                thrust::copy(queries.begin(), queries.end(), queries_d.begin());
                thrust::device_vector<float> result(N);
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(queries_d.begin(), result.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(queries_d.end(), result.end())),
                                 [bvh_dev] __device__(thrust::tuple<float2 &, float &> t)
                                 {
                                     thrust::get<1>(t) = lbvh::query_device(bvh_dev, lbvh::nearest(thrust::get<0>(t)), lbvh::scene<2>::distance_calculator()).second;
                                 });
                // result_h = result;
                thrust::copy(result.begin(), result.end(), result_h.begin());
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                // std::cout << "SNCH-LBVH: " << query_duration.count() << std::endl;
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }
            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "SNCH-LBVH closest primitive: " << query_time << "us" << std::endl;
#endif // BENCHMARK_CLOSEST_PRIMITIVE
#ifdef BENCHMARK_CLOSEST_SILHOUETTE
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            fcpw::Scene<2> scene;
            fcpw::SceneLoader<2> loader;
            fcpw::files.clear();
            fcpw::files.push_back({path, fcpw::LoadingOption::ObjLineSegments});
            loader.loadFiles(scene, false);

            scene.computeSilhouettes();
            scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false, false);
            fcpw::GPUScene<2> scene_d((std::filesystem::current_path().parent_path() / "ext" / "fcpw").string());
            scene_d.transferToGPU(scene);

            std::vector<fcpw::GPUBoundingSphere> boundingSpheres;
            // Prepare fcpw query spheres.
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                boundingSpheres.push_back(fcpw::GPUBoundingSphere({x * 2.0f, y * 2.0f, 0.0f}, INFINITY));
            }

            std::vector<fcpw::GPUInteraction> interactions;
            std::vector<uint32_t> flip_normal(boundingSpheres.size(), 0u);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                scene_d.findClosestSilhouettePoints(boundingSpheres, flip_normal, interactions);
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }
            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "FCPW closest silhouette: " << query_time << "us" << std::endl;
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            lbvh::scene_loader<2> loader(path);
            const auto &vertices = loader.get_vertices();
            const auto &indices = loader.get_indices();
            lbvh::scene<2> scene(vertices.begin(), vertices.end(), indices.begin(), indices.end());
            scene.compute_silhouettes();
            scene.build_bvh();
            const auto bvh_dev = scene.get_bvh_device_ptr();

            thrust::host_vector<float2> queries;
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                queries.push_back({x * 2.0f, y * 2.0f});
            }

            std::vector<float> result_h;
            result_h.resize(N);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                // thrust::device_vector<float2> queries_d = queries;
                thrust::device_vector<float2> queries_d(N);
                thrust::copy(queries.begin(), queries.end(), queries_d.begin());
                thrust::device_vector<float> result(N);
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(queries_d.begin(), result.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(queries_d.end(), result.end())),
                                 [bvh_dev] __device__(thrust::tuple<float2 &, float &> t)
                                 {
                                     thrust::get<1>(t) = lbvh::query_device(bvh_dev, lbvh::nearest_silhouette(thrust::get<0>(t), false), lbvh::scene<2>::silhouette_distance_calculator());
                                 });
                // result_h = result;
                thrust::copy(result.begin(), result.end(), result_h.begin());
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }

            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "SNCH-LBVH closest silhouette: " << query_time << "us" << std::endl;
#endif // BENCHMARK_CLOSEST_SILHOUETTE
#ifdef BENCHMARK_RAY_INTERSECTION
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            fcpw::Scene<2> scene;
            fcpw::SceneLoader<2> loader;
            fcpw::files.clear();
            fcpw::files.push_back({path, fcpw::LoadingOption::ObjLineSegments});
            loader.loadFiles(scene, false);

            scene.computeSilhouettes();
            scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false, false);
            fcpw::GPUScene<2> scene_d((std::filesystem::current_path().parent_path() / "ext" / "fcpw").string());
            scene_d.transferToGPU(scene);

            std::vector<fcpw::GPURay> rays;
            // Prepare fcpw query spheres.
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                rays.push_back(fcpw::GPURay(fcpw::float3{x * 1.0f, y * 1.0f, 0.0f}, fcpw::float3{1.0f / std::sqrt(2.0f), 1.0f / std::sqrt(2.0f), 0.0f}));
            }

            std::vector<fcpw::GPUInteraction> interactions;
            // std::vector<uint32_t> flip_normal(boundingSpheres.size(), 0u);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                scene_d.intersect(rays, interactions, false);
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }
            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "FCPW ray intersection: " << query_time << "us" << std::endl;
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            lbvh::scene_loader<2> loader(path);
            const auto &vertices = loader.get_vertices();
            const auto &indices = loader.get_indices();
            lbvh::scene<2> scene(vertices.begin(), vertices.end(), indices.begin(), indices.end());
            scene.compute_silhouettes();
            scene.build_bvh();
            const auto bvh_dev = scene.get_bvh_device_ptr();

            thrust::host_vector<lbvh::Line<float, 2>> queries;
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                queries.push_back(lbvh::Line<float, 2>({x * 1.0f, y * 1.0f}, lbvh::normalize(make_float2(1.0f, 1.0f))));
            }

            std::vector<float> result_h;
            result_h.resize(N);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                // thrust::device_vector<float2> queries_d = queries;
                thrust::device_vector<lbvh::Line<float, 2>> queries_d = queries;
                thrust::device_vector<float> result(N);
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(queries_d.begin(), result.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(queries_d.end(), result.end())),
                                 [bvh_dev] __device__(thrust::tuple<lbvh::Line<float, 2> &, float &> t)
                                 {
                                     thrust::get<1>(t) = lbvh::query_device(bvh_dev, lbvh::line_intersect(thrust::get<0>(t)), lbvh::scene<2>::intersect_test()).second;
                                 });
                // result_h = result;
                thrust::copy(result.begin(), result.end(), result_h.begin());
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }

            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "SNCH-LBVH ray intersection: " << query_time << "us" << std::endl;
#endif // BENCHMARK_RAY_INTERSECTION
    }
}

void benchmark_3d()
{
    std::vector<std::string> paths = {
        "example/3d/suzanne.obj",
        "example/3d/suzanne_subdiv.obj",
        "example/3d/bunny.obj",
        "example/3d/armadillo.obj",
        "example/3d/kitten.obj"};

    for (const auto &path : paths)
    {
        std::cout << "Case: " << path << std::endl;
        int height = 400, width = 400;
        int N = height * width;
        double query_time;
        constexpr int repetitions = 32;
#ifdef BENCHMARK_CLOSEST_PRIMITIVE
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            fcpw::Scene<3> scene;
            fcpw::SceneLoader<3> loader;
            fcpw::files.clear();
            fcpw::files.push_back({path, fcpw::LoadingOption::ObjTriangles});
            loader.loadFiles(scene, false);
            scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false, false);
            fcpw::GPUScene<3> scene_d((std::filesystem::current_path().parent_path() / "ext" / "fcpw").string());
            scene_d.transferToGPU(scene);

            std::vector<fcpw::GPUBoundingSphere> boundingSpheres;
            // Prepare fcpw query spheres.
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                boundingSpheres.push_back(fcpw::GPUBoundingSphere({x * 2.0f, y * 2.0f, 0.0f}, INFINITY));
            }

            std::vector<fcpw::GPUInteraction> interactions;
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                scene_d.findClosestPoints(boundingSpheres, interactions);
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                // std::cout << "FCPW: " << query_duration.count() << std::endl;
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }
            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "FCPW closest primitive: " << query_time << "us" << std::endl;
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            lbvh::scene_loader<3> loader(path);
            const auto &vertices = loader.get_vertices();
            const auto &indices = loader.get_indices();
            lbvh::scene<3> scene(vertices.begin(), vertices.end(), indices.begin(), indices.end());
            scene.compute_silhouettes();
            scene.build_bvh();
            const auto bvh_dev = scene.get_bvh_device_ptr();

            thrust::host_vector<float3> queries;
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                queries.push_back({x * 2.0f, y * 2.0f, 0.0f});
            }

            std::vector<float> result_h;
            result_h.resize(N);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                // thrust::device_vector<float2> queries_d = queries;
                thrust::device_vector<float3> queries_d(N);
                thrust::copy(queries.begin(), queries.end(), queries_d.begin());
                thrust::device_vector<float> result(N);
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(queries_d.begin(), result.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(queries_d.end(), result.end())),
                                 [bvh_dev] __device__(thrust::tuple<float3 &, float &> t)
                                 {
                                     thrust::get<1>(t) = lbvh::query_device(bvh_dev, lbvh::nearest(thrust::get<0>(t)), lbvh::scene<3>::distance_calculator()).second;
                                 });
                // result_h = result;
                thrust::copy(result.begin(), result.end(), result_h.begin());
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                // std::cout << "SNCH-LBVH: " << query_duration.count() << std::endl;
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }
            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "SNCH-LBVH closest primitive: " << query_time << "us" << std::endl;
#endif // BENCHMARK_CLOSEST_PRIMITIVE
#ifdef BENCHMARK_CLOSEST_SILHOUETTE
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            fcpw::Scene<3> scene;
            fcpw::SceneLoader<3> loader;
            fcpw::files.clear();
            fcpw::files.push_back({path, fcpw::LoadingOption::ObjTriangles});
            loader.loadFiles(scene, false);

            scene.computeSilhouettes();
            scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false, false);
            fcpw::GPUScene<3> scene_d((std::filesystem::current_path().parent_path() / "ext" / "fcpw").string());
            scene_d.transferToGPU(scene);

            std::vector<fcpw::GPUBoundingSphere> boundingSpheres;
            // Prepare fcpw query spheres.
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                boundingSpheres.push_back(fcpw::GPUBoundingSphere({x * 2.0f, y * 2.0f, 0.0f}, INFINITY));
            }

            std::vector<fcpw::GPUInteraction> interactions;
            std::vector<uint32_t> flip_normal(boundingSpheres.size(), 0u);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                scene_d.findClosestSilhouettePoints(boundingSpheres, flip_normal, interactions);
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }
            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "FCPW closest silhouette: " << query_time << "us" << std::endl;
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            lbvh::scene_loader<3> loader(path);
            const auto &vertices = loader.get_vertices();
            const auto &indices = loader.get_indices();
            lbvh::scene<3> scene(vertices.begin(), vertices.end(), indices.begin(), indices.end());
            scene.compute_silhouettes();
            scene.build_bvh();
            const auto bvh_dev = scene.get_bvh_device_ptr();

            thrust::host_vector<float3> queries;
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                queries.push_back({x * 2.0f, y * 2.0f, 0.0f});
            }

            std::vector<float> result_h;
            result_h.resize(N);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                // thrust::device_vector<float2> queries_d = queries;
                thrust::device_vector<float3> queries_d(N);
                thrust::copy(queries.begin(), queries.end(), queries_d.begin());
                thrust::device_vector<float> result(N);
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(queries_d.begin(), result.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(queries_d.end(), result.end())),
                                 [bvh_dev] __device__(thrust::tuple<float3 &, float &> t)
                                 {
                                     thrust::get<1>(t) = lbvh::query_device(bvh_dev, lbvh::nearest_silhouette(thrust::get<0>(t), false), lbvh::scene<3>::silhouette_distance_calculator());
                                 });
                // result_h = result;
                thrust::copy(result.begin(), result.end(), result_h.begin());
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }

            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "SNCH-LBVH closest silhouette: " << query_time << "us" << std::endl;
#endif // BENCHMARK_CLOSEST_SILHOUETTE
#ifdef BENCHMARK_RAY_INTERSECTION
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            fcpw::Scene<3> scene;
            fcpw::SceneLoader<3> loader;
            fcpw::files.clear();
            fcpw::files.push_back({path, fcpw::LoadingOption::ObjTriangles});
            loader.loadFiles(scene, false);

            scene.computeSilhouettes();
            scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false, false);
            fcpw::GPUScene<3> scene_d((std::filesystem::current_path().parent_path() / "ext" / "fcpw").string());
            scene_d.transferToGPU(scene);

            std::vector<fcpw::GPURay> rays;
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                rays.push_back(fcpw::GPURay(fcpw::float3{x * 1.0f, y * 1.0f, 0.0f}, fcpw::float3{1.0f / std::sqrt(2.0f), 1.0f / std::sqrt(2.0f), 0.0f}));
            }

            std::vector<fcpw::GPUInteraction> interactions;
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                scene_d.intersect(rays, interactions, false);
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }
            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "FCPW ray intersection: " << query_time << "us" << std::endl;
        query_time = 0;
        {
            SUPPRESS_STDOUT();
            lbvh::scene_loader<3> loader(path);
            const auto &vertices = loader.get_vertices();
            const auto &indices = loader.get_indices();
            lbvh::scene<3> scene(vertices.begin(), vertices.end(), indices.begin(), indices.end());
            scene.compute_silhouettes();
            scene.build_bvh();
            const auto bvh_dev = scene.get_bvh_device_ptr();

            thrust::host_vector<lbvh::Line<float, 3>> queries;
            for (int i = 0; i < N; ++i)
            {
                float x = (static_cast<float>(i % width) / static_cast<float>(width)) * 2.0f - 1.0f;
                float y = (static_cast<float>(i / width) / static_cast<float>(height)) * 2.0f - 1.0f;
                queries.push_back(lbvh::Line<float, 3>(lbvh::vec3_to_vec4(make_float3(x * 1.0f, y * 1.0f, 0.0f)), lbvh::vec3_to_vec4(lbvh::normalize(make_float3(1.0f, 1.0f, 0.0f)))));
            }

            std::vector<float> result_h;
            result_h.resize(N);
            for (int i = 0; i < repetitions + 1; ++i)
            {
                auto query_start = std::chrono::high_resolution_clock::now();
                // thrust::device_vector<float2> queries_d = queries;
                thrust::device_vector<lbvh::Line<float, 3>> queries_d = queries;
                thrust::device_vector<float> result(N);
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(queries_d.begin(), result.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(queries_d.end(), result.end())),
                                 [bvh_dev] __device__(thrust::tuple<lbvh::Line<float, 3> &, float &> t)
                                 {
                                     thrust::get<1>(t) = lbvh::query_device(bvh_dev, lbvh::line_intersect(thrust::get<0>(t)), lbvh::scene<3>::intersect_test()).second;
                                 });
                // result_h = result;
                thrust::copy(result.begin(), result.end(), result_h.begin());
                auto query_end = std::chrono::high_resolution_clock::now();
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);
                if (i != 0)
                {
                    query_time += query_duration.count();
                }
            }

            query_time /= repetitions;
            RESTORE_STDOUT();
        }
        std::cout << "SNCH-LBVH ray intersection: " << query_time << "us" << std::endl;
#endif // BENCHMARK_RAY_INTERSECTION
    }
}

int main(int argc, char *argv[])
{
    benchmark_2d();
    benchmark_3d();
    return 0;
}
