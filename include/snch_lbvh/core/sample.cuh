#pragma once
#include "predicator.cuh"

namespace lbvh
{

    template <
        typename Real, unsigned int dim, typename Objects, bool IsConst,
        typename SampleFunc>
    SNCH_LBVH_DEVICE typename vector_of<Real, dim>::type sample_on_object(
        const detail::basic_device_bvh<Real, dim, Objects, IsConst> &bvh,
        const int object_idx,
        SampleFunc sample_func,
        float u,
        float v
    )
    {
        const auto &object = bvh.objects[object_idx];
        typename vector_of<Real, dim>::type ret = sample_func(object, u, v);
        return ret;
    }

    template <
        typename Real, unsigned int dim, typename Objects, bool IsConst,
        typename SphereIntersectionTestFunc, typename MeasurementFunc, typename WeightFunc>
    SNCH_LBVH_DEVICE thrust::pair<int /*object index*/, float /*pdf*/> sample_object_in_sphere(
        const detail::basic_device_bvh<Real, dim, Objects, IsConst> &bvh,
        const query_sphere_intersect<Real, dim> q,
        SphereIntersectionTestFunc sphere_intersects,
        MeasurementFunc measure,
        WeightFunc weight,
        float u) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, dim, Objects, IsConst>;
        using real_type = typename bvh_type::real_type;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        // pair of {node_idx, pdf}
        thrust::pair<index_type, real_type> stack[64];
        thrust::pair<index_type, real_type> *stack_ptr = stack;
        *stack_ptr++ = thrust::make_pair(0, 1.0f);

        int object_index = -1;
        float object_pdf;
        do
        {
            const auto node = *--stack_ptr;

            const auto obj_idx = bvh.nodes[node.first].object_idx;
            // printf("node id: %d, object index: %d\n", node.first, obj_idx);
            if(obj_idx != 0xFFFFFFFF) // leaf
            {
                // printf("Entering leaf.\n");
                if(sphere_intersects(q.sph, bvh.objects[obj_idx]))
                {
                    // printf("Intersection pass.\n");
                    object_index = obj_idx;
                    object_pdf = node.second;
                }
                // object_index = obj_idx;
                // object_pdf = node.second;
            }
            else
            {
                const index_type L_idx = bvh.nodes[node.first].left_idx;
                const index_type R_idx = bvh.nodes[node.first].right_idx;

                const aabb_type &L_box = bvh.aabbs[L_idx];
                const aabb_type &R_box = bvh.aabbs[R_idx];

                const real_type L_weight = intersect_sphere(q.sph, L_box) ? weight(q.sph.origin, centroid(L_box)) : 0;
                const real_type R_weight = intersect_sphere(q.sph, R_box) ? weight(q.sph.origin, centroid(R_box)) : 0;

                const real_type total_weight = L_weight + R_weight;

                if (total_weight > 0)
                {
                    const real_type L_prob = L_weight / total_weight;
                    // printf("selection prob:%f, L_prob: %f, L_weight: %f, total_weight: %f\n", selection_prob, L_prob, L_weight, total_weight);
                    if (u < L_prob)
                    {
                        u /= L_prob;
                        *stack_ptr++ = thrust::make_pair(L_idx, L_prob * node.second);
                    }
                    else
                    {
                        const real_type R_prob = 1.0f - L_prob;
                        u = (u - L_prob) / R_prob;
                        *stack_ptr++ = thrust::make_pair(R_idx, R_prob * node.second);
                    }
                }
            }
            // printf("--------\n");
        } while (stack < stack_ptr);
        return thrust::make_pair(object_index, object_pdf);
    }
}