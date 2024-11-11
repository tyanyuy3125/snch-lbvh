#pragma once
#include "predicator.cuh"

namespace lbvh
{
    template <
        typename Real, unsigned int dim, typename Objects, bool IsConst,
        typename SphereIntersectionTestFunc>
    SNCH_LBVH_DEVICE thrust::pair<int /*object index*/, float /*pdf*/> sample_triangle_in_sphere(
        const detail::basic_device_bvh<Real, dim, Objects, IsConst> &bvh,
        const query_sphere_intersect<Real, dim> q,
        SphereIntersectionTestFunc sphere_intersects) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, dim, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        index_type stack[64];
        index_type *stack_ptr = stack;
        *stack_ptr++ = 0;

        
        do
        {
            const index_type node = *--stack_ptr;

            const auto obj_idx = bvh.nodes[node.first].object_idx;
            if(obj_idx != 0xFFFFFFFF) // leaf
            {
                float total_area = 0;

            }
        } while (stack < stack_ptr);
    }
}