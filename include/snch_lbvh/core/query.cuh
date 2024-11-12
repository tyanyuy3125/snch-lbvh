#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH
#include "predicator.cuh"

namespace lbvh
{
    // query object indices that intersects with query line.
    //
    // requirements:
    // - OutputIterator should be writable and its object_type should be uint32_t
    //
    template <
        typename Real, unsigned int dim, typename Objects, bool IsConst, typename OutputIterator,
        typename IntersectionTestFunc>
    SNCH_LBVH_DEVICE unsigned int query_device(
        const detail::basic_device_bvh<Real, dim, Objects, IsConst> &bvh, const query_line_intersect<Real, dim> q,
        IntersectionTestFunc element_intersects, OutputIterator outiter, const unsigned int max_buffer_size) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, dim, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        index_type stack[64]; // is it okay?
        index_type *stack_ptr = stack;
        *stack_ptr++ = 0; // root node is always 0

        unsigned int num_found = 0;
        do
        {
            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            if (intersects(q.l, bvh.aabbs[L_idx]))
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    auto flag_data = element_intersects(q.l, bvh.objects[obj_idx]);
                    if (flag_data.first)
                    {
                        if (num_found < max_buffer_size)
                        {
                            *outiter++ = thrust::pair<unsigned int, decltype(flag_data.second)>(obj_idx, flag_data.second);
                        }
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = L_idx;
                }
            }
            if (intersects(q.l, bvh.aabbs[R_idx]))
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    auto flag_data = element_intersects(q.l, bvh.objects[obj_idx]);
                    if (flag_data.first)
                    {
                        if (num_found < max_buffer_size)
                        {
                            *outiter++ = thrust::pair<unsigned int, decltype(flag_data.second)>(obj_idx, flag_data.second);
                        }
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = R_idx;
                }
            }
        } while (stack < stack_ptr);
        return num_found;
    }

    template <
        typename Real, unsigned int dim, typename Objects, bool IsConst,
        typename IntersectionTestFunc>
    SNCH_LBVH_DEVICE thrust::pair<bool, Real> query_device(
        const detail::basic_device_bvh<Real, dim, Objects, IsConst> &bvh,
        const query_ray_intersect<Real, dim> q,
        IntersectionTestFunc element_intersects) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, dim, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;
        using real_type = typename bvh_type::real_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        thrust::pair<index_type, real_type> stack[64]; // is it okay?
        thrust::pair<index_type, real_type> *stack_ptr = stack;
        // *stack_ptr++ = 0; // root node is always 0
        *stack_ptr++ = thrust::make_pair(0, infinity<real_type>());

        Real min_dist = infinity<real_type>();
        bool intersection_found = false;

        do
        {
            const auto node = *--stack_ptr;
            if (node.second > min_dist)
            {
                continue;
            }

            const auto obj_idx = bvh.nodes[node.first].object_idx;
            if (obj_idx != 0xFFFFFFFF) // leaf
            {
                auto flag_data = element_intersects(q.r, bvh.objects[obj_idx]);
                if (flag_data.first && flag_data.second < min_dist)
                {
                    min_dist = flag_data.second;
                    intersection_found = true;
                }
            }
            else // not leaf
            {
                const index_type L_idx = bvh.nodes[node.first].left_idx;
                const index_type R_idx = bvh.nodes[node.first].right_idx;

                float L_dist, R_dist;
                bool L_hit = intersects_d(q.r, bvh.aabbs[L_idx], &L_dist);
                bool R_hit = intersects_d(q.r, bvh.aabbs[R_idx], &R_dist);

                if (L_hit && R_hit)
                {
                    index_type closer = L_idx;
                    index_type other = R_idx;

                    if (R_dist < L_dist)
                    {
                        swap(L_dist, R_dist);
                        swap(closer, other);
                    }

                    *stack_ptr++ = thrust::make_pair(other, R_dist);
                    *stack_ptr++ = thrust::make_pair(closer, L_dist);
                }
                else if (L_hit)
                {
                    *stack_ptr++ = thrust::make_pair(L_idx, L_dist);
                }
                else if (R_hit)
                {
                    *stack_ptr++ = thrust::make_pair(R_idx, R_dist);
                }
            }
        } while (stack < stack_ptr);

        return thrust::make_pair(intersection_found, min_dist);
    }

    // query object indices that potentially overlaps with query aabb.
    //
    // requirements:
    // - OutputIterator should be writable and its object_type should be uint32_t
    //
    template <typename Real, unsigned int dim, typename Objects, bool IsConst, typename OutputIterator>
    SNCH_LBVH_DEVICE unsigned int query_device(
        const detail::basic_device_bvh<Real, dim, Objects, IsConst> &bvh, const query_overlap<Real, dim> q,
        OutputIterator outiter, const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, dim, Objects, IsConst>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        index_type stack[64]; // is it okay?
        index_type *stack_ptr = stack;
        *stack_ptr++ = 0; // root node is always 0

        unsigned int num_found = 0;
        do
        {
            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            if (intersects(q.target, bvh.aabbs[L_idx]))
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if (num_found < max_buffer_size)
                    {
                        *outiter++ = obj_idx;
                    }
                    ++num_found;
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = L_idx;
                }
            }
            if (intersects(q.target, bvh.aabbs[R_idx]))
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if (num_found < max_buffer_size)
                    {
                        *outiter++ = obj_idx;
                    }
                    ++num_found;
                }
                else // the node is not a leaf.
                {
                    *stack_ptr++ = R_idx;
                }
            }
        } while (stack < stack_ptr);
        return num_found;
    }

    // query object index that is the nearst to the query point.
    //
    // requirements:
    // - DistanceCalculator must be able to calc distance between a point to an object.
    //
    template <typename Real, unsigned int dim, typename Objects, bool IsConst, typename DistanceCalculator>
    SNCH_LBVH_DEVICE thrust::pair<unsigned int, Real> query_device(
        const detail::basic_device_bvh<Real, dim, Objects, IsConst> &bvh, const query_nearest<Real, dim> &q,
        DistanceCalculator calc_dist) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, dim, Objects, IsConst>;
        using real_type = typename bvh_type::real_type;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        // pair of {node_idx, mindist}
        thrust::pair<index_type, real_type> stack[64];
        thrust::pair<index_type, real_type> *stack_ptr = stack;
        *stack_ptr++ = thrust::make_pair(0, mindist(bvh.aabbs[0], q.target));

        unsigned int nearest = 0xFFFFFFFF;
        real_type dist_to_nearest_object = infinity<real_type>();
        do
        {
            const auto node = *--stack_ptr;
            if (node.second > dist_to_nearest_object)
            {
                // if aabb mindist > already_found_mindist, it cannot have a nearest
                continue;
            }

            const index_type L_idx = bvh.nodes[node.first].left_idx;
            const index_type R_idx = bvh.nodes[node.first].right_idx;

            const aabb_type &L_box = bvh.aabbs[L_idx];
            const aabb_type &R_box = bvh.aabbs[R_idx];

            const real_type L_mindist = mindist(L_box, q.target);
            const real_type R_mindist = mindist(R_box, q.target);

            const real_type L_minmaxdist = minmaxdist(L_box, q.target);
            const real_type R_minmaxdist = minmaxdist(R_box, q.target);

            // there should be an object that locates within minmaxdist.

            if (L_mindist <= R_minmaxdist) // L is worth considering
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    real_type dist = calc_dist(q.target, bvh.objects[obj_idx]);
                    dist *= dist;
                    if (dist <= dist_to_nearest_object)
                    {
                        dist_to_nearest_object = dist;
                        nearest = obj_idx;
                    }
                }
                else
                {
                    *stack_ptr++ = thrust::make_pair(L_idx, L_mindist);
                }
            }
            if (R_mindist <= L_minmaxdist) // R is worth considering
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    real_type dist = calc_dist(q.target, bvh.objects[obj_idx]);
                    dist *= dist;
                    if (dist <= dist_to_nearest_object)
                    {
                        dist_to_nearest_object = dist;
                        nearest = obj_idx;
                    }
                }
                else
                {
                    *stack_ptr++ = thrust::make_pair(R_idx, R_mindist);
                }
            }
            assert(stack_ptr < stack + 64);
        } while (stack < stack_ptr);
        return thrust::make_pair(nearest, std::sqrt(dist_to_nearest_object));
    }

    // query object index that is the nearst to the query point.
    //
    // requirements:
    // - DistanceCalculator must be able to calc distance between a point to an object.
    //
    template <typename Real, unsigned int dim, typename Objects, bool IsConst, typename SilhouetteDistanceCalculator>
    SNCH_LBVH_DEVICE Real query_device(
        const detail::basic_device_bvh<Real, dim, Objects, IsConst> &bvh, const query_nearest_silhouette<Real, dim> &q,
        SilhouetteDistanceCalculator calc_dist) noexcept
    {
        using bvh_type = detail::basic_device_bvh<Real, dim, Objects, IsConst>;
        using real_type = typename bvh_type::real_type;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using cone_type = typename bvh_type::cone_type;
        using node_type = typename bvh_type::node_type;

        // pair of {node_idx, mindist}
        thrust::pair<index_type, real_type> stack[64];
        thrust::pair<index_type, real_type> *stack_ptr = stack;
        *stack_ptr++ = thrust::make_pair(0, mindist(bvh.aabbs[0], q.target));

        // unsigned int nearest = 0xFFFFFFFF;
        real_type dist_to_nearest_silhouette = infinity<real_type>();
        do
        {
            const auto node = *--stack_ptr;
            if (node.second > dist_to_nearest_silhouette * dist_to_nearest_silhouette)
            {
                continue;
            }

            const index_type L_idx = bvh.nodes[node.first].left_idx;
            const index_type R_idx = bvh.nodes[node.first].right_idx;

            const aabb_type &L_box = bvh.aabbs[L_idx];
            const aabb_type &R_box = bvh.aabbs[R_idx];

            const cone_type &L_cone = bvh.cones[L_idx];
            const cone_type &R_cone = bvh.cones[R_idx];

            const real_type L_mindist_squared = mindist(L_box, q.target);
            const real_type R_mindist_squared = mindist(R_box, q.target);

            real_type stubs[2];

            const bool hit_L = is_valid(L_cone) && overlap(L_cone, q.target, L_box, L_mindist_squared, &stubs[0], &stubs[1]);
            const bool hit_R = is_valid(R_cone) && overlap(R_cone, q.target, R_box, R_mindist_squared, &stubs[0], &stubs[1]);

            if (hit_L)
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    real_type dist = infinity<real_type>();
                    bool found = calc_dist(
                        q.target,
                        bvh.objects[obj_idx],
                        dist_to_nearest_silhouette * dist_to_nearest_silhouette,
                        dist,
                        q.flip_normal_orientation,
                        0.0f // TODO
                    );
                    if (found && dist <= dist_to_nearest_silhouette)
                    {
                        dist_to_nearest_silhouette = dist;
                        // TODO: identify nearest index
                    }
                }
                else
                {
                    *stack_ptr++ = thrust::make_pair(L_idx, L_mindist_squared);
                }
            }
            if (hit_R)
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    real_type dist = infinity<real_type>();
                    bool found = calc_dist(
                        q.target,
                        bvh.objects[obj_idx],
                        dist_to_nearest_silhouette * dist_to_nearest_silhouette,
                        dist,
                        q.flip_normal_orientation,
                        0.0f // TODO
                    );
                    if (found && dist <= dist_to_nearest_silhouette)
                    {
                        dist_to_nearest_silhouette = dist;
                        // TODO: identify nearest index
                    }
                }
                else
                {
                    *stack_ptr++ = thrust::make_pair(R_idx, R_mindist_squared);
                }
            }

            assert(stack_ptr < stack + 64);
        } while (stack < stack_ptr);
        return dist_to_nearest_silhouette;
    }

    // query object indices that intersects with query line.
    //
    // requirements:
    // - OutputIterator should be writable and its object_type should be uint32_t
    //
    template <
        typename Real, unsigned int dim, typename Objects, typename AABBGetter, typename MortonCodeCalculator,
        typename OutputIterator, typename IntersectionTestFunc>
    SNCH_LBVH_DEVICE unsigned int query_host(
        const bvh<Real, dim, Objects, AABBGetter, MortonCodeCalculator> &tree, const query_line_intersect<Real, dim> q,
        IntersectionTestFunc element_intersects, OutputIterator outiter, const unsigned int max_buffer_size = 64) noexcept
    {
        using bvh_type = ::lbvh::bvh<Real, dim, Objects, AABBGetter, MortonCodeCalculator>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        if (!tree.query_host_enabled())
        {
            throw std::runtime_error("lbvh::bvh query_host is not enabled");
        }

        std::vector<std::size_t> stack;
        stack.reserve(64);
        stack.push_back(0);

        unsigned int num_found = 0;
        do
        {
            const index_type node = stack.back();
            stack.pop_back();
            const index_type L_idx = tree.nodes_host()[node].left_idx;
            const index_type R_idx = tree.nodes_host()[node].right_idx;

            if (intersects(q.l, tree.aabbs_host()[L_idx]))
            {
                const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if (element_intersects(q.l, tree.objects_host()[obj_idx]))
                    {
                        if (num_found < max_buffer_size)
                        {
                            *outiter++ = obj_idx;
                        }
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    stack.push_back(L_idx);
                }
            }
            if (intersects(q.l, tree.aabbs_host()[R_idx]))
            {
                const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if (element_intersects(q.l, tree.objects_host()[obj_idx]))
                    {
                        if (num_found < max_buffer_size)
                        {
                            *outiter++ = obj_idx;
                        }
                        ++num_found;
                    }
                }
                else // the node is not a leaf.
                {
                    stack.push_back(R_idx);
                }
            }
        } while (!stack.empty());
        return num_found;
    }

    template <
        typename Real, unsigned int dim, typename Objects, typename AABBGetter, typename MortonCodeCalculator,
        typename OutputIterator>
    unsigned int query_host(
        const bvh<Real, dim, Objects, AABBGetter, MortonCodeCalculator> &tree, const query_overlap<Real, dim> q,
        OutputIterator outiter, const unsigned int max_buffer_size = 0xFFFFFFFF)
    {
        using bvh_type = ::lbvh::bvh<Real, dim, Objects, AABBGetter, MortonCodeCalculator>;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        if (!tree.query_host_enabled())
        {
            throw std::runtime_error("lbvh::bvh query_host is not enabled");
        }

        std::vector<std::size_t> stack;
        stack.reserve(64);
        stack.push_back(0);

        unsigned int num_found = 0;
        do
        {
            const index_type node = stack.back();
            stack.pop_back();
            const index_type L_idx = tree.nodes_host()[node].left_idx;
            const index_type R_idx = tree.nodes_host()[node].right_idx;

            if (intersects(q.target, tree.aabbs_host()[L_idx]))
            {
                const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if (num_found < max_buffer_size)
                    {
                        *outiter++ = obj_idx;
                    }
                    ++num_found;
                }
                else // the node is not a leaf.
                {
                    stack.push_back(L_idx);
                }
            }
            if (intersects(q.target, tree.aabbs_host()[R_idx]))
            {
                const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF)
                {
                    if (num_found < max_buffer_size)
                    {
                        *outiter++ = obj_idx;
                    }
                    ++num_found;
                }
                else // the node is not a leaf.
                {
                    stack.push_back(R_idx);
                }
            }
        } while (!stack.empty());
        return num_found;
    }

    template <
        typename Real, unsigned int dim, typename Objects, typename AABBGetter, typename MortonCodeCalculator,
        typename DistanceCalculator>
    std::pair<unsigned int, Real> query_host(
        const bvh<Real, dim, Objects, AABBGetter, MortonCodeCalculator> &tree, const query_nearest<Real, dim> &q,
        DistanceCalculator calc_dist) noexcept
    {
        using bvh_type = ::lbvh::bvh<Real, dim, Objects, AABBGetter, MortonCodeCalculator>;
        using real_type = typename bvh_type::real_type;
        using index_type = typename bvh_type::index_type;
        using aabb_type = typename bvh_type::aabb_type;
        using node_type = typename bvh_type::node_type;

        if (!tree.query_host_enabled())
        {
            throw std::runtime_error("lbvh::bvh query_host is not enabled");
        }

        // pair of {node_idx, mindist}
        std::vector<std::pair<index_type, real_type>> stack = {{0, mindist(tree.aabbs_host()[0], q.target)}};
        stack.reserve(64);

        unsigned int nearest = 0xFFFFFFFF;
        real_type current_nearest_dist = infinity<real_type>();
        do
        {
            const auto node = stack.back();
            stack.pop_back();
            if (node.second > current_nearest_dist)
            {
                // if aabb mindist > already_found_mindist, it cannot have a nearest
                continue;
            }

            const index_type L_idx = tree.nodes_host()[node.first].left_idx;
            const index_type R_idx = tree.nodes_host()[node.first].right_idx;

            const aabb_type &L_box = tree.aabbs_host()[L_idx];
            const aabb_type &R_box = tree.aabbs_host()[R_idx];

            const real_type L_mindist = mindist(L_box, q.target);
            const real_type R_mindist = mindist(R_box, q.target);

            const real_type L_minmaxdist = minmaxdist(L_box, q.target);
            const real_type R_minmaxdist = minmaxdist(R_box, q.target);

            // there should be an object that locates within minmaxdist.

            if (L_mindist <= R_minmaxdist) // L is worth considering
            {
                const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const real_type dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                    if (dist <= current_nearest_dist)
                    {
                        current_nearest_dist = dist;
                        nearest = obj_idx;
                    }
                }
                else
                {
                    stack.emplace_back(L_idx, L_mindist);
                }
            }
            if (R_mindist <= L_minmaxdist) // R is worth considering
            {
                const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const real_type dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                    if (dist <= current_nearest_dist)
                    {
                        current_nearest_dist = dist;
                        nearest = obj_idx;
                    }
                }
                else
                {
                    stack.emplace_back(R_idx, R_mindist);
                }
            }
        } while (!stack.empty());
        return std::make_pair(nearest, current_nearest_dist);
    }
} // namespace lbvh
#endif // LBVH_QUERY_CUH
