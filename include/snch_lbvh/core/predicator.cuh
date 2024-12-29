#ifndef LBVH_PREDICATOR_CUH
#define LBVH_PREDICATOR_CUH
#include "aabb.cuh"

namespace lbvh
{

    template <typename Real, unsigned int dim>
    struct query_line_intersect
    {
        using vector_type = typename vector_of<Real, dim>::type;

        SNCH_LBVH_HOST_DEVICE query_line_intersect(const line<Real, dim> &l) : l(l) {}

        query_line_intersect() = default;
        ~query_line_intersect() = default;
        query_line_intersect(const query_line_intersect &) = default;
        query_line_intersect(query_line_intersect &&) = default;
        query_line_intersect &operator=(const query_line_intersect &) = default;
        query_line_intersect &operator=(query_line_intersect &&) = default;

        line<Real, dim> l;
    };

    SNCH_LBVH_CALLABLE query_line_intersect<float, 2> line_intersect(const line<float, 2> &l) noexcept
    {
        return query_line_intersect<float, 2>(l);
    }
    SNCH_LBVH_CALLABLE query_line_intersect<double, 2> line_intersect(const line<double, 2> &l) noexcept
    {
        return query_line_intersect<double, 2>(l);
    }
    SNCH_LBVH_CALLABLE query_line_intersect<float, 3> line_intersect(const line<float, 3> &l) noexcept
    {
        return query_line_intersect<float, 3>(l);
    }
    SNCH_LBVH_CALLABLE query_line_intersect<double, 3> line_intersect(const line<double, 3> &l) noexcept
    {
        return query_line_intersect<double, 3>(l);
    }

    template <typename Real, unsigned int dim, bool TestOnly>
    struct query_ray_intersect
    {
        using vector_type = typename vector_of<Real, dim>::type;

        SNCH_LBVH_HOST_DEVICE query_ray_intersect(const ray<Real, dim> &r, const float max_dist) : r(r), max_dist(max_dist) {}

        query_ray_intersect() = default;
        ~query_ray_intersect() = default;
        query_ray_intersect(const query_ray_intersect &) = default;
        query_ray_intersect(query_ray_intersect &&) = default;
        query_ray_intersect &operator=(const query_ray_intersect &) = default;
        query_ray_intersect &operator=(query_ray_intersect &&) = default;

        ray<Real, dim> r;
        float max_dist;
    };

    template <bool TestOnly = false>
    SNCH_LBVH_CALLABLE query_ray_intersect<float, 2, TestOnly> ray_intersect(const ray<float, 2> &r, const float max_dist) noexcept
    {
        return query_ray_intersect<float, 2, TestOnly>(r, max_dist);
    }
    template <bool TestOnly = false>
    SNCH_LBVH_CALLABLE query_ray_intersect<double, 2, TestOnly> ray_intersect(const ray<double, 2> &r, const float max_dist) noexcept
    {
        return query_ray_intersect<double, 2, TestOnly>(r, max_dist);
    }
    template <bool TestOnly = false>
    SNCH_LBVH_CALLABLE query_ray_intersect<float, 3, TestOnly> ray_intersect(const ray<float, 3> &r, const float max_dist) noexcept
    {
        return query_ray_intersect<float, 3, TestOnly>(r, max_dist);
    }
    template <bool TestOnly = false>
    SNCH_LBVH_CALLABLE query_ray_intersect<double, 3, TestOnly> ray_intersect(const ray<double, 3> &r, const float max_dist) noexcept
    {
        return query_ray_intersect<double, 3, TestOnly>(r, max_dist);
    }

    template <typename Real, unsigned int dim>
    struct query_sphere_intersect
    {
        using vector_type = typename vector_of<Real, dim>::type;

        SNCH_LBVH_HOST_DEVICE query_sphere_intersect(const sphere<Real, dim> &sph) : sph(sph) {}

        query_sphere_intersect() = default;
        ~query_sphere_intersect() = default;
        query_sphere_intersect(const query_sphere_intersect &) = default;
        query_sphere_intersect(query_sphere_intersect &&) = default;
        query_sphere_intersect &operator=(const query_sphere_intersect &) = default;
        query_sphere_intersect &operator=(query_sphere_intersect &&) = default;

        sphere<Real, dim> sph;
    };

    SNCH_LBVH_CALLABLE query_sphere_intersect<float, 2> sphere_intersect(const sphere<float, 2> &sphere) noexcept
    {
        return query_sphere_intersect<float, 2>(sphere);
    }
    SNCH_LBVH_CALLABLE query_sphere_intersect<double, 2> sphere_intersect(const sphere<double, 2> &sphere) noexcept
    {
        return query_sphere_intersect<double, 2>(sphere);
    }
    SNCH_LBVH_CALLABLE query_sphere_intersect<float, 3> sphere_intersect(const sphere<float, 3> &sphere) noexcept
    {
        return query_sphere_intersect<float, 3>(sphere);
    }
    SNCH_LBVH_CALLABLE query_sphere_intersect<double, 3> sphere_intersect(const sphere<double, 3> &sphere) noexcept
    {
        return query_sphere_intersect<double, 3>(sphere);
    }

    template <typename Real, unsigned int dim>
    struct query_overlap
    {
        SNCH_LBVH_HOST_DEVICE query_overlap(const aabb<Real, dim> &tgt) : target(tgt) {}

        query_overlap() = default;
        ~query_overlap() = default;
        query_overlap(const query_overlap &) = default;
        query_overlap(query_overlap &&) = default;
        query_overlap &operator=(const query_overlap &) = default;
        query_overlap &operator=(query_overlap &&) = default;

        SNCH_LBVH_CALLABLE bool operator()(const aabb<Real, dim> &box) noexcept { return intersects(box, target); }

        aabb<Real, dim> target;
    };

    template <typename Real, unsigned int dim>
    SNCH_LBVH_CALLABLE query_overlap<Real, dim> overlaps(const aabb<Real, dim> &region) noexcept
    {
        return query_overlap<Real, dim>(region);
    }

    template <typename Real, unsigned int dim>
    struct query_nearest
    {
        using vector_type = typename vector_of<Real, dim>::type;

        SNCH_LBVH_HOST_DEVICE query_nearest(const vector_type &tgt) : target(tgt) {}

        query_nearest() = default;
        ~query_nearest() = default;
        query_nearest(const query_nearest &) = default;
        query_nearest(query_nearest &&) = default;
        query_nearest &operator=(const query_nearest &) = default;
        query_nearest &operator=(query_nearest &&) = default;

        vector_type target;
    };

    SNCH_LBVH_CALLABLE query_nearest<float, 2> nearest(const float2 &point) noexcept
    {
        return query_nearest<float, 2>(point);
    }
    SNCH_LBVH_CALLABLE query_nearest<double, 2> nearest(const double2 &point) noexcept
    {
        return query_nearest<double, 2>(point);
    }

    SNCH_LBVH_CALLABLE query_nearest<float, 3> nearest(const float3 &point) noexcept
    {
        return query_nearest<float, 3>(point);
    }
    SNCH_LBVH_CALLABLE query_nearest<double, 3> nearest(const double3 &point) noexcept
    {
        return query_nearest<double, 3>(point);
    }

    template <typename Real, unsigned int dim>
    struct query_nearest_silhouette
    {
        using vector_type = typename vector_of<Real, dim>::type;

        SNCH_LBVH_HOST_DEVICE query_nearest_silhouette(const vector_type &tgt, const bool flip_normal_orientation) : target(tgt), flip_normal_orientation(flip_normal_orientation) {}

        query_nearest_silhouette() = default;
        ~query_nearest_silhouette() = default;
        query_nearest_silhouette(const query_nearest_silhouette &) = default;
        query_nearest_silhouette(query_nearest_silhouette &&) = default;
        query_nearest_silhouette &operator=(const query_nearest_silhouette &) = default;
        query_nearest_silhouette &operator=(query_nearest_silhouette &&) = default;

        vector_type target;
        bool flip_normal_orientation;
    };

    SNCH_LBVH_CALLABLE query_nearest_silhouette<float, 2> nearest_silhouette(const float2 &point, const bool flip_normal_orientation) noexcept
    {
        return query_nearest_silhouette<float, 2>(point, flip_normal_orientation);
    }
    SNCH_LBVH_CALLABLE query_nearest_silhouette<double, 2> nearest_silhouette(const double2 &point, const bool flip_normal_orientation) noexcept
    {
        return query_nearest_silhouette<double, 2>(point, flip_normal_orientation);
    }

    SNCH_LBVH_CALLABLE query_nearest_silhouette<float, 3> nearest_silhouette(const float3 &point, const bool flip_normal_orientation) noexcept
    {
        return query_nearest_silhouette<float, 3>(point, flip_normal_orientation);
    }
    SNCH_LBVH_CALLABLE query_nearest_silhouette<double, 3> nearest_silhouette(const double3 &point, const bool flip_normal_orientation) noexcept
    {
        return query_nearest_silhouette<double, 3>(point, flip_normal_orientation);
    }

} // namespace lbvh
#endif // LBVH_PREDICATOR_CUH
