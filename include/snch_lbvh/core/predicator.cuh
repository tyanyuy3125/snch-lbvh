#ifndef LBVH_PREDICATOR_CUH
#define LBVH_PREDICATOR_CUH
#include "aabb.cuh"

namespace lbvh
{

    template <typename Real, unsigned int dim>
    struct query_line_intersect
    {
        using vector_type = typename vector_of<Real, dim>::type;

        SNCH_LBVH_HOST_DEVICE query_line_intersect(const Line<Real, dim> &line) : line(line) {}

        query_line_intersect() = default;
        ~query_line_intersect() = default;
        query_line_intersect(const query_line_intersect &) = default;
        query_line_intersect(query_line_intersect &&) = default;
        query_line_intersect &operator=(const query_line_intersect &) = default;
        query_line_intersect &operator=(query_line_intersect &&) = default;

        Line<Real, dim> line;
    };

    SNCH_LBVH_CALLABLE query_line_intersect<float, 2> line_intersect(const Line<float, 2> &line) noexcept
    {
        return query_line_intersect<float, 2>(line);
    }
    SNCH_LBVH_CALLABLE query_line_intersect<double, 2> line_intersect(const Line<double, 2> &line) noexcept
    {
        return query_line_intersect<double, 2>(line);
    }
    SNCH_LBVH_CALLABLE query_line_intersect<float, 3> line_intersect(const Line<float, 3> &line) noexcept
    {
        return query_line_intersect<float, 3>(line);
    }
    SNCH_LBVH_CALLABLE query_line_intersect<double, 3> line_intersect(const Line<double, 3> &line) noexcept
    {
        return query_line_intersect<double, 3>(line);
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

    SNCH_LBVH_CALLABLE query_nearest<float, 3> nearest(const float4 &point) noexcept
    {
        return query_nearest<float, 3>(point);
    }
    SNCH_LBVH_CALLABLE query_nearest<float, 3> nearest(const float3 &point) noexcept
    {
        return query_nearest<float, 3>(make_float4(point.x, point.y, point.z, 0.0f));
    }
    SNCH_LBVH_CALLABLE query_nearest<double, 3> nearest(const double4 &point) noexcept
    {
        return query_nearest<double, 3>(point);
    }
    SNCH_LBVH_CALLABLE query_nearest<double, 3> nearest(const double3 &point) noexcept
    {
        return query_nearest<double, 3>(make_double4(point.x, point.y, point.z, 0.0));
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

    SNCH_LBVH_CALLABLE query_nearest_silhouette<float, 3> nearest_silhouette(const float4 &point, const bool flip_normal_orientation) noexcept
    {
        return query_nearest_silhouette<float, 3>(point, flip_normal_orientation);
    }
    SNCH_LBVH_CALLABLE query_nearest_silhouette<float, 3> nearest_silhouette(const float3 &point, const bool flip_normal_orientation) noexcept
    {
        return query_nearest_silhouette<float, 3>(make_float4(point.x, point.y, point.z, 0.0f), flip_normal_orientation);
    }
    SNCH_LBVH_CALLABLE query_nearest_silhouette<double, 3> nearest_silhouette(const double4 &point, const bool flip_normal_orientation) noexcept
    {
        return query_nearest_silhouette<double, 3>(point, flip_normal_orientation);
    }
    SNCH_LBVH_CALLABLE query_nearest_silhouette<double, 3> nearest_silhouette(const double3 &point, const bool flip_normal_orientation) noexcept
    {
        return query_nearest_silhouette<double, 3>(make_double4(point.x, point.y, point.z, 0.0), flip_normal_orientation);
    }

} // namespace lbvh
#endif // LBVH_PREDICATOR_CUH
