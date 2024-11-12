#pragma once

#include "utility.cuh"
#include <cmath>
#include <limits>
#include <thrust/swap.h>

namespace lbvh
{

    template <typename T, unsigned int dim>
    struct aabb
    {
        typename vector_of<T, dim>::type upper;
        typename vector_of<T, dim>::type lower;

        aabb() = default;
        SNCH_LBVH_HOST_DEVICE aabb(typename vector_of<T, dim>::type upper, typename vector_of<T, dim>::type lower)
            : upper(upper), lower(lower) {}

        SNCH_LBVH_HOST_DEVICE aabb(const typename vector_of<T, dim>::type &p)
        {
            if constexpr (dim == 2)
            {
                upper.x = p.x + epsilon<T>();
                upper.y = p.y + epsilon<T>();
                lower.x = p.x - epsilon<T>();
                lower.y = p.y - epsilon<T>();
            }
            else if constexpr (dim == 3)
            {
                upper.x = p.x + epsilon<T>();
                upper.y = p.y + epsilon<T>();
                upper.z = p.z + epsilon<T>();
                lower.x = p.x - epsilon<T>();
                lower.y = p.y - epsilon<T>();
                lower.z = p.z - epsilon<T>();
            }
            else if constexpr (dim == 4)
            {
                upper.x = p.x + epsilon<T>();
                upper.y = p.y + epsilon<T>();
                upper.z = p.z + epsilon<T>();
                upper.w = T(0);
                lower.x = p.x - epsilon<T>();
                lower.y = p.y - epsilon<T>();
                lower.z = p.z - epsilon<T>();
                lower.w = T(0);
            }
        }
    };

    template <typename T>
    SNCH_LBVH_CALLABLE bool intersects(const aabb<T, 2> &lhs, const aabb<T, 2> &rhs) noexcept
    {
        if (lhs.upper.x < rhs.lower.x || rhs.upper.x < lhs.lower.x)
        {
            return false;
        }
        if (lhs.upper.y < rhs.lower.y || rhs.upper.y < lhs.lower.y)
        {
            return false;
        }
        return true;
    }
    template <typename T>
    SNCH_LBVH_CALLABLE bool intersects(const aabb<T, 3> &lhs, const aabb<T, 3> &rhs) noexcept
    {
        if (lhs.upper.x < rhs.lower.x || rhs.upper.x < lhs.lower.x)
        {
            return false;
        }
        if (lhs.upper.y < rhs.lower.y || rhs.upper.y < lhs.lower.y)
        {
            return false;
        }
        if (lhs.upper.z < rhs.lower.z || rhs.upper.z < lhs.lower.z)
        {
            return false;
        }
        return true;
    }

    template <typename T>
    SNCH_LBVH_CALLABLE void expand_to_include(aabb<T, 2> *box, const typename vector_of<T, 2>::type &p)
    {
        using vector_type = typename vector_of<T, 2>::type;
        vector_type p_lower = {p.x - epsilon<float>(), p.y - epsilon<float>()};
        vector_type p_upper = {p.x + epsilon<float>(), p.y + epsilon<float>()};
        box->lower = cwisemin(box->lower, p_lower);
        box->upper = cwisemax(box->upper, p_upper);
    }
    template <typename T>
    SNCH_LBVH_CALLABLE void expand_to_include(aabb<T, 3> *box, const typename vector_of<T, 3>::type &p)
    {
        using vector_type = typename vector_of<T, 3>::type;
        vector_type p_lower = {p.x - epsilon<float>(), p.y - epsilon<float>(), p.z - epsilon<float>()};
        vector_type p_upper = {p.x + epsilon<float>(), p.y + epsilon<float>(), p.z + epsilon<float>()};
        box->lower = cwisemin({box->lower.x, box->lower.y, box->lower.z}, p_lower);
        box->upper = cwisemax({box->upper.x, box->upper.y, box->upper.z}, p_upper);
    }

    template <typename T>
    SNCH_LBVH_CALLABLE aabb<T, 2> merge(const aabb<T, 2> &lhs, const aabb<T, 2> &rhs) noexcept
    {
        aabb<T, 2> merged;
        merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
        merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
        merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
        merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
        return merged;
    }
    template <typename T>
    SNCH_LBVH_CALLABLE aabb<T, 3> merge(const aabb<T, 3> &lhs, const aabb<T, 3> &rhs) noexcept
    {
        aabb<T, 3> merged;
        merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
        merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
        merged.upper.z = ::fmax(lhs.upper.z, rhs.upper.z);
        merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
        merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
        merged.lower.z = ::fmin(lhs.lower.z, rhs.lower.z);
        return merged;
    }

    // metrics defined in
    // Nearest Neighbor Queries (1995) ACS-SIGMOD
    // - Nick Roussopoulos, Stephen Kelley FredericVincent

    SNCH_LBVH_CALLABLE float mindist(const aabb<float, 2> &lhs, const float2 &rhs) noexcept
    {
        const float dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
        const float dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
        return dx * dx + dy * dy;
    }

    SNCH_LBVH_CALLABLE double mindist(const aabb<double, 2> &lhs, const double2 &rhs) noexcept
    {
        const double dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
        const double dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
        return dx * dx + dy * dy;
    }

    SNCH_LBVH_CALLABLE float mindist(const aabb<float, 3> &lhs, const float4 &rhs) noexcept
    {
        const float dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
        const float dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
        const float dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    }

    SNCH_LBVH_CALLABLE double mindist(const aabb<double, 3> &lhs, const double4 &rhs) noexcept
    {
        const double dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
        const double dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
        const double dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    }

    SNCH_LBVH_CALLABLE float minmaxdist(const aabb<float, 2> &lhs, const float2 &rhs) noexcept
    {
        float2 rm_sq =
            make_float2((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x), (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y));
        float2 rM_sq =
            make_float2((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x), (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y));

        if ((lhs.upper.x + lhs.lower.x) * 0.5f < rhs.x)
        {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if ((lhs.upper.y + lhs.lower.y) * 0.5f < rhs.y)
        {
            thrust::swap(rm_sq.y, rM_sq.y);
        }

        const float dx = rm_sq.x + rM_sq.y;
        const float dy = rM_sq.x + rm_sq.y;

        return ::fmin(dx, dy);
    }

    SNCH_LBVH_CALLABLE float minmaxdist(const aabb<float, 3> &lhs, const float4 &rhs) noexcept
    {
        float3 rm_sq = make_float3(
            (lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x), (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
            (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z));
        float3 rM_sq = make_float3(
            (lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x), (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
            (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z));

        if ((lhs.upper.x + lhs.lower.x) * 0.5f < rhs.x)
        {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if ((lhs.upper.y + lhs.lower.y) * 0.5f < rhs.y)
        {
            thrust::swap(rm_sq.y, rM_sq.y);
        }
        if ((lhs.upper.z + lhs.lower.z) * 0.5f < rhs.z)
        {
            thrust::swap(rm_sq.z, rM_sq.z);
        }

        const float dx = rm_sq.x + rM_sq.y + rM_sq.z;
        const float dy = rM_sq.x + rm_sq.y + rM_sq.z;
        const float dz = rM_sq.x + rM_sq.y + rm_sq.z;

        return ::fmin(dx, ::fmin(dy, dz));
    }

    SNCH_LBVH_CALLABLE double minmaxdist(const aabb<double, 2> &lhs, const double2 &rhs) noexcept
    {
        double2 rm_sq =
            make_double2((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x), (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y));
        double2 rM_sq =
            make_double2((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x), (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y));

        if ((lhs.upper.x + lhs.lower.x) * 0.5 < rhs.x)
        {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if ((lhs.upper.y + lhs.lower.y) * 0.5 < rhs.y)
        {
            thrust::swap(rm_sq.y, rM_sq.y);
        }

        const double dx = rm_sq.x + rM_sq.y;
        const double dy = rM_sq.x + rm_sq.y;

        return ::fmin(dx, dy);
    }

    SNCH_LBVH_CALLABLE double minmaxdist(const aabb<double, 3> &lhs, const double4 &rhs) noexcept
    {
        double3 rm_sq = make_double3(
            (lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x), (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
            (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z));
        double3 rM_sq = make_double3(
            (lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x), (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
            (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z));

        if ((lhs.upper.x + lhs.lower.x) * 0.5 < rhs.x)
        {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if ((lhs.upper.y + lhs.lower.y) * 0.5 < rhs.y)
        {
            thrust::swap(rm_sq.y, rM_sq.y);
        }
        if ((lhs.upper.z + lhs.lower.z) * 0.5 < rhs.z)
        {
            thrust::swap(rm_sq.z, rM_sq.z);
        }

        const double dx = rm_sq.x + rM_sq.y + rM_sq.z;
        const double dy = rM_sq.x + rm_sq.y + rM_sq.z;
        const double dz = rM_sq.x + rM_sq.y + rm_sq.z;

        return ::fmin(dx, ::fmin(dy, dz));
    }

    template <typename T>
    SNCH_LBVH_CALLABLE typename vector_of<T, 2>::type centroid(const aabb<T, 2> &box) noexcept
    {
        typename vector_of<T, 2>::type c;
        c.x = (box.upper.x + box.lower.x) * 0.5;
        c.y = (box.upper.y + box.lower.y) * 0.5;
        return c;
    }

    template <typename T>
    SNCH_LBVH_CALLABLE typename vector_of<T, 3>::type centroid(const aabb<T, 3> &box) noexcept
    {
        typename vector_of<T, 3>::type c;
        c.x = (box.upper.x + box.lower.x) * 0.5;
        c.y = (box.upper.y + box.lower.y) * 0.5;
        c.z = (box.upper.z + box.lower.z) * 0.5;
        return c;
    }

    template <typename T, unsigned int dim>
    struct Line
    {
        typename vector_of<T, dim>::type origin;
        typename vector_of<T, dim>::type dir;
        typename vector_of<T, dim>::type dir_inv;
        SNCH_LBVH_HOST_DEVICE
        Line(const typename vector_of<T, dim>::type &origin, const typename vector_of<T, dim>::type &dir)
            : origin(origin), dir(dir)
        {
            dir_inv.x = 1 / dir.x;
            dir_inv.y = 1 / dir.y;
            if constexpr (dim == 3)
                dir_inv.z = 1 / dir.z;
        }
    };

    template <typename T, unsigned int dim>
    struct sphere
    {
        typename vector_of<T, dim>::type origin;
        float radius;
        SNCH_LBVH_HOST_DEVICE
        sphere(const typename vector_of<T, dim>::type &origin, const float radius)
            : origin(origin), radius(radius)
        {
        }
    };

    // reference: https://tavianator.com/2015/ray_box_nan.html
    // Note we use fmin and fmax in the implementaitons below, which always suppress NaN whenever possible.
    template <typename T>
    SNCH_LBVH_CALLABLE bool intersects(const Line<T, 3> &line, const aabb<T, 3> &aabb) noexcept
    {
        T t1 = (aabb.lower.x - line.origin.x) * line.dir_inv.x;
        T t2 = (aabb.upper.x - line.origin.x) * line.dir_inv.x;
        T tmin = fmin(t1, t2);
        T tmax = fmax(t1, t2);

        t1 = (aabb.lower.y - line.origin.y) * line.dir_inv.y;
        t2 = (aabb.upper.y - line.origin.y) * line.dir_inv.y;
        tmin = fmax(tmin, fmin(t1, t2));
        tmax = fmin(tmax, fmax(t1, t2));

        t1 = (aabb.lower.z - line.origin.z) * line.dir_inv.z;
        t2 = (aabb.upper.z - line.origin.z) * line.dir_inv.z;
        tmin = fmax(tmin, fmin(t1, t2));
        tmax = fmin(tmax, fmax(t1, t2));

        return tmax >= tmin; // we should not add "&& tmax > 0" becasue this is line intersection, not ray intersection.
    }

    template <typename T>
    SNCH_LBVH_CALLABLE bool intersects(const Line<T, 2> &line, const aabb<T, 2> &aabb) noexcept
    {
        T t1 = (aabb.lower.x - line.origin.x) * line.dir_inv.x;
        T t2 = (aabb.upper.x - line.origin.x) * line.dir_inv.x;
        T tmin = fmin(t1, t2);
        T tmax = fmax(t1, t2);

        t1 = (aabb.lower.y - line.origin.y) * line.dir_inv.y;
        t2 = (aabb.upper.y - line.origin.y) * line.dir_inv.y;
        tmin = fmax(tmin, fmin(t1, t2));
        tmax = fmin(tmax, fmax(t1, t2));

        return tmax >= tmin; // we should not add "&& tmax > 0" becasue this is line intersection, not ray intersection.
    }

    template <typename T>
    SNCH_LBVH_CALLABLE bool intersects_d(const Line<T, 2> &line, const aabb<T, 2> &aabb, T *distance) noexcept
    {
        T t1 = (aabb.lower.x - line.origin.x) * line.dir_inv.x;
        T t2 = (aabb.upper.x - line.origin.x) * line.dir_inv.x;
        T tmin = std::fmin(t1, t2);
        T tmax = std::fmax(t1, t2);

        t1 = (aabb.lower.y - line.origin.y) * line.dir_inv.y;
        t2 = (aabb.upper.y - line.origin.y) * line.dir_inv.y;
        tmin = std::fmax(tmin, std::fmin(t1, t2));
        tmax = std::fmin(tmax, std::fmax(t1, t2));

        if (tmax >= tmin && tmax >= static_cast<T>(0))
        {
            if (tmin >= static_cast<T>(0))
            {
                *distance = tmin;
            }
            else
            {
                *distance = static_cast<T>(0);
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename T>
    SNCH_LBVH_CALLABLE bool intersects_d(const Line<T, 3> &line, const aabb<T, 3> &aabb, T *distance) noexcept
    {
        T t1 = (aabb.lower.x - line.origin.x) * line.dir_inv.x;
        T t2 = (aabb.upper.x - line.origin.x) * line.dir_inv.x;
        T tmin = std::fmin(t1, t2);
        T tmax = std::fmax(t1, t2);

        t1 = (aabb.lower.y - line.origin.y) * line.dir_inv.y;
        t2 = (aabb.upper.y - line.origin.y) * line.dir_inv.y;
        tmin = std::fmax(tmin, std::fmin(t1, t2));
        tmax = std::fmin(tmax, std::fmax(t1, t2));

        t1 = (aabb.lower.z - line.origin.z) * line.dir_inv.z;
        t2 = (aabb.upper.z - line.origin.z) * line.dir_inv.z;
        tmin = std::fmax(tmin, std::fmin(t1, t2));
        tmax = std::fmin(tmax, std::fmax(t1, t2));

        if (tmax >= tmin && tmax >= static_cast<T>(0))
        {
            if (tmin >= static_cast<T>(0))
            {
                *distance = tmin;
            }
            else
            {
                *distance = static_cast<T>(0);
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename T, unsigned int dim>
    SNCH_LBVH_CALLABLE bool intersect_sphere(const sphere<T, dim> &sph, const aabb<T, dim> &aabb) noexcept
    {
        const T closest_x = std::max(aabb.lower.x, std::min(sph.origin.x, aabb.upper.x));
        const T closest_y = std::max(aabb.lower.y, std::min(sph.origin.y, aabb.upper.y));
        const T dx = closest_x - sph.origin.x;
        const T dy = closest_y - sph.origin.y;
        T distance_squared = dx * dx + dy * dy;
        if constexpr (dim == 3)
        {
            const T closest_z = std::max(aabb.lower.z, std::min(sph.origin.z, aabb.upper.z));
            const T dz = closest_z - sph.origin.z;
            distance_squared += dz * dz;
        }

        return distance_squared <= (sph.radius * sph.radius);
    }
} // namespace lbvh
