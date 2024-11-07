#ifndef LBVH_AABB_CUH
#define LBVH_AABB_CUH
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
        __device__ __host__ aabb(typename vector_of<T, dim>::type upper, typename vector_of<T, dim>::type lower)
            : upper(upper), lower(lower) {}

        __device__ __host__ aabb(const typename vector_of<T, dim>::type &p)
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
    __device__ __host__ inline bool intersects(const aabb<T, 2> &lhs, const aabb<T, 2> &rhs) noexcept
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
    __device__ __host__ inline bool intersects(const aabb<T, 3> &lhs, const aabb<T, 3> &rhs) noexcept
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

    __device__ __host__ inline void expand_to_include(aabb<float, 2> *box, const float2 &p)
    {
        float2 p_lower = make_float2(p.x - epsilon<float>(), p.y - epsilon<float>());
        float2 p_upper = make_float2(p.x + epsilon<float>(), p.y + epsilon<float>());
        box->lower = cwisemin(box->lower, p_lower);
        box->upper = cwisemax(box->upper, p_upper);
    }
    __device__ __host__ inline void expand_to_include(aabb<float, 3> *box, const float3 &p)
    {
        float3 p_lower = make_float3(p.x - epsilon<float>(), p.y - epsilon<float>(), p.z - epsilon<float>());
        float3 p_upper = make_float3(p.x + epsilon<float>(), p.y + epsilon<float>(), p.z + epsilon<float>());
        box->lower = vec3_to_vec4(cwisemin(make_float3(box->lower.x, box->lower.y, box->lower.z), p_lower));
        box->upper = vec3_to_vec4(cwisemax(make_float3(box->upper.x, box->upper.y, box->upper.z), p_upper));
    }

    __device__ __host__ inline void expand_to_include(aabb<double, 2> *box, const double2 &p)
    {
        double2 p_lower = make_double2(p.x - epsilon<double>(), p.y - epsilon<double>());
        double2 p_upper = make_double2(p.x + epsilon<double>(), p.y + epsilon<double>());
        box->lower = cwisemin(box->lower, p_lower);
        box->upper = cwisemax(box->upper, p_upper);
    }
    __device__ __host__ inline void expand_to_include(aabb<double, 3> *box, const double3 &p)
    {
        double3 p_lower = make_double3(p.x - epsilon<double>(), p.y - epsilon<double>(), p.z - epsilon<double>());
        double3 p_upper = make_double3(p.x + epsilon<double>(), p.y + epsilon<double>(), p.z + epsilon<double>());
        box->lower = vec3_to_vec4(cwisemin(make_double3(box->lower.x, box->lower.y, box->lower.z), p_lower));
        box->upper = vec3_to_vec4(cwisemax(make_double3(box->upper.x, box->upper.y, box->upper.z), p_upper));
    }

    __device__ __host__ inline aabb<double, 2> merge(const aabb<double, 2> &lhs, const aabb<double, 2> &rhs) noexcept
    {
        aabb<double, 2> merged;
        merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
        merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
        merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
        merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
        return merged;
    }

    __device__ __host__ inline aabb<float, 2> merge(const aabb<float, 2> &lhs, const aabb<float, 2> &rhs) noexcept
    {
        aabb<float, 2> merged;
        merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
        merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
        merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
        merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
        return merged;
    }

    __device__ __host__ inline aabb<double, 3> merge(const aabb<double, 3> &lhs, const aabb<double, 3> &rhs) noexcept
    {
        aabb<double, 3> merged;
        merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
        merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
        merged.upper.z = ::fmax(lhs.upper.z, rhs.upper.z);
        merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
        merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
        merged.lower.z = ::fmin(lhs.lower.z, rhs.lower.z);
        return merged;
    }

    __device__ __host__ inline aabb<float, 3> merge(const aabb<float, 3> &lhs, const aabb<float, 3> &rhs) noexcept
    {
        aabb<float, 3> merged;
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

    __device__ __host__ inline float mindist(const aabb<float, 2> &lhs, const float2 &rhs) noexcept
    {
        const float dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
        const float dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
        return dx * dx + dy * dy;
    }

    __device__ __host__ inline double mindist(const aabb<double, 2> &lhs, const double2 &rhs) noexcept
    {
        const double dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
        const double dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
        return dx * dx + dy * dy;
    }

    __device__ __host__ inline float mindist(const aabb<float, 3> &lhs, const float4 &rhs) noexcept
    {
        const float dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
        const float dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
        const float dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ __host__ inline double mindist(const aabb<double, 3> &lhs, const double4 &rhs) noexcept
    {
        const double dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
        const double dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
        const double dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ __host__ inline float minmaxdist(const aabb<float, 2> &lhs, const float2 &rhs) noexcept
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

    __device__ __host__ inline float minmaxdist(const aabb<float, 3> &lhs, const float4 &rhs) noexcept
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

    __device__ __host__ inline double minmaxdist(const aabb<double, 2> &lhs, const double2 &rhs) noexcept
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

    __device__ __host__ inline double minmaxdist(const aabb<double, 3> &lhs, const double4 &rhs) noexcept
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
    __device__ __host__ inline typename vector_of<T, 2>::type centroid(const aabb<T, 2> &box) noexcept
    {
        typename vector_of<T, 2>::type c;
        c.x = (box.upper.x + box.lower.x) * 0.5;
        c.y = (box.upper.y + box.lower.y) * 0.5;
        return c;
    }

    template <typename T>
    __device__ __host__ inline typename vector_of<T, 3>::type centroid(const aabb<T, 3> &box) noexcept
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
        __host__ __device__
        Line(const typename vector_of<T, dim>::type &origin, const typename vector_of<T, dim>::type &dir)
            : origin(origin), dir(dir)
        {
            dir_inv.x = 1 / dir.x;
            dir_inv.y = 1 / dir.y;
            if constexpr (dim == 3)
                dir_inv.z = 1 / dir.z;
        };
    };

    // reference: https://tavianator.com/2015/ray_box_nan.html
    // Note we use fmin and fmax in the implementaitons below, which always suppress NaN whenever possible.
    template <typename T>
    __device__ __host__ inline bool intersects(const Line<T, 3> &line, const aabb<T, 3> &aabb) noexcept
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
    __device__ __host__ inline bool intersects(const Line<T, 2> &line, const aabb<T, 2> &aabb) noexcept
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
    __device__ __host__ inline T intersects_d(const Line<T, 2> &line, const aabb<T, 2> &aabb) noexcept
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
                return tmin;
            else
                return static_cast<T>(0);
        }
        else
        {
            return infinity<T>();
        }
    }

    template <typename T>
    __device__ __host__ inline T intersects_d(const Line<T, 3> &line, const aabb<T, 3> &aabb) noexcept
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
                return tmin;
            else
                return static_cast<T>(0);
        }
        else
        {
            return infinity<T>();
        }
    }
} // namespace lbvh

#endif // LBVH_AABB_CUH
