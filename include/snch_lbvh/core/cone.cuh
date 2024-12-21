#ifndef LBVH_CONE_CUH
#define LBVH_CONE_CUH
#include "aabb.cuh"

#include <thrust/swap.h>

namespace lbvh
{

    template <typename T, unsigned int dim>
    struct cone
    {
        typename vector_of<T, dim>::type axis;
        T half_angle;
        T radius;
    };

    template <typename T, unsigned int dim>
    SNCH_LBVH_CALLABLE bool is_valid(const cone<T, dim> &c) noexcept
    {
        return c.half_angle >= T(0);
    }

    SNCH_LBVH_CALLABLE void compute_orthonormal_basis(const double3 &n, double3 *b1, double3 *b2) noexcept
    {
        double sign = std::copysignf(1.0f, n.z);
        const double a = -1.0 / (sign + n.z);
        const double b = n.x * n.y * a;

        *b1 = make_double3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        *b2 = make_double3(b, sign + n.y * n.y * a, -n.y);
    }

    SNCH_LBVH_CALLABLE void compute_orthonormal_basis(const float3 &n, float3 *b1, float3 *b2) noexcept
    {
        float sign = std::copysignf(1.0f, n.z);
        const float a = -1.0f / (sign + n.z);
        const float b = n.x * n.y * a;

        *b1 = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
        *b2 = make_float3(b, sign + n.y * n.y * a, -n.y);
    }

    SNCH_LBVH_CALLABLE float project_to_plane(const float2 &n, const float2 &e) noexcept
    {
        float2 b = make_float2(-n.y, n.x);
        float r = dot(e, cwiseabs(b));
        return std::abs(r);
    }

    SNCH_LBVH_CALLABLE double project_to_plane(const double2 &n, const double2 &e) noexcept
    {
        double2 b = make_double2(-n.y, n.x);
        double r = dot(e, cwiseabs(b));
        return std::abs(r);
    }

    SNCH_LBVH_CALLABLE float project_to_plane(const float3 &n, const float3 &e) noexcept
    {
        float3 b1, b2;
        compute_orthonormal_basis(n, &b1, &b2);

        float r1 = dot(e, cwiseabs(b1));
        float r2 = dot(e, cwiseabs(b2));
        return std::sqrt(r1 * r1 + r2 * r2);
    }

    SNCH_LBVH_CALLABLE double project_to_plane(const double3 &n, const double3 &e) noexcept
    {
        double3 b1, b2;
        compute_orthonormal_basis(n, &b1, &b2);

        double r1 = dot(e, cwiseabs(b1));
        double r2 = dot(e, cwiseabs(b2));
        return std::sqrt(r1 * r1 + r2 * r2);
    }

    SNCH_LBVH_CALLABLE bool overlap(const cone<float, 2> &bc, const float2 &o, const aabb<float, 2> &b, const float dist_to_box, float *min_angle_range, float *max_angle_range) noexcept
    {
        *min_angle_range = 0.0f;
        *max_angle_range = static_cast<float>(M_PI_2);

        if (bc.half_angle >= static_cast<float>(M_PI_2) || dist_to_box < epsilon<float>())
        {
            return true;
        }

        float2 c = centroid(b);
        float2 view_cone_axis = make_float2(c.x - o.x, c.y - o.y);
        float l = length(view_cone_axis);
        view_cone_axis.x /= l;
        view_cone_axis.y /= l;

        float d_axis_angle = std::acos(std::max(-1.0f, std::min(1.0f, dot(bc.axis, view_cone_axis))));
        if (inrange(static_cast<float>(M_PI_2), d_axis_angle - bc.half_angle, d_axis_angle + bc.half_angle))
        {
            return true;
        }

        if (l > bc.radius)
        {
            float view_cone_half_angle = std::asin(bc.radius / l);
            float half_angle_sum = bc.half_angle + view_cone_half_angle;
            *min_angle_range = d_axis_angle - half_angle_sum;
            *max_angle_range = d_axis_angle + half_angle_sum;
            return half_angle_sum >= static_cast<float>(M_PI_2) ? true : inrange(static_cast<float>(M_PI_2), *min_angle_range, *max_angle_range);
        }

        float2 e = make_float2(b.upper.x - c.x, b.upper.y - c.y);
        float d = dot(e, cwiseabs(view_cone_axis));
        float s = l - d;
        if (s <= 0.0f)
            return true;

        d = project_to_plane(view_cone_axis, e);
        float view_cone_half_angle = std::atan2(d, s);
        float half_angle_sum = bc.half_angle + view_cone_half_angle;
        *min_angle_range = d_axis_angle - half_angle_sum;
        *max_angle_range = d_axis_angle + half_angle_sum;
        return half_angle_sum >= static_cast<float>(M_PI_2) ? true : inrange(static_cast<float>(M_PI_2), *min_angle_range, *max_angle_range);
    }

    SNCH_LBVH_CALLABLE bool overlap(const cone<double, 2> &bc, const double2 &o, const aabb<double, 2> &b, const double dist_to_box, double *min_angle_range, double *max_angle_range) noexcept
    {
        *min_angle_range = 0.0;
        *max_angle_range = M_PI_2;

        if (bc.half_angle >= M_PI_2 || dist_to_box < epsilon<double>())
        {
            return true;
        }

        double2 c = centroid(b);
        double2 view_cone_axis = make_double2(c.x - o.x, c.y - o.y);
        float l = length(view_cone_axis);
        view_cone_axis.x /= l;
        view_cone_axis.y /= l;

        float d_axis_angle = std::acos(std::max(-1.0, std::min(1.0, dot(bc.axis, view_cone_axis))));
        if (inrange(M_PI_2, d_axis_angle - bc.half_angle, d_axis_angle + bc.half_angle))
        {
            return true;
        }

        if (l > bc.radius)
        {
            float view_cone_half_angle = std::asin(bc.radius / l);
            float half_angle_sum = bc.half_angle + view_cone_half_angle;
            *min_angle_range = d_axis_angle - half_angle_sum;
            *max_angle_range = d_axis_angle + half_angle_sum;
            return half_angle_sum >= M_PI_2 ? true : inrange(M_PI_2, *min_angle_range, *max_angle_range);
        }

        double2 e = make_double2(b.upper.x - c.x, b.upper.y - c.y);
        float d = dot(e, cwiseabs(view_cone_axis));
        float s = l - d;
        if (s <= 0.0)
            return true;

        d = project_to_plane(view_cone_axis, e);
        float view_cone_half_angle = std::atan2(d, s);
        float half_angle_sum = bc.half_angle + view_cone_half_angle;
        *min_angle_range = d_axis_angle - half_angle_sum;
        *max_angle_range = d_axis_angle + half_angle_sum;
        return half_angle_sum >= M_PI_2 ? true : inrange(M_PI_2, *min_angle_range, *max_angle_range);
    }

    SNCH_LBVH_CALLABLE bool overlap(const cone<float, 3> &bc, const float3 &o, const aabb<float, 3> &b, const float dist_to_box, float *min_angle_range, float *max_angle_range) noexcept
    {
        *min_angle_range = 0.0f;
        *max_angle_range = static_cast<float>(M_PI_2);

        if (bc.half_angle >= static_cast<float>(M_PI_2) || dist_to_box < epsilon<float>())
        {
            return true;
        }

        auto c = centroid(b);
        float3 view_cone_axis = make_float3(c.x - o.x, c.y - o.y, c.z - o.z);
        float l = length(view_cone_axis);
        view_cone_axis.x /= l;
        view_cone_axis.y /= l;
        view_cone_axis.z /= l;

        float d_axis_angle = std::acos(std::max(-1.0f, std::min(1.0f, dot(make_float3(bc.axis.x, bc.axis.y, bc.axis.z), view_cone_axis))));
        if (inrange(static_cast<float>(M_PI_2), d_axis_angle - bc.half_angle, d_axis_angle + bc.half_angle))
        {
            return true;
        }

        if (l > bc.radius)
        {
            float view_cone_half_angle = std::asin(bc.radius / l);
            float half_angle_sum = bc.half_angle + view_cone_half_angle;
            *min_angle_range = d_axis_angle - half_angle_sum;
            *max_angle_range = d_axis_angle + half_angle_sum;
            return half_angle_sum >= static_cast<float>(M_PI_2) ? true : inrange(static_cast<float>(M_PI_2), *min_angle_range, *max_angle_range);
        }

        float3 e = make_float3(b.upper.x - c.x, b.upper.y - c.y, b.upper.z - c.z);
        float d = dot(e, cwiseabs(view_cone_axis));
        float s = l - d;
        if (s <= 0.0f)
            return true;

        d = project_to_plane(view_cone_axis, e);
        float view_cone_half_angle = std::atan2(d, s);
        float half_angle_sum = bc.half_angle + view_cone_half_angle;
        *min_angle_range = d_axis_angle - half_angle_sum;
        *max_angle_range = d_axis_angle + half_angle_sum;
        return half_angle_sum >= static_cast<float>(M_PI_2) ? true : inrange(static_cast<float>(M_PI_2), *min_angle_range, *max_angle_range);
    }

    SNCH_LBVH_CALLABLE bool overlap(const cone<double, 3> &bc, const double3 &o, const aabb<double, 3> &b, const double dist_to_box, double *min_angle_range, double *max_angle_range) noexcept
    {
        *min_angle_range = 0.0;
        *max_angle_range = M_PI_2;

        if (bc.half_angle >= M_PI_2 || dist_to_box < epsilon<double>())
        {
            return true;
        }

        auto c = centroid(b);
        // float3 view_cone_axis = c - o;
        double3 view_cone_axis = make_double3(c.x - o.x, c.y - o.y, c.z - o.z);
        // float l = norm(3, &view_cone_axis);
        double l = length(view_cone_axis);
        // view_cone_axis /= l;
        view_cone_axis.x /= l;
        view_cone_axis.y /= l;
        view_cone_axis.z /= l;

        double d_axis_angle = std::acos(std::max(-1.0, std::min(1.0, dot(make_double3(bc.axis.x, bc.axis.y, bc.axis.z), view_cone_axis))));
        if (inrange(M_PI_2, d_axis_angle - bc.half_angle, d_axis_angle + bc.half_angle))
        {
            return true;
        }

        if (l > bc.radius)
        {
            double view_cone_half_angle = std::asin(bc.radius / l);
            double half_angle_sum = bc.half_angle + view_cone_half_angle;
            *min_angle_range = d_axis_angle - half_angle_sum;
            *max_angle_range = d_axis_angle + half_angle_sum;
            return half_angle_sum >= M_PI_2 ? true : inrange(M_PI_2, *min_angle_range, *max_angle_range);
        }

        double3 e = make_double3(b.upper.x - c.x, b.upper.y - c.y, b.upper.z - c.z);
        // e.x = b.upper.x - c.x;
        // e.y = b.upper.y - c.y;
        // e.z = b.upper.z - c.z;
        double d = dot(e, cwiseabs(view_cone_axis));
        double s = l - d;
        if (s <= 0.0)
            return true;

        d = project_to_plane(view_cone_axis, e);
        double view_cone_half_angle = std::atan2(d, s);
        double half_angle_sum = bc.half_angle + view_cone_half_angle;
        *min_angle_range = d_axis_angle - half_angle_sum;
        *max_angle_range = d_axis_angle + half_angle_sum;
        return half_angle_sum >= M_PI_2 ? true : inrange(M_PI_2, *min_angle_range, *max_angle_range);
    }

    SNCH_LBVH_CALLABLE float2 rotate(const float2 &u, const float2 &v, float theta)
    {
        float det = u.x * v.y - u.y * v.x;
        theta *= std::copysign(1.0f, det);
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        return make_float2(cos_theta * u.x - sin_theta * u.y,
                           sin_theta * u.x + cos_theta * u.y);
    }

    SNCH_LBVH_CALLABLE double2 rotate(const double2 &u, const double2 &v, double theta)
    {
        double det = u.x * v.y - u.y * v.x;
        theta *= std::copysign(1.0, det);
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);

        return make_double2(cos_theta * u.x - sin_theta * u.y,
                            sin_theta * u.x + cos_theta * u.y);
    }

    SNCH_LBVH_CALLABLE float3 rotate(const float3 &u, const float3 &v, float theta)
    {
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);
        float3 w = normalize(cross(u, v));
        float3 one_minus_cos_theta_w = make_float3((1.0f - cos_theta) * w.x, (1.0f - cos_theta) * w.y, (1.0f - cos_theta) * w.z);
        float R[3][3] = {
            {cos_theta + one_minus_cos_theta_w.x * w.x, one_minus_cos_theta_w.y * w.x - sin_theta * w.z, one_minus_cos_theta_w.z * w.x + sin_theta * w.y},
            {one_minus_cos_theta_w.x * w.y + sin_theta * w.z, cos_theta + one_minus_cos_theta_w.y * w.y, one_minus_cos_theta_w.z * w.y - sin_theta * w.x},
            {one_minus_cos_theta_w.x * w.z - sin_theta * w.y, one_minus_cos_theta_w.y * w.z + sin_theta * w.x, cos_theta + one_minus_cos_theta_w.z * w.z}};
        return make_float3(
            R[0][0] * u.x + R[0][1] * u.y + R[0][2] * u.z,
            R[1][0] * u.x + R[1][1] * u.y + R[1][2] * u.z,
            R[2][0] * u.x + R[2][1] * u.y + R[2][2] * u.z);
    }

    SNCH_LBVH_CALLABLE double3 rotate(const double3 &u, const double3 &v, double theta)
    {
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        double3 w = normalize(cross(u, v));
        double3 one_minus_cos_theta_w = make_double3((1.0 - cos_theta) * w.x, (1.0 - cos_theta) * w.y, (1.0 - cos_theta) * w.z);
        double R[3][3] = {
            {cos_theta + one_minus_cos_theta_w.x * w.x, one_minus_cos_theta_w.y * w.x - sin_theta * w.z, one_minus_cos_theta_w.z * w.x + sin_theta * w.y},
            {one_minus_cos_theta_w.x * w.y + sin_theta * w.z, cos_theta + one_minus_cos_theta_w.y * w.y, one_minus_cos_theta_w.z * w.y - sin_theta * w.x},
            {one_minus_cos_theta_w.x * w.z - sin_theta * w.y, one_minus_cos_theta_w.y * w.z + sin_theta * w.x, cos_theta + one_minus_cos_theta_w.z * w.z}};
        return make_double3(
            R[0][0] * u.x + R[0][1] * u.y + R[0][2] * u.z,
            R[1][0] * u.x + R[1][1] * u.y + R[1][2] * u.z,
            R[2][0] * u.x + R[2][1] * u.y + R[2][2] * u.z);
    }

    SNCH_LBVH_CALLABLE cone<float, 2> merge(const cone<float, 2> &cone_a, const cone<float, 2> &cone_b, const float2 &origin_a, const float2 &origin_b, const float2 &new_origin) noexcept
    {
        cone<float, 2> ret;
        if (is_valid(cone_a) && is_valid(cone_b))
        {
            float2 axis_a = cone_a.axis;
            float2 axis_b = cone_b.axis;
            float half_angle_a = cone_a.half_angle;
            float half_angle_b = cone_b.half_angle;
            float2 d_origin_a = make_float2(new_origin.x - origin_a.x, new_origin.y - origin_a.y);
            float2 d_origin_b = make_float2(new_origin.x - origin_b.x, new_origin.y - origin_b.y);
            ret.radius = std::sqrt(std::max(cone_a.radius * cone_a.radius + squared_length(d_origin_a), cone_b.radius * cone_b.radius + squared_length(d_origin_b)));

            if (half_angle_b > half_angle_a)
            {
                thrust::swap(axis_a, axis_b);
                thrust::swap(half_angle_a, half_angle_b);
            }

            float theta = std::acos(std::max(-1.0f, std::min(1.0f, dot(axis_a, axis_b))));
            if (std::min(theta + half_angle_b, static_cast<float>(M_PI)) <= half_angle_a)
            {
                ret.axis = axis_a;
                ret.half_angle = half_angle_a;
                return ret;
            }

            float o_theta = (half_angle_a + theta + half_angle_b) / 2.0f;
            if (o_theta >= static_cast<float>(M_PI))
            {
                ret.axis = axis_a;
                return ret;
            }

            float r_theta = o_theta - half_angle_a;
            ret.axis = rotate(axis_a, axis_b, r_theta);
            ret.half_angle = o_theta;
        }
        else if (is_valid(cone_a))
        {
            ret = cone_a;
        }
        else if (is_valid(cone_b))
        {
            ret = cone_b;
        }
        else
        {
            ret.half_angle = -static_cast<float>(M_PI);
        }

        return ret;
    }

    SNCH_LBVH_CALLABLE cone<double, 2> merge(const cone<double, 2> &cone_a, const cone<double, 2> &cone_b, const double2 &origin_a, const double2 &origin_b, const double2 &new_origin) noexcept
    {
        cone<double, 2> ret;
        if (is_valid(cone_a) && is_valid(cone_b))
        {
            double2 axis_a = cone_a.axis;
            double2 axis_b = cone_b.axis;
            double half_angle_a = cone_a.half_angle;
            double half_angle_b = cone_b.half_angle;
            double2 d_origin_a = make_double2(new_origin.x - origin_a.x, new_origin.y - origin_a.y);
            double2 d_origin_b = make_double2(new_origin.x - origin_b.x, new_origin.y - origin_b.y);
            ret.radius = std::sqrt(std::max(cone_a.radius * cone_a.radius + squared_length(d_origin_a), cone_b.radius * cone_b.radius + squared_length(d_origin_b)));

            if (half_angle_b > half_angle_a)
            {
                thrust::swap(axis_a, axis_b);
                thrust::swap(half_angle_a, half_angle_b);
            }

            double theta = std::acos(std::max(-1.0, std::min(1.0, dot(axis_a, axis_b))));
            if (std::min(theta + half_angle_b, M_PI) <= half_angle_a)
            {
                ret.axis = axis_a;
                ret.half_angle = half_angle_a;
                return ret;
            }

            double o_theta = (half_angle_a + theta + half_angle_b) / 2.0;
            if (o_theta >= M_PI)
            {
                ret.axis = axis_a;
                return ret;
            }

            double r_theta = o_theta - half_angle_a;
            ret.axis = rotate(axis_a, axis_b, r_theta);
            ret.half_angle = o_theta;
        }
        else if (is_valid(cone_a))
        {
            ret = cone_a;
        }
        else if (is_valid(cone_b))
        {
            ret = cone_b;
        }
        else
        {
            ret.half_angle = -M_PI;
        }
        return ret;
    }

    SNCH_LBVH_CALLABLE cone<float, 3> merge(const cone<float, 3> &cone_a, const cone<float, 3> &cone_b, const float3 &origin_a, const float3 &origin_b, const float3 &new_origin) noexcept
    {
        cone<float, 3> ret;
        if (is_valid(cone_a) && is_valid(cone_b))
        {
            auto axis_a = cone_a.axis;
            auto axis_b = cone_b.axis;
            float half_angle_a = cone_a.half_angle;
            float half_angle_b = cone_b.half_angle;
            float3 d_origin_a = make_float3(new_origin.x - origin_a.x, new_origin.y - origin_a.y, new_origin.z - origin_a.z);
            float3 d_origin_b = make_float3(new_origin.x - origin_b.x, new_origin.y - origin_b.y, new_origin.z - origin_b.z);
            ret.radius = std::sqrt(std::max(cone_a.radius * cone_a.radius + squared_length(d_origin_a), cone_b.radius * cone_b.radius + squared_length(d_origin_b)));

            if (half_angle_b > half_angle_a)
            {
                thrust::swap(axis_a, axis_b);
                thrust::swap(half_angle_a, half_angle_b);
            }

            float theta = std::acos(std::max(-1.0f, std::min(1.0f, dot(make_float3(axis_a.x, axis_a.y, axis_a.z), make_float3(axis_b.x, axis_b.y, axis_b.z)))));
            if (std::min(theta + half_angle_b, static_cast<float>(M_PI)) <= half_angle_a)
            {
                ret.axis = axis_a;
                ret.half_angle = half_angle_a;
                return ret;
            }

            float o_theta = (half_angle_a + theta + half_angle_b) / 2.0f;
            if (o_theta >= static_cast<float>(M_PI))
            {
                ret.axis = axis_a;
                return ret;
            }

            float r_theta = o_theta - half_angle_a;
            float3 axis3 = rotate(make_float3(axis_a.x, axis_a.y, axis_a.z), make_float3(axis_b.x, axis_b.y, axis_b.z), r_theta);
            ret.axis = make_float3(axis3.x, axis3.y, axis3.z);
            ret.half_angle = o_theta;
        }
        else if (is_valid(cone_a))
        {
            ret = cone_a;
        }
        else if (is_valid(cone_b))
        {
            ret = cone_b;
        }
        else
        {
            ret.half_angle = -static_cast<float>(M_PI);
        }

        return ret;
    }

    SNCH_LBVH_CALLABLE cone<double, 3> merge(const cone<double, 3> &cone_a, const cone<double, 3> &cone_b, const double3 &origin_a, const double3 &origin_b, const double3 &new_origin) noexcept
    {
        cone<double, 3> ret;
        if (is_valid(cone_a) && is_valid(cone_b))
        {
            auto axis_a = cone_a.axis;
            auto axis_b = cone_b.axis;
            double half_angle_a = cone_a.half_angle;
            double half_angle_b = cone_b.half_angle;
            double3 d_origin_a = make_double3(new_origin.x - origin_a.x, new_origin.y - origin_a.y, new_origin.z - origin_a.z);
            double3 d_origin_b = make_double3(new_origin.x - origin_b.x, new_origin.y - origin_b.y, new_origin.z - origin_b.z);
            ret.radius = std::sqrt(std::max(cone_a.radius * cone_a.radius + squared_length(d_origin_a), cone_b.radius * cone_b.radius + squared_length(d_origin_b)));

            if (half_angle_b > half_angle_a)
            {
                thrust::swap(axis_a, axis_b);
                thrust::swap(half_angle_a, half_angle_b);
            }

            double theta = std::acos(std::max(-1.0, std::min(1.0, dot(make_double3(axis_a.x, axis_a.y, axis_a.z), make_double3(axis_b.x, axis_b.y, axis_b.z)))));
            if (std::min(theta + half_angle_b, M_PI) <= half_angle_a)
            {
                ret.axis = axis_a;
                ret.half_angle = half_angle_a;
                return ret;
            }

            double o_theta = (half_angle_a + theta + half_angle_b) / 2.0;
            if (o_theta >= M_PI)
            {
                ret.axis = axis_a;
                return ret;
            }

            double r_theta = o_theta - half_angle_a;
            double3 axis3 = rotate(make_double3(axis_a.x, axis_a.y, axis_a.z), make_double3(axis_b.x, axis_b.y, axis_b.z), r_theta);
            ret.axis = make_double3(axis3.x, axis3.y, axis3.z);
            ret.half_angle = o_theta;
        }
        else if (is_valid(cone_a))
        {
            ret = cone_a;
        }
        else if (is_valid(cone_b))
        {
            ret = cone_b;
        }
        else
        {
            ret.half_angle = -M_PI;
        }

        return ret;
    }

} // namespace lbvhs

#endif // LBVH_CONE_CUH