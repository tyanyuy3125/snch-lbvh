#pragma once
#include <limits>
#ifdef __CUDACC__
#include <math_constants.h>
#include <vector_types.h>

#define SNCH_LBVH_DEVICE __device__
#define SNCH_LBVH_HOST __host__
#else
#define SNCH_LBVH_DEVICE
#define SNCH_LBVH_HOST
#if !__has_include(<vector_types.h>)
#if defined(__type_exists)
#define TYPE_EXISTS(T) __type_exists(T)
#else
#define TYPE_EXISTS(T) 0
#endif

#ifndef uint2
#if !TYPE_EXISTS(uint2)
struct uint2
{
    unsigned int x, y;
};
#endif
#endif

#ifndef uint3
#if !TYPE_EXISTS(uint3)
struct uint3
{
    unsigned int x, y, z;
};
#endif
#endif

#ifndef uint4
#if !TYPE_EXISTS(uint4)
struct uint4
{
    unsigned int x, y, z, w;
};
#endif
#endif

#ifndef int2
#if !TYPE_EXISTS(int2)
struct int2
{
    int x, y;
};
#endif
#endif

#ifndef int3
#if !TYPE_EXISTS(int3)
struct int3
{
    int x, y, z;
};
#endif
#endif

#ifndef int4
#if !TYPE_EXISTS(int4)
struct int4
{
    int x, y, z, w;
};
#endif
#endif

#ifndef float2
#if !TYPE_EXISTS(float2)
struct float2
{
    float x, y;
};
#endif
#endif

#ifndef float3
#if !TYPE_EXISTS(float3)
struct float3
{
    float x, y, z;
};
#endif
#endif

#ifndef float4
#if !TYPE_EXISTS(float4)
struct float4
{
    float x, y, z, w;
};
#endif
#endif

#ifndef double2
#if !TYPE_EXISTS(double2)
struct double2
{
    double x, y;
};
#endif
#endif

#ifndef double3
#if !TYPE_EXISTS(double3)
struct double3
{
    double x, y, z;
};
#endif
#endif

#ifndef double4
#if !TYPE_EXISTS(double4)
struct double4
{
    double x, y, z, w;
};
#endif
#endif

#ifndef make_uint2
uint2 make_uint2(unsigned int x, unsigned int y) { return {x, y}; }
#endif

#ifndef make_uint3
uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return {x, y, z}; }
#endif

#ifndef make_uint4
uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x, y, z, w}; }
#endif

#ifndef make_int2
int2 make_int2(int x, int y) { return {x, y}; }
#endif

#ifndef make_int3
int3 make_int3(int x, int y, int z) { return {x, y, z}; }
#endif

#ifndef make_int4
int4 make_int4(int x, int y, int z, int w) { return {x, y, z, w}; }
#endif

#ifndef make_float2
float2 make_float2(float x, float y) { return {x, y}; }
#endif

#ifndef make_float3
float3 make_float3(float x, float y, float z) { return {x, y, z}; }
#endif

#ifndef make_float4
float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
#endif

#ifndef make_double2
double2 make_double2(double x, double y) { return {x, y}; }
#endif

#ifndef make_double3
double3 make_double3(double x, double y, double z) { return {x, y, z}; }
#endif

#ifndef make_double4
double4 make_double4(double x, double y, double z, double w) { return {x, y, z, w}; }
#endif
#endif
#endif

#define SNCH_LBVH_CALLABLE inline SNCH_LBVH_DEVICE SNCH_LBVH_HOST
#define SNCH_LBVH_HOST_DEVICE SNCH_LBVH_DEVICE SNCH_LBVH_HOST
#define SNCH_LBVH_DEVICE_INLINE inline SNCH_LBVH_DEVICE

namespace lbvh
{
    template <typename T>
    SNCH_LBVH_CALLABLE T epsilon() noexcept;
    template <>
    SNCH_LBVH_CALLABLE float epsilon<float>() noexcept { return std::numeric_limits<float>::epsilon(); }
    template <>
    SNCH_LBVH_CALLABLE double epsilon<double>() noexcept { return std::numeric_limits<double>::epsilon(); }

    template <typename T>
    SNCH_LBVH_CALLABLE T one_minus_epsilon() noexcept;
    template <>
    SNCH_LBVH_CALLABLE float one_minus_epsilon<float>() noexcept { return 1.0f - std::numeric_limits<float>::epsilon(); }
    template <>
    SNCH_LBVH_CALLABLE double one_minus_epsilon<double>() noexcept { return 1.0 - std::numeric_limits<double>::epsilon(); }

    SNCH_LBVH_CALLABLE float3 vec4_to_vec3(float4 p) { return {p.x, p.y, p.z}; }
    SNCH_LBVH_CALLABLE double3 vec4_to_vec3(double4 p) { return {p.x, p.y, p.z}; }
    SNCH_LBVH_CALLABLE float4 vec3_to_vec4(float3 p, float w = 0.0f) { return {p.x, p.y, p.z, w}; }
    SNCH_LBVH_CALLABLE double4 vec3_to_vec4(double3 p, double w = 0.0f) { return {p.x, p.y, p.z, w}; }

    template <typename T, unsigned int dim>
    struct vector_of;
    template <>
    struct vector_of<float, 2>
    {
        using type = float2;
    };
    template <>
    struct vector_of<double, 2>
    {
        using type = double2;
    };
    template <>
    struct vector_of<float, 3>
    {
        using type = float3;
    };
    template <>
    struct vector_of<double, 3>
    {
        using type = double3;
    };

    template <typename T, unsigned int dim>
    using vector_of_t = typename vector_of<T, dim>::type;

    template <typename T>
    SNCH_LBVH_CALLABLE T infinity() noexcept;

    template <>
    SNCH_LBVH_CALLABLE float infinity<float>() noexcept { return std::numeric_limits<float>::infinity(); }
    template <>
    SNCH_LBVH_CALLABLE double infinity<double>() noexcept { return std::numeric_limits<double>::infinity(); }

    SNCH_LBVH_CALLABLE float dot(const float2 &a, const float2 &b) { return a.x * b.x + a.y * b.y; }
    SNCH_LBVH_CALLABLE double dot(const double2 &a, const double2 &b) { return a.x * b.x + a.y * b.y; }
    SNCH_LBVH_CALLABLE float dot(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    SNCH_LBVH_CALLABLE double dot(const double3 &a, const double3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

    SNCH_LBVH_CALLABLE bool inrange(float val, float low, float high) { return val >= low && val <= high; }
    SNCH_LBVH_CALLABLE bool inrange(double val, double low, double high) { return val >= low && val <= high; }

    SNCH_LBVH_CALLABLE float2 cwiseabs(const float2 &a) noexcept
    {
        float2 result;
        result.x = fabsf(a.x);
        result.y = fabsf(a.y);
        return result;
    }
    SNCH_LBVH_CALLABLE double2 cwiseabs(const double2 &a) noexcept
    {
        double2 result;
        result.x = fabs(a.x);
        result.y = fabs(a.y);
        return result;
    }
    SNCH_LBVH_CALLABLE float3 cwiseabs(const float3 &a) noexcept
    {
        float3 result;
        result.x = fabsf(a.x);
        result.y = fabsf(a.y);
        result.z = fabsf(a.z);
        return result;
    }
    SNCH_LBVH_CALLABLE double3 cwiseabs(const double3 &a) noexcept
    {
        double3 result;
        result.x = fabs(a.x);
        result.y = fabs(a.y);
        result.z = fabs(a.z);
        return result;
    }

    SNCH_LBVH_CALLABLE float2 cwisemin(const float2 &a, const float2 &b) noexcept
    {
        float2 result;
        result.x = fminf(a.x, b.x);
        result.y = fminf(a.y, b.y);
        return result;
    }
    SNCH_LBVH_CALLABLE double2 cwisemin(const double2 &a, const double2 &b) noexcept
    {
        double2 result;
        result.x = fmin(a.x, b.x);
        result.y = fmin(a.y, b.y);
        return result;
    }
    SNCH_LBVH_CALLABLE float3 cwisemin(const float3 &a, const float3 &b) noexcept
    {
        float3 result;
        result.x = fminf(a.x, b.x);
        result.y = fminf(a.y, b.y);
        result.z = fminf(a.z, b.z);
        return result;
    }
    SNCH_LBVH_CALLABLE double3 cwisemin(const double3 &a, const double3 &b) noexcept
    {
        double3 result;
        result.x = fmin(a.x, b.x);
        result.y = fmin(a.y, b.y);
        result.z = fmin(a.z, b.z);
        return result;
    }
    SNCH_LBVH_CALLABLE float4 cwisemin(const float4 &a, const float4 &b) noexcept
    {
        float4 result;
        result.x = fminf(a.x, b.x);
        result.y = fminf(a.y, b.y);
        result.z = fminf(a.z, b.z);
        result.w = fminf(a.w, b.w);
        return result;
    }
    SNCH_LBVH_CALLABLE double4 cwisemin(const double4 &a, const double4 &b) noexcept
    {
        double4 result;
        result.x = fmin(a.x, b.x);
        result.y = fmin(a.y, b.y);
        result.z = fmin(a.z, b.z);
        result.w = fmin(a.w, b.w);
        return result;
    }

    SNCH_LBVH_CALLABLE float2 cwisemax(const float2 &a, const float2 &b) noexcept
    {
        float2 result;
        result.x = fmaxf(a.x, b.x);
        result.y = fmaxf(a.y, b.y);
        return result;
    }
    SNCH_LBVH_CALLABLE double2 cwisemax(const double2 &a, const double2 &b) noexcept
    {
        double2 result;
        result.x = fmax(a.x, b.x);
        result.y = fmax(a.y, b.y);
        return result;
    }
    SNCH_LBVH_CALLABLE float3 cwisemax(const float3 &a, const float3 &b) noexcept
    {
        float3 result;
        result.x = fmaxf(a.x, b.x);
        result.y = fmaxf(a.y, b.y);
        result.z = fmaxf(a.z, b.z);
        return result;
    }
    SNCH_LBVH_CALLABLE double3 cwisemax(const double3 &a, const double3 &b) noexcept
    {
        double3 result;
        result.x = fmax(a.x, b.x);
        result.y = fmax(a.y, b.y);
        result.z = fmax(a.z, b.z);
        return result;
    }
    SNCH_LBVH_CALLABLE float4 cwisemax(const float4 &a, const float4 &b) noexcept
    {
        float4 result;
        result.x = fmaxf(a.x, b.x);
        result.y = fmaxf(a.y, b.y);
        result.z = fmaxf(a.z, b.z);
        result.w = fmaxf(a.w, b.w);
        return result;
    }
    SNCH_LBVH_CALLABLE double4 cwisemax(const double4 &a, const double4 &b) noexcept
    {
        double4 result;
        result.x = fmax(a.x, b.x);
        result.y = fmax(a.y, b.y);
        result.z = fmax(a.z, b.z);
        result.w = fmax(a.w, b.w);
        return result;
    }

    SNCH_LBVH_CALLABLE float length(const float4 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }
    SNCH_LBVH_CALLABLE float length(const double4 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }

    SNCH_LBVH_CALLABLE float length(const float3 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    }
    SNCH_LBVH_CALLABLE double length(const double3 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    }

    SNCH_LBVH_CALLABLE float length(const float2 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y);
    }
    SNCH_LBVH_CALLABLE double length(const double2 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y);
    }

    SNCH_LBVH_CALLABLE float squared_length(const float3 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y + a.z * a.z);
    }
    SNCH_LBVH_CALLABLE double squared_length(const double3 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y + a.z * a.z);
    }

    SNCH_LBVH_CALLABLE float squared_length(const float4 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }
    SNCH_LBVH_CALLABLE double squared_length(const double4 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }

    SNCH_LBVH_CALLABLE float squared_length(const float2 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y);
    }
    SNCH_LBVH_CALLABLE double squared_length(const double2 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y);
    }

    SNCH_LBVH_CALLABLE float3 normalize(const float3 &v)
    {
        float norm = length(v);
        return make_float3(v.x / norm, v.y / norm, v.z / norm);
    }
    SNCH_LBVH_CALLABLE double3 normalize(const double3 &v)
    {
        double norm = length(v);
        return make_double3(v.x / norm, v.y / norm, v.z / norm);
    }

    SNCH_LBVH_CALLABLE float2 normalize(const float2 &v)
    {
        float norm = length(v);
        return make_float2(v.x / norm, v.y / norm);
    }
    SNCH_LBVH_CALLABLE double2 normalize(const double2 &v)
    {
        double norm = length(v);
        return make_double2(v.x / norm, v.y / norm);
    }

    SNCH_LBVH_CALLABLE float3 cross(const float3 &u, const float3 &v) noexcept
    {
        return make_float3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x);
    }
    SNCH_LBVH_CALLABLE double3 cross(const double3 &u, const double3 &v) noexcept
    {
        return make_double3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x);
    }

    template <typename T>
    SNCH_LBVH_CALLABLE void swap(T &a, T &b)
    {
        T temp = a;
        a = b;
        b = temp;
    }

    SNCH_LBVH_CALLABLE int &get(int2 &vec, int index)
    {
        return reinterpret_cast<int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE int &get(int3 &vec, int index)
    {
        return reinterpret_cast<int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE int &get(int4 &vec, int index)
    {
        return reinterpret_cast<int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE int get(const int2 &vec, int index)
    {
        return reinterpret_cast<const int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE int get(const int3 &vec, int index)
    {
        return reinterpret_cast<const int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE int get(const int4 &vec, int index)
    {
        return reinterpret_cast<const int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE float &get(float2 &vec, int index)
    {
        return reinterpret_cast<float *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE float &get(float3 &vec, int index)
    {
        return reinterpret_cast<float *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE float &get(float4 &vec, int index)
    {
        return reinterpret_cast<float *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE float get(const float2 &vec, int index)
    {
        return reinterpret_cast<const float *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE float get(const float3 &vec, int index)
    {
        return reinterpret_cast<const float *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE float get(const float4 &vec, int index)
    {
        return reinterpret_cast<const float *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE double &get(double2 &vec, int index)
    {
        return reinterpret_cast<double *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE double &get(double3 &vec, int index)
    {
        return reinterpret_cast<double *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE double &get(double4 &vec, int index)
    {
        return reinterpret_cast<double *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE double get(const double2 &vec, int index)
    {
        return reinterpret_cast<const double *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE double get(const double3 &vec, int index)
    {
        return reinterpret_cast<const double *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE double get(const double4 &vec, int index)
    {
        return reinterpret_cast<const double *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE unsigned int &get(uint2 &vec, int index)
    {
        return reinterpret_cast<unsigned int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE unsigned int &get(uint3 &vec, int index)
    {
        return reinterpret_cast<unsigned int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE unsigned int &get(uint4 &vec, int index)
    {
        return reinterpret_cast<unsigned int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE unsigned int get(const uint2 &vec, int index)
    {
        return reinterpret_cast<const unsigned int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE unsigned int get(const uint3 &vec, int index)
    {
        return reinterpret_cast<const unsigned int *>(&vec)[index];
    }
    SNCH_LBVH_CALLABLE unsigned int get(const uint4 &vec, int index)
    {
        return reinterpret_cast<const unsigned int *>(&vec)[index];
    }

    SNCH_LBVH_CALLABLE int checkPointSide(float2 p0, float2 p1, float2 point)
    {
        float2 direction;
        direction.x = p1.x - p0.x;
        direction.y = p1.y - p0.y;

        float2 normal;
        normal.x = -direction.y;
        normal.y = direction.x;

        float2 vec;
        vec.x = point.x - p0.x;
        vec.y = point.y - p0.y;

        float dotProduct = vec.x * normal.x + vec.y * normal.y;

        if (dotProduct > 0)
        {
            return 1;
        }
        else if (dotProduct < 0)
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }

    SNCH_LBVH_CALLABLE int checkPointSide(float3 p0, float3 p1, float3 p2, float3 point)
    {
        float3 edge1;
        edge1.x = p1.x - p0.x;
        edge1.y = p1.y - p0.y;
        edge1.z = p1.z - p0.z;

        float3 edge2;
        edge2.x = p2.x - p0.x;
        edge2.y = p2.y - p0.y;
        edge2.z = p2.z - p0.z;

        float3 normal;
        normal.x = edge1.y * edge2.z - edge1.z * edge2.y;
        normal.y = edge1.z * edge2.x - edge1.x * edge2.z;
        normal.z = edge1.x * edge2.y - edge1.y * edge2.x;

        float3 vec;
        vec.x = point.x - p0.x;
        vec.y = point.y - p0.y;
        vec.z = point.z - p0.z;

        float dotProduct = vec.x * normal.x + vec.y * normal.y + vec.z * normal.z;

        if (dotProduct > 0)
        {
            return 1;
        }
        else if (dotProduct < 0)
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }

    SNCH_LBVH_CALLABLE float computeProjectionRatio(float2 p0, float2 p1, float2 point)
    {
        float2 direction;
        direction.x = p1.x - p0.x;
        direction.y = p1.y - p0.y;

        float lengthSquared = direction.x * direction.x + direction.y * direction.y;

        if (lengthSquared < 1e-8f)
        {
            return 0.0f;
        }

        float2 vec;
        vec.x = point.x - p0.x;
        vec.y = point.y - p0.y;

        float dotProduct = vec.x * direction.x + vec.y * direction.y;
        float t = dotProduct / lengthSquared;

        // if (t < 0.0f)
        // {
        //     t = 0.0f;
        // }
        // else if (t > 1.0f)
        // {
        //     t = 1.0f;
        // }

        return t;
    }

    SNCH_LBVH_CALLABLE float2 computeProjectionRatio(float3 p0, float3 p1, float3 p2, float3 point)
    {
        float3 edge1;
        edge1.x = p1.x - p0.x;
        edge1.y = p1.y - p0.y;
        edge1.z = p1.z - p0.z;

        float3 edge2;
        edge2.x = p2.x - p0.x;
        edge2.y = p2.y - p0.y;
        edge2.z = p2.z - p0.z;

        float3 vec;
        vec.x = point.x - p0.x;
        vec.y = point.y - p0.y;
        vec.z = point.z - p0.z;

        float edge1DotEdge1 = edge1.x * edge1.x + edge1.y * edge1.y + edge1.z * edge1.z;
        float edge1DotEdge2 = edge1.x * edge2.x + edge1.y * edge2.y + edge1.z * edge2.z;
        float edge2DotEdge2 = edge2.x * edge2.x + edge2.y * edge2.y + edge2.z * edge2.z;
        float vecDotEdge1 = vec.x * edge1.x + vec.y * edge1.y + vec.z * edge1.z;
        float vecDotEdge2 = vec.x * edge2.x + vec.y * edge2.y + vec.z * edge2.z;

        float denominator = edge1DotEdge1 * edge2DotEdge2 - edge1DotEdge2 * edge1DotEdge2;
        float u = (vecDotEdge1 * edge2DotEdge2 - vecDotEdge2 * edge1DotEdge2) / denominator;
        float v = (vecDotEdge2 * edge1DotEdge1 - vecDotEdge1 * edge1DotEdge2) / denominator;

        // if (u < 0.0f) u = 0.0f;
        // if (v < 0.0f) v = 0.0f;
        // if (u + v > 1.0f) {
        //     float excess = u + v - 1.0f;
        //     u -= excess * (u / (u + v));
        //     v -= excess * (v / (u + v));
        // }

        float2 result;
        result.x = u;
        result.y = v;
        return result;
    }
} // namespace lbvh
