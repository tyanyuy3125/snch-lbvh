#ifndef LBVH_UTILITY_CUH
#define LBVH_UTILITY_CUH
#include <limits>
#ifdef __CUDACC__
#include <math_constants.h>
#include <vector_types.h>
#else
#define __device__
#define __host__
struct uint2
{
    unsigned int x, y;
};
struct uint3
{
    unsigned int x, y, z;
};
struct uint4
{
    unsigned int x, y, z, w;
};
struct int2
{
    int x, y;
};
struct int3
{
    int x, y, z;
};
struct int4
{
    int x, y, z, w;
};
struct float2
{
    float x, y;
};
struct float3
{
    float x, y, z;
};
struct float4
{
    float x, y, z, w;
};
struct double2
{
    double x, y;
};
struct double3
{
    double x, y, z;
};
struct double4
{
    double x, y, z, w;
};
uint2 make_uint2(unsigned int x, unsigned int y) { return {x, y}; }
uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return {x, y, z}; }
uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x, y, z, w}; }
uint2 make_int2(int x, int y) { return {x, y}; }
uint3 make_int3(int x, int y, int z) { return {x, y, z}; }
uint4 make_int4(int x, int y, int z, int w) { return {x, y, z, w}; }
float2 make_float2(float x, float y) { return {x, y}; }
float3 make_float3(float x, float y, float z) { return {x, y, z}; }
float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
double2 make_double2(double x, double y) { return {x, y}; }
double3 make_double3(double x, double y, double z) { return {x, y, z}; }
double4 make_double4(double x, double y, double z, double w) { return {x, y, z, w}; }
#endif

namespace lbvh
{
    template <typename T>
    __device__ __host__ inline T epsilon() noexcept;
    template <>
    __device__ __host__ inline float epsilon<float>() noexcept { return std::numeric_limits<float>::epsilon(); }
    template <>
    __device__ __host__ inline double epsilon<double>() noexcept { return std::numeric_limits<double>::epsilon(); }

    template <typename T>
    __device__ __host__ inline T one_minus_epsilon() noexcept;
    template <>
    __device__ __host__ inline float one_minus_epsilon<float>() noexcept { return 1.0f - std::numeric_limits<float>::epsilon(); }
    template <>
    __device__ __host__ inline double one_minus_epsilon<double>() noexcept { return 1.0 - std::numeric_limits<double>::epsilon(); }

    __device__ __host__ float3 vec4_to_vec3(float4 p) { return {p.x, p.y, p.z}; }
    __device__ __host__ double3 vec4_to_vec3(double4 p) { return {p.x, p.y, p.z}; }
    __device__ __host__ float4 vec3_to_vec4(float3 p, float w = 0.0f) { return {p.x, p.y, p.z, w}; }
    __device__ __host__ double4 vec3_to_vec4(double3 p, double w = 0.0f) { return {p.x, p.y, p.z, w}; }

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
        using type = float4;
    };
    template <>
    struct vector_of<double, 3>
    {
        using type = double4;
    };

    template <typename T, unsigned int dim>
    using vector_of_t = typename vector_of<T, dim>::type;

    template <typename T>
    __device__ inline T infinity() noexcept;

    template <>
    __device__ __host__ inline float infinity<float>() noexcept { return std::numeric_limits<float>::infinity(); }
    template <>
    __device__ __host__ inline double infinity<double>() noexcept { return std::numeric_limits<double>::infinity(); }

    __device__ __host__ inline float dot(const float2 &a, const float2 &b) { return a.x * b.x + a.y * b.y; }
    __device__ __host__ inline double dot(const double2 &a, const double2 &b) { return a.x * b.x + a.y * b.y; }
    __device__ __host__ inline float dot(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    __device__ __host__ inline double dot(const double3 &a, const double3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

    __device__ __host__ inline bool inrange(float val, float low, float high) { return val >= low && val <= high; }
    __device__ __host__ inline bool inrange(double val, double low, double high) { return val >= low && val <= high; }

    __device__ __host__ inline float2 cwiseabs(const float2 &a) noexcept
    {
        float2 result;
        result.x = fabsf(a.x);
        result.y = fabsf(a.y);
        return result;
    }
    __device__ __host__ inline double2 cwiseabs(const double2 &a) noexcept
    {
        double2 result;
        result.x = fabs(a.x);
        result.y = fabs(a.y);
        return result;
    }
    __device__ __host__ inline float3 cwiseabs(const float3 &a) noexcept
    {
        float3 result;
        result.x = fabsf(a.x);
        result.y = fabsf(a.y);
        result.z = fabsf(a.z);
        return result;
    }
    __device__ __host__ inline double3 cwiseabs(const double3 &a) noexcept
    {
        double3 result;
        result.x = fabs(a.x);
        result.y = fabs(a.y);
        result.z = fabs(a.z);
        return result;
    }

    __device__ __host__ inline float2 cwisemin(const float2 &a, const float2 &b) noexcept
    {
        float2 result;
        result.x = fminf(a.x, b.x);
        result.y = fminf(a.y, b.y);
        return result;
    }
    __device__ __host__ inline double2 cwisemin(const double2 &a, const double2 &b) noexcept
    {
        double2 result;
        result.x = fmin(a.x, b.x);
        result.y = fmin(a.y, b.y);
        return result;
    }
    __device__ __host__ inline float3 cwisemin(const float3 &a, const float3 &b) noexcept
    {
        float3 result;
        result.x = fminf(a.x, b.x);
        result.y = fminf(a.y, b.y);
        result.z = fminf(a.z, b.z);
        return result;
    }
    __device__ __host__ inline double3 cwisemin(const double3 &a, const double3 &b) noexcept
    {
        double3 result;
        result.x = fmin(a.x, b.x);
        result.y = fmin(a.y, b.y);
        result.z = fmin(a.z, b.z);
        return result;
    }

    __device__ __host__ inline float2 cwisemax(const float2 &a, const float2 &b) noexcept
    {
        float2 result;
        result.x = fmaxf(a.x, b.x);
        result.y = fmaxf(a.y, b.y);
        return result;
    }
    __device__ __host__ inline double2 cwisemax(const double2 &a, const double2 &b) noexcept
    {
        double2 result;
        result.x = fmax(a.x, b.x);
        result.y = fmax(a.y, b.y);
        return result;
    }
    __device__ __host__ inline float3 cwisemax(const float3 &a, const float3 &b) noexcept
    {
        float3 result;
        result.x = fmaxf(a.x, b.x);
        result.y = fmaxf(a.y, b.y);
        result.z = fmaxf(a.z, b.z);
        return result;
    }
    __device__ __host__ inline double3 cwisemax(const double3 &a, const double3 &b) noexcept
    {
        double3 result;
        result.x = fmax(a.x, b.x);
        result.y = fmax(a.y, b.y);
        result.z = fmax(a.z, b.z);
        return result;
    }

    __device__ __host__ inline float length(const float4 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }
    __device__ __host__ inline float length(const double4 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }

    __device__ __host__ inline float length(const float3 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    }
    __device__ __host__ inline double length(const double3 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    }

    __device__ __host__ inline float length(const float2 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y);
    }
    __device__ __host__ inline double length(const double2 &a) noexcept
    {
        return std::sqrt(a.x * a.x + a.y * a.y);
    }

    __device__ __host__ inline float squared_length(const float3 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y + a.z * a.z);
    }
    __device__ __host__ inline double squared_length(const double3 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y + a.z * a.z);
    }

    __device__ __host__ inline float squared_length(const float4 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }
    __device__ __host__ inline double squared_length(const double4 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    }

    __device__ __host__ inline float squared_length(const float2 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y);
    }
    __device__ __host__ inline double squared_length(const double2 &a) noexcept
    {
        return (a.x * a.x + a.y * a.y);
    }

    __device__ __host__ inline float3 normalize(const float3 &v)
    {
        float norm = length(v);
        return make_float3(v.x / norm, v.y / norm, v.z / norm);
    }
    __device__ __host__ inline double3 normalize(const double3 &v)
    {
        double norm = length(v);
        return make_double3(v.x / norm, v.y / norm, v.z / norm);
    }

    __device__ __host__ inline float2 normalize(const float2 &v)
    {
        float norm = length(v);
        return make_float2(v.x / norm, v.y / norm);
    }
    __device__ __host__ inline double2 normalize(const double2 &v)
    {
        double norm = length(v);
        return make_double2(v.x / norm, v.y / norm);
    }

    __device__ __host__ inline float3 cross(const float3 &u, const float3 &v) noexcept
    {
        return make_float3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x);
    }
    __device__ __host__ inline double3 cross(const double3 &u, const double3 &v) noexcept
    {
        return make_double3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x);
    }

    __device__ __host__ inline void swap(int &a, int &b)
    {
        int temp = a;
        a = b;
        b = temp;
    }

    __host__ __device__ inline int &get(int2 &vec, int index)
    {
        return reinterpret_cast<int *>(&vec)[index];
    }
    __host__ __device__ inline int &get(int3 &vec, int index)
    {
        return reinterpret_cast<int *>(&vec)[index];
    }
    __host__ __device__ inline int &get(int4 &vec, int index)
    {
        return reinterpret_cast<int *>(&vec)[index];
    }
    __host__ __device__ inline int get(const int2 &vec, int index)
    {
        return reinterpret_cast<const int *>(&vec)[index];
    }
    __host__ __device__ inline int get(const int3 &vec, int index)
    {
        return reinterpret_cast<const int *>(&vec)[index];
    }
    __host__ __device__ inline int get(const int4 &vec, int index)
    {
        return reinterpret_cast<const int *>(&vec)[index];
    }
    __host__ __device__ inline float &get(float2 &vec, int index)
    {
        return reinterpret_cast<float *>(&vec)[index];
    }
    __host__ __device__ inline float &get(float3 &vec, int index)
    {
        return reinterpret_cast<float *>(&vec)[index];
    }
    __host__ __device__ inline float &get(float4 &vec, int index)
    {
        return reinterpret_cast<float *>(&vec)[index];
    }
    __host__ __device__ inline float get(const float2 &vec, int index)
    {
        return reinterpret_cast<const float *>(&vec)[index];
    }
    __host__ __device__ inline float get(const float3 &vec, int index)
    {
        return reinterpret_cast<const float *>(&vec)[index];
    }
    __host__ __device__ inline float get(const float4 &vec, int index)
    {
        return reinterpret_cast<const float *>(&vec)[index];
    }
    __host__ __device__ inline double &get(double2 &vec, int index)
    {
        return reinterpret_cast<double *>(&vec)[index];
    }
    __host__ __device__ inline double &get(double3 &vec, int index)
    {
        return reinterpret_cast<double *>(&vec)[index];
    }
    __host__ __device__ inline double &get(double4 &vec, int index)
    {
        return reinterpret_cast<double *>(&vec)[index];
    }
    __host__ __device__ inline double get(const double2 &vec, int index)
    {
        return reinterpret_cast<const double *>(&vec)[index];
    }
    __host__ __device__ inline double get(const double3 &vec, int index)
    {
        return reinterpret_cast<const double *>(&vec)[index];
    }
    __host__ __device__ inline double get(const double4 &vec, int index)
    {
        return reinterpret_cast<const double *>(&vec)[index];
    }
    __host__ __device__ inline unsigned int &get(uint2 &vec, int index)
    {
        return reinterpret_cast<unsigned int *>(&vec)[index];
    }
    __host__ __device__ inline unsigned int &get(uint3 &vec, int index)
    {
        return reinterpret_cast<unsigned int *>(&vec)[index];
    }
    __host__ __device__ inline unsigned int &get(uint4 &vec, int index)
    {
        return reinterpret_cast<unsigned int *>(&vec)[index];
    }
    __host__ __device__ inline unsigned int get(const uint2 &vec, int index)
    {
        return reinterpret_cast<const unsigned int *>(&vec)[index];
    }
    __host__ __device__ inline unsigned int get(const uint3 &vec, int index)
    {
        return reinterpret_cast<const unsigned int *>(&vec)[index];
    }
    __host__ __device__ inline unsigned int get(const uint4 &vec, int index)
    {
        return reinterpret_cast<const unsigned int *>(&vec)[index];
    }

} // namespace lbvh
#endif // LBVH_UTILITY_CUH
