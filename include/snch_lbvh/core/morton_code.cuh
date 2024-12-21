#ifndef LBVH_MORTON_CODE_CUH
#define LBVH_MORTON_CODE_CUH
#include "utility.cuh"
#include <bit>
#include <cstdint>

namespace lbvh
{

    // SNCH_LBVH_CALLABLE std::uint32_t expand_bits(std::uint32_t v) noexcept
    // {
    //     v = (v * 0x00010001u) & 0xFF0000FFu;
    //     v = (v * 0x00000101u) & 0x0F00F00Fu;
    //     v = (v * 0x00000011u) & 0xC30C30C3u;
    //     v = (v * 0x00000005u) & 0x49249249u;
    //     return v;
    // }

	SNCH_LBVH_CALLABLE std::uint32_t expand_bits(std::uint32_t v) noexcept
	{
		assert(v < ((std::uint32_t) 1 << 11));
		// _______32 bits_______
		// |000.............vvv| - 11-v significant bits
		//
		// convert to:
		//
		// _______32 bits_______
		// |0v00v........00v00v|

		// See https://stackoverflow.com/a/1024889
		v = (v | (v << 16)) & 0x070000FF; // 0000 0111 0000 0000  0000 0000 1111 1111
		v = (v | (v <<  8)) & 0x0700F00F; // 0000 0111 0000 0000  1111 0000 0000 1111
		v = (v | (v <<  4)) & 0x430C30C3; // 0100 0011 0000 1100  0011 0000 1100 0011
		v = (v | (v <<  2)) & 0x49249249; // 0100 1001 0010 0100  1001 0010 0100 1001

		assert(v < ((std::uint32_t) 1 << 31));
		return v;
	}

	SNCH_LBVH_CALLABLE std::uint64_t expand_bits(std::uint64_t v) noexcept
	{
		assert(v < ((std::uint64_t) 1 << 22));
		// _______32 bits_____________32 bits_______
		// |000.............000|000.............vvv| - 22-v significant bits
		//
		// convert to:
		//
		// _______32 bits_____________32 bits_______
		// |v00............00v0|0v...........00v00v|

		v = (v | (v << 16)) & 0x0000003F0000FFFFull; // 0000 0000 0000 0000  0000 0000 0011 1111   0000 0000 0000 0000  1111 1111 1111 1111
		v = (v | (v << 16)) & 0x003F0000FF0000FFull; // 0000 0000 0011 1111  0000 0000 0000 0000   1111 1111 0000 0000  0000 0000 1111 1111
		v = (v | (v <<  8)) & 0x300F00F00F00F00Full; // 0011 0000 0000 1111  0000 0000 1111 0000   0000 1111 0000 0000  1111 0000 0000 1111
		v = (v | (v <<  4)) & 0x30C30C30C30C30C3ull; // 0011 0000 1100 0011  0000 1100 0011 0000   1100 0011 0000 1100  0011 0000 1100 0011
		v = (v | (v <<  2)) & 0x9249249249249249ull; // 1001 0010 0100 1001  0010 0100 1001 0010   0100 1001 0010 0100  1001 0010 0100 1001
		return v;
	}

    // Calculates a 30-bit Morton code for the
    // given 3D point located within the unit cube [0,1].
    SNCH_LBVH_CALLABLE std::uint32_t morton_code(float3 xyz, float resolution = 1024.0f) noexcept
    {
        xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
        xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
        xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
        const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
        const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
        const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
        return xx * 4 + yy * 2 + zz;
    }

    SNCH_LBVH_CALLABLE std::uint32_t morton_code(double3 xyz, double resolution = 1024.0) noexcept
    {
        xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0), resolution - 1.0);
        xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0), resolution - 1.0);
        xyz.z = ::fmin(::fmax(xyz.z * resolution, 0.0), resolution - 1.0);
        const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
        const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
        const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
        return xx * 4 + yy * 2 + zz;
    }

    // Calculates a 30-bit Morton code for the
    // given 2D point located within the unit cube [0,1].
    SNCH_LBVH_CALLABLE std::uint32_t morton_code(float2 xyz, float resolution = 1024.0f) noexcept
    {
        xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
        xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
        const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
        const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
        return xx * 2 + yy;
    }

    SNCH_LBVH_CALLABLE std::uint32_t morton_code(double2 xyz, double resolution = 1024.0) noexcept
    {
        xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0), resolution - 1.0);
        xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0), resolution - 1.0);
        const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
        const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
        return xx * 2 + yy;
    }

    SNCH_LBVH_CALLABLE std::uint64_t morton_code64(float3 xyz, float resolution = 1048576.0f) noexcept
    {
        xyz.x = std::min(std::max(xyz.x * resolution, 0.0f), resolution - 1.0f);
        xyz.y = std::min(std::max(xyz.y * resolution, 0.0f), resolution - 1.0f);
        xyz.z = std::min(std::max(xyz.z * resolution, 0.0f), resolution - 1.0f);
        const std::uint64_t xx = expand_bits(static_cast<std::uint64_t>(xyz.x));
        const std::uint64_t yy = expand_bits(static_cast<std::uint64_t>(xyz.y));
        const std::uint64_t zz = expand_bits(static_cast<std::uint64_t>(xyz.z));
        return (xx << 2) | (yy << 1) | zz;
    }

    SNCH_LBVH_CALLABLE std::uint64_t morton_code64(double3 xyz, double resolution = 1048576.0) noexcept
    {
        xyz.x = std::min(std::max(xyz.x * resolution, 0.0), resolution - 1.0);
        xyz.y = std::min(std::max(xyz.y * resolution, 0.0), resolution - 1.0);
        xyz.z = std::min(std::max(xyz.z * resolution, 0.0), resolution - 1.0);
        const std::uint64_t xx = expand_bits(static_cast<std::uint64_t>(xyz.x));
        const std::uint64_t yy = expand_bits(static_cast<std::uint64_t>(xyz.y));
        const std::uint64_t zz = expand_bits(static_cast<std::uint64_t>(xyz.z));
        return (xx << 2) | (yy << 1) | zz;
    }

    // 计算给定2D点的64位Morton码，点位于单位正方形[0,1]内
    SNCH_LBVH_CALLABLE std::uint64_t morton_code64(float2 xy, float resolution = 1048576.0f) noexcept
    {
        xy.x = std::min(std::max(xy.x * resolution, 0.0f), resolution - 1.0f);
        xy.y = std::min(std::max(xy.y * resolution, 0.0f), resolution - 1.0f);
        const std::uint64_t xx = expand_bits(static_cast<std::uint64_t>(xy.x));
        const std::uint64_t yy = expand_bits(static_cast<std::uint64_t>(xy.y));
        return (xx << 1) | yy;
    }

    SNCH_LBVH_CALLABLE std::uint64_t morton_code64(double2 xy, double resolution = 1048576.0) noexcept
    {
        xy.x = std::min(std::max(xy.x * resolution, 0.0), resolution - 1.0);
        xy.y = std::min(std::max(xy.y * resolution, 0.0), resolution - 1.0);
        const std::uint64_t xx = expand_bits(static_cast<std::uint64_t>(xy.x));
        const std::uint64_t yy = expand_bits(static_cast<std::uint64_t>(xy.y));
        return (xx << 1) | yy;
    }

    SNCH_LBVH_DEVICE_INLINE int common_upper_bits(const unsigned int lhs, const unsigned int rhs) noexcept
    {
#ifdef __CUDACC__
        return ::__clz(lhs ^ rhs);
#else
#if __cplusplus >= 202002L
        return std::countl_zero(lhs ^ rhs);
#else
        return __builtin_clz(lhs ^ rhs);
#endif
#endif
    }
    SNCH_LBVH_DEVICE_INLINE int common_upper_bits(const unsigned long long int lhs, const unsigned long long int rhs) noexcept
    {
#ifdef __CUDACC__
        return ::__clzll(lhs ^ rhs);
#else
#if __cplusplus >= 202002L
        return std::countl_zero(lhs ^ rhs);
#else
        return __builtin_clzll(lhs ^ rhs);
#endif
#endif
    }

} // namespace lbvh
#endif // LBVH_MORTON_CODE_CUH
