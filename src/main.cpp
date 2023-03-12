#include "cpuid-parser/cpuinfo.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <stdlib.h>
#include <string.h>

#include <emmintrin.h>
#include <immintrin.h>

inline void* malloc_aligned(size_t size, size_t alignment) noexcept
{
#ifdef _WIN32
	return _aligned_malloc(size, alignment);
#else
	return aligned_alloc(alignment, size);
#endif
}

#ifdef _WIN32
#define free_aligned _aligned_free
#else
#define free_aligned free
#endif

struct Bench {
	explicit Bench(const size_t megabytes = 100) :
		_taskSizeBytes{ megabytes * 1024 * 1024 },
		_a{ reinterpret_cast<char8_t*>(malloc_aligned(_taskSizeBytes, 256 / 8)), &free_aligned },
		_b{ reinterpret_cast<char8_t*>(malloc_aligned(_taskSizeBytes, 256 / 8)), &free_aligned }
	{
		if (!_a || !_b)
			throw std::runtime_error("Failed to allocate memory!");

		if (((size_t)_a.get()) % (256 / 8) != 0 || ((size_t)_b.get()) % (256 / 8) != 0)
			throw std::runtime_error("Memory not aligned!");

		// Init the memory - required for the OS to actually allocate all the pages!
		::memset(_a.get(), 0xAA, _taskSizeBytes);
		::memset(_b.get(), 0xEE, _taskSizeBytes);
	}

	size_t runReadBenchmark()
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		static constexpr size_t stride = 256 / 8;

		const char8_t* __restrict aPtr = _a.get();
		const char8_t* __restrict bPtr = _b.get();

		__m256i sum256 {0};
		for (const auto* end = aPtr + _taskSizeBytes; aPtr != end; aPtr += stride, bPtr += stride)
		{
			auto a256 = _mm256_load_si256(reinterpret_cast<const __m256i*>(aPtr));
			auto b256 = _mm256_load_si256(reinterpret_cast<const __m256i*>(bPtr));

			auto localSum256 = _mm256_add_epi64(a256, b256);
			sum256 = _mm256_add_epi64(sum256, localSum256);
		}
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

#ifdef _WIN32
		_result = sum256.m256i_u64[0] + sum256.m256i_u64[1] + sum256.m256i_u64[2] + sum256.m256i_u64[3];
#else
		_result = sum256[0] + sum256[1] + sum256[2] + sum256[3];
#endif

		// Verifying the result. The formula for sum of all consecutive numbers 1..N is n * (n + 1) / 2
		const uint64_t N = _taskSizeBytes / sizeof(uint64_t) * 2 - 1 /* there is 0 */;
		const uint64_t expectedSum = N * (N + 1) / 2;
		if (expectedSum != _result)
			throw std::runtime_error("Result verification failed! Memory error or CPU instability?");

		if (us == 0)
			return 0;

		return _taskSizeBytes * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runReadBenchmarkSSE2()
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		static constexpr size_t stride = 128 / 8;

		const char8_t* __restrict aPtr = _a.get();
		const char8_t* __restrict bPtr = _b.get();

		__m128i sum{ 0 };
		for (const auto* end = aPtr + _taskSizeBytes; aPtr != end; aPtr += stride * 2, bPtr += stride * 2)
		{
			_mm_prefetch(reinterpret_cast<const char*>(aPtr) + stride * 4, _MM_HINT_T0);

			auto a128 = _mm_load_si128(reinterpret_cast<const __m128i*>(aPtr));
			auto b128 = _mm_load_si128(reinterpret_cast<const __m128i*>(bPtr));


			auto c128 = _mm_load_si128(reinterpret_cast<const __m128i*>(aPtr) + 1);
			auto d128 = _mm_load_si128(reinterpret_cast<const __m128i*>(bPtr) + 1);

			sum = _mm_add_epi64(sum, _mm_add_epi64(a128, b128));
			sum = _mm_add_epi64(sum, _mm_add_epi64(c128, d128));
		}
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

#ifdef _WIN32
		_result = sum.m128i_u64[0] + sum.m128i_u64[1];
#else
		_result = sum[0] + sum[1];
#endif

		// Verifying the result. The formula for sum of all consecutive numbers 1..N is n * (n + 1) / 2
		const uint64_t N = _taskSizeBytes / sizeof(uint64_t) * 2 - 1 /* there is 0 */;
		const uint64_t expectedSum = N * (N + 1) / 2;
		if (expectedSum != _result)
			throw std::runtime_error("Result verification failed! Memory error or CPU instability?");

		if (us == 0)
			return 0;

		return _taskSizeBytes * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runWriteBenchmark()
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* __restrict aPtr = _a.get();
		auto* __restrict bPtr = _b.get();

		static constexpr size_t stride = 256 / 8;

		const __m256i inc = _mm256_set1_epi64x(8);

		__m256i valuesEven = _mm256_set_epi64x(0, 2, 4, 6);
		__m256i valuesOdd = _mm256_set_epi64x(1, 3, 5, 7);
		for (const auto* end = aPtr + _taskSizeBytes; aPtr != end; aPtr += stride, bPtr += stride)
		{
			_mm256_stream_si256(reinterpret_cast<__m256i*>(aPtr), valuesEven);
			valuesEven = _mm256_add_epi64(valuesEven, inc);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(bPtr), valuesOdd);
			valuesOdd = _mm256_add_epi64(valuesOdd, inc);
		}
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		if (us == 0)
			return 0;

		return _taskSizeBytes * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runWriteBenchmarkSSE2()
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* __restrict aPtr = _a.get();
		auto* __restrict bPtr = _b.get();

		static constexpr size_t stride = 128 / 8;

		const __m128i inc = _mm_set1_epi64x(4);

		__m128i valuesEven = _mm_set_epi64x(0, 2);
		__m128i valuesOdd = _mm_set_epi64x(1, 3);
		for (const auto* end = aPtr + _taskSizeBytes; aPtr != end; aPtr += stride, bPtr += stride)
		{
			_mm_stream_si128(reinterpret_cast<__m128i*>(aPtr), valuesEven);
			valuesEven = _mm_add_epi64(valuesEven, inc);

			_mm_stream_si128(reinterpret_cast<__m128i*>(bPtr), valuesOdd);
			valuesOdd = _mm_add_epi64(valuesOdd, inc);
		}
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		if (us == 0)
			return 0;

		return _taskSizeBytes * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runCopyBenchmark()
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* __restrict aPtr = _a.get();
		auto* __restrict bPtr = _b.get();

		static constexpr size_t stride = 256 / 8;

		for (const auto* end = aPtr + _taskSizeBytes; aPtr != end; aPtr += stride * 2, bPtr += stride * 2)
		{
			_mm_prefetch(reinterpret_cast<char*>(aPtr) + stride * 2, _MM_HINT_T0);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(bPtr),     _mm256_load_si256(reinterpret_cast<__m256i*>(aPtr)));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(bPtr) + 1, _mm256_load_si256(reinterpret_cast<__m256i*>(aPtr) + 1));
		}

		//memcpy(bPtr, aPtr, _taskSizeBytes);

		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		if (us == 0)
			return 0;

		return _taskSizeBytes * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runCopyBenchmarkSSE2()
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* __restrict aPtr = _a.get();
		auto* __restrict bPtr = _b.get();

		static constexpr size_t stride = 128 / 8;

		for (const auto* end = aPtr + _taskSizeBytes; aPtr != end; aPtr += stride * 2, bPtr += stride * 2)
		{
			_mm_prefetch(reinterpret_cast<char*>(aPtr) + stride * 2, _MM_HINT_T0);

			_mm_stream_si128(reinterpret_cast<__m128i*>(bPtr), _mm_load_si128(reinterpret_cast<__m128i*>(aPtr)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(bPtr) + 1, _mm_load_si128(reinterpret_cast<__m128i*>(aPtr) + 1));
		}

		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		if (us == 0)
			return 0;

		return _taskSizeBytes * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t taskSizeMib() const noexcept
	{
		return _taskSizeBytes / (1024 * 1024);
	}

	uint64_t result() const noexcept
	{
		return _result;
	}

private:
	const size_t _taskSizeBytes;

	std::unique_ptr<char8_t, decltype(&free_aligned)> _a;
	std::unique_ptr<char8_t, decltype(&free_aligned)> _b;

	uint64_t _result = 0;
};

static float bestOfN(Bench& bench, size_t (Bench::* method)(), const size_t N)
{
	float best = 0.0f;
	for (size_t i = 0; i < N; ++i)
	{
		// Highest value (MiB/s) = best result
		if (const auto result = (float)(bench.*method)(); result > best)
			best = result;
	}

	return best;
}

int main()
{
	bool success = false;
	try {
		CPUInfo cpuInfo;
		std::cout << "Running on " << cpuInfo.model() << '\n';

		Bench bench{ 1000 };
		std::cout << "Task size: " << bench.taskSizeMib() << " MiB (x2)" << "\n\n";
		std::cout << std::fixed << std::setprecision(1);

		std::cout << "---------------------------------------------------" << '\n';
		std::cout << "\tWrite\t\t" << "Read\t\t" << "Copy\t\t" << '\n';
		std::cout << "---------------------------------------------------" << '\n';
		std::cout << "AVX2\t";
		if (cpuInfo.haveAVX2())
		{
			std::cout << bestOfN(bench, &Bench::runWriteBenchmark, 30) / 1024.0f << " GiB/s\t";
			std::cout << bestOfN(bench, &Bench::runReadBenchmark, 30) / 1024.0f << " GiB/s\t";
			std::cout << bestOfN(bench, &Bench::runCopyBenchmark, 30) / 1024.0f << " GiB/s\t";
		}
		else
			std::cout << "N/A\t\t" << "N/A\t\t" << "N/A\t\t";

		std::cout << '\n';

		std::cout << "SSE2\t";
		std::cout << bestOfN(bench, &Bench::runWriteBenchmarkSSE2, 30) / 1024.0f << " GiB/s\t";
		std::cout << bestOfN(bench, &Bench::runReadBenchmarkSSE2, 30) / 1024.0f << " GiB/s\t";
		std::cout << bestOfN(bench, &Bench::runCopyBenchmarkSSE2, 30) / 1024.0f << " GiB/s\t";
		std::cout << '\n';
		std::cout << "---------------------------------------------------" << '\n';

		success = true;
	}
	catch (const std::exception& e) {
		std::cout << "!!! Error !!!\n" << e.what() << '\n';
	}

	std::cout << "\nPress Enter to exit...";
	std::cin.get();

	return success ? 0 : 1;
}
