#include <chrono>
#include <iostream>
#include <memory>
#include <string.h>

#include <emmintrin.h>
#include <immintrin.h>

struct Bench {
	explicit Bench(const size_t megabytes = 100 * 1024 * 1024) noexcept :
		_taskSize{ megabytes / sizeof(size_t)}
	{
		_a = std::unique_ptr<size_t[]>(new size_t[_taskSize]);
		_b = std::unique_ptr<size_t[]>(new size_t[_taskSize]);

		if (((size_t)_a.get()) % 32 != 0 || ((size_t)_b.get()) % 32 != 0)
			::abort();

		::memset(_a.get(), 0xE2, megabytes); // Init the memory - required for the OS to actually allocate all the pages!
		::memset(_b.get(), 0x5F, megabytes);
	}

	size_t runReadBenchmark() noexcept
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		static constexpr size_t stride = 256 / 8 / sizeof(size_t);

		const auto* __restrict aPtr = _a.get();
		const auto* __restrict bPtr = _b.get();
		__m256i sum256 {0};
		for (const auto* end = aPtr + _taskSize; aPtr != end; aPtr += stride, bPtr += stride)
		{
			auto a256 = _mm256_load_si256(reinterpret_cast<const __m256i*>(aPtr));
			auto b256 = _mm256_load_si256(reinterpret_cast<const __m256i*>(bPtr));

			auto localSum256 = _mm256_add_epi64(a256, b256);
			sum256 = _mm256_add_epi64(sum256, localSum256);
		}
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		_result = sum256.m256i_u64[0] + sum256.m256i_u64[1] + sum256.m256i_u64[2] + sum256.m256i_u64[3];

		if (us == 0)
			return 0;

		return _taskSize * sizeof(size_t) * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runReadBenchmarkSSE2() noexcept
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		static constexpr size_t stride = 128 / 8 / sizeof(size_t);

		const auto* __restrict aPtr = _a.get();
		const auto* __restrict bPtr = _b.get();
		__m128i sum{ 0 };
		for (const auto* end = aPtr + _taskSize; aPtr != end; aPtr += stride, bPtr += stride)
		{
			auto a128 = _mm_load_si128(reinterpret_cast<const __m128i*>(aPtr));
			auto b128 = _mm_load_si128(reinterpret_cast<const __m128i*>(bPtr));

			auto localSum = _mm_add_epi64(a128, b128);
			sum = _mm_add_epi64(sum, localSum);
		}
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		_result = sum.m128i_u64[0] + sum.m128i_u64[1];

		if (us == 0)
			return 0;

		return _taskSize * sizeof(size_t) * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runWriteBenchmark() noexcept
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* __restrict aPtr = _a.get();
		auto* __restrict bPtr = _b.get();

		static constexpr size_t stride = 256 / 8 / sizeof(size_t);

		__m256i sum256 = _mm256_set1_epi64x(_result);
		for (const auto* end = aPtr + _taskSize; aPtr != end; aPtr += stride, bPtr += stride)
		{
			_mm256_store_si256(reinterpret_cast<__m256i*>(aPtr), sum256);
			_mm256_store_si256(reinterpret_cast<__m256i*>(bPtr), sum256);
		}
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		if (us == 0)
			return 0;

		return _taskSize * sizeof(size_t) * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runWriteBenchmarkSSE2() noexcept
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* __restrict aPtr = _a.get();
		auto* __restrict bPtr = _b.get();

		static constexpr size_t stride = 128 / 8 / sizeof(size_t);

		__m128i sum = _mm_set1_epi64x(_result);
		for (const auto* end = aPtr + _taskSize; aPtr != end; aPtr += stride, bPtr += stride)
		{
			_mm_storeu_si128(reinterpret_cast<__m128i*>(aPtr), sum);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(bPtr), sum);
		}
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		if (us == 0)
			return 0;

		return _taskSize * sizeof(size_t) * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runCopyBenchmark() noexcept
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* __restrict aPtr = _a.get();
		auto* __restrict bPtr = _b.get();

		/*static constexpr size_t stride = 256 / 8 / sizeof(size_t);

		for (const auto* end = aPtr + _taskSize; aPtr != end; aPtr += stride * 2, bPtr += stride * 2)
		{
			auto tmp1 = _mm256_load_si256(reinterpret_cast<__m256i*>(aPtr));
			auto tmp2 = _mm256_load_si256(reinterpret_cast<__m256i*>(aPtr) + 1);

			_mm256_store_si256(reinterpret_cast<__m256i*>(bPtr), tmp1);
			_mm256_store_si256(reinterpret_cast<__m256i*>(bPtr) + 1, tmp2);
		}*/

		memcpy(bPtr, aPtr, _taskSize * sizeof(size_t));

		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		if (us == 0)
			return 0;

		return _taskSize * sizeof(size_t) * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t runCopyBenchmarkSSE2() noexcept
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* __restrict aPtr = _a.get();
		auto* __restrict bPtr = _b.get();

		static constexpr size_t stride = 128 / 8 / sizeof(size_t);

		for (const auto* end = aPtr + _taskSize; aPtr != end; aPtr += stride * 2, bPtr += stride * 2)
		{
			auto tmp1 = _mm_load_si128(reinterpret_cast<__m128i*>(aPtr));
			auto tmp2 = _mm_load_si128(reinterpret_cast<__m128i*>(aPtr) + 1);

			_mm_store_si128(reinterpret_cast<__m128i*>(bPtr), tmp1);
			_mm_store_si128(reinterpret_cast<__m128i*>(bPtr) + 1, tmp2);
		}

		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

		if (us == 0)
			return 0;

		return _taskSize * sizeof(size_t) * 2 /* A and B*/ * 1'000'000 / (1024 * 1024) / us;
	}

	size_t taskSizeMib() const noexcept
	{
		return _taskSize * sizeof(size_t) / (1024 * 1024);
	}

	auto result() const noexcept
	{
		return _result;
	}

private:
	std::unique_ptr<size_t[]> _a;
	std::unique_ptr<size_t[]> _b;

	const size_t _taskSize;
	size_t _result = 0;
};

int main()
{
	Bench bench{1000 * 1024 * 1024};
	std::cout << "Task size: " << bench.taskSizeMib() << " MiB (x2)" << "\n\n";
	std::cout << std::fixed << std::setprecision(1);

	std::cout << "----------------------------------------------" << '\n';
	std::cout << "Read\t\t" << "Write\t\t" << "Copy\t\t" << '\n';
	std::cout << "----------------------------------------------" << '\n';
	std::cout << (float)bench.runReadBenchmark() / 1024.0f << " GiB/s\t";
	std::cout << (float)bench.runWriteBenchmark() / 1024.0f << " GiB/s\t";
	std::cout << (float)bench.runCopyBenchmark() / 1024.0f << " GiB/s\t";
	std::cout << '\n';
	std::cout << "----------------------------------------------" << '\n';

	std::cout << "\nPress Enter to exit...";
	std::cin.get();

	return bench.result() > 0 ? 0 : 1;
}
