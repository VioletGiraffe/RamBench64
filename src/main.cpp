#include <chrono>
#include <iostream>
#include <memory>
#include <string.h>

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

		const auto* aPtr = _a.get();
		const auto* bPtr = _b.get();
		__m256i sum256 {0};
		for (const auto* end = aPtr + _taskSize; aPtr != end; aPtr += 256/8, bPtr += 256/8)
		{
			auto a256 = _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(aPtr));
			auto b256 = _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(bPtr));

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

	size_t runWriteBenchmark() noexcept
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto* aPtr = _a.get();
		auto* bPtr = _b.get();

		__m256i sum256 = _mm256_set1_epi64x(_result);
		for (const auto* end = aPtr + _taskSize; aPtr != end; aPtr += 256 / 8, bPtr += 256 / 8)
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

	std::cout << "Reading from RAM - " << bench.taskSizeMib() << "(x2) MiB: " << (float)bench.runReadBenchmark() / 1024.0f << " GiB/s" << '\n';
	std::cout << "Writing to RAM - " << bench.taskSizeMib() << "(x2) MiB: " << (float)bench.runWriteBenchmark() / 1024.0f << " GiB/s" << '\n';

	return bench.result() > 0 ? 0 : 1;
}
