#pragma once

#include <memory>
#include <stdint.h>

#ifdef _WIN32
#define free_aligned _aligned_free
#else
#define free_aligned free
#endif

struct Bench {
	enum InstructionSet {SSE2, AVX, AVX2};

	explicit Bench(const size_t megabytes = 100);

	[[nodiscard]] size_t runReadBenchmark(InstructionSet simdVersion);
	[[nodiscard]] size_t runWriteBenchmark(InstructionSet simdVersion);
	[[nodiscard]] size_t runCopyBenchmark(InstructionSet simdVersion);

	[[nodiscard]] size_t taskSizeMib() const noexcept;
	[[nodiscard]] uint64_t result() const noexcept;

private:
	const size_t _taskSizeBytes;

	std::unique_ptr<std::byte, decltype(&free_aligned)> _a;
	std::unique_ptr<std::byte, decltype(&free_aligned)> _b;

	uint64_t _result = 0;
};
