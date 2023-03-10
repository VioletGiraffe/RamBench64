#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <string.h>

struct Bench {
	explicit Bench(const size_t taskSize = 100 * 1024 * 1024) noexcept :
		_taskSize{ taskSize }
	{
		_a = std::unique_ptr<size_t[]>(new size_t[taskSize]);
		_b = std::unique_ptr<size_t[]>(new size_t[taskSize]);

		::memset(_a.get(), 0, sizeof(size_t) * taskSize); // Init the memory - probably not necessary before iota
		::memset(_b.get(), 0, sizeof(size_t) * taskSize);

		std::iota(_a.get(), _a.get() + taskSize, 0);
		std::iota(_b.get(), _b.get() + taskSize, 0);
	}

	size_t run() noexcept
	{
		const auto startTime = std::chrono::high_resolution_clock::now();

		auto sumArray = std::unique_ptr<size_t[]>(new size_t[_taskSize]);

		size_t* sum = sumArray.get();
		for (size_t i = 0; i < _taskSize; ++i)
		{
			*sum++ = (_a[i] + _b[i]);
		}

		_result = *(--sum);

		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
		if (us == 0)
			return 0;

		return _taskSize * 2 * 1'000'000 / (1024 * 1024) / us;
	}

	size_t taskSizeMib() const noexcept
	{
		return _taskSize / (1024 * 1024);
	}

	auto result() const noexcept
	{
		return _result;
	}

private:
	std::unique_ptr<size_t[]> _a;
	std::unique_ptr<size_t[]> _b;

	const size_t _taskSize;
	size_t _result;
};

int main()
{
	Bench bench;

	const auto mibPerSecond = bench.run();
	std::cout << "Processing " << bench.taskSizeMib() << "(x2) MiB: " << (float)mibPerSecond / 1024.0f << " GiB/s" << '\n';

	return bench.result() > 0 ? 0 : 1;
}
