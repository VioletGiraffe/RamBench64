#include "cpuid-parser/cpuinfo.hpp"
#include "bench.h"
#include "system_info.h"

#include <iomanip>
#include <iostream>
#include <sstream>

inline float bestOfN(Bench& bench, size_t (Bench::* method)(Bench::InstructionSet), const Bench::InstructionSet simdVersion, const size_t N)
{
	float best = 0.0f;
	for (size_t i = 0; i < N; ++i)
	{
		// Highest value (MiB/s) = best result
		if (const auto result = (float)(bench.*method)(simdVersion); result > best)
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

		const auto ramInfo = queryMemoryInfo();
		if (!ramInfo.empty())
		{
			std::cout << "RAM info (per module):\n";
			// Format as a table
			std::cout << std::left << std::setw(20) << "Bank" << std::setw(20) << "Manufacturer" << std::setw(26) << "Model" << std::setw(10) << "Capacity" << std::setw(15) << "Module speed" << std::setw(15) << "Actual Clock (MT/s)" << '\n';
			for (const auto& info : ramInfo)
			{
				std::ostringstream capacity;
				capacity << std::setprecision(2) << (info.capacity / (1024.0f * 1024.0f * 1024.0f)) << " GiB";

				std::ostringstream moduleSpeed;
				moduleSpeed << "DDR" << info.ddrStandardNumber << "-" << info.moduleMaxSpeed;

				std::cout << std::left
					<< std::setw(20) << info.bank
					<< std::setw(20) << info.manufacturer
					<< std::setw(26) << info.model
					<< std::setw(10) << capacity.str()
					<< std::setw(15) << moduleSpeed.str()
					<< std::setw(15) << info.clock
					<< '\n';
			}
		}
		else
			std::cout << "RAM: Information not available.\n";

		Bench bench{ 1000 };
		std::cout << "Task size: " << bench.taskSizeMib() << " MiB (x2)" << "\n\n";
		std::cout << std::fixed << std::setprecision(1);

		// CPU warm-up
		bestOfN(bench, &Bench::runWriteBenchmark, Bench::SSE2, 30);

		const auto runBenchmarkSetAndPrintResults = [&bench](Bench::InstructionSet simdVersion) {
			std::cout << bestOfN(bench, &Bench::runWriteBenchmark, simdVersion, 30) / 1024.0f << " GiB/s\t";
			std::cout << bestOfN(bench, &Bench::runReadBenchmark, simdVersion, 30) / 1024.0f << " GiB/s\t";
			std::cout << bestOfN(bench, &Bench::runCopyBenchmark, simdVersion, 30) / 1024.0f << " GiB/s\t";
		};

		std::cout << "---------------------------------------------------" << '\n';
		std::cout << "\tWrite\t\t" << "Read\t\t" << "Copy\t\t" << '\n';
		std::cout << "---------------------------------------------------" << '\n';
		std::cout << "AVX2\t";
		if (cpuInfo.haveAVX2())
			runBenchmarkSetAndPrintResults(Bench::AVX2);
		else
			std::cout << "N/A\t\t" << "N/A\t\t" << "N/A\t\t";
		std::cout << '\n';

		//std::cout << "AVX\t";
		//if (cpuInfo.haveAVX())
		//	runBenchmarkSetAndPrintResults(Bench::AVX);
		//else
		//	std::cout << "N/A\t\t" << "N/A\t\t" << "N/A\t\t";
		//std::cout << '\n';

		std::cout << "SSE2\t";
		if (cpuInfo.haveSSE2())
			runBenchmarkSetAndPrintResults(Bench::SSE2);
		else
			std::cout << "N/A\t\t" << "N/A\t\t" << "N/A\t\t";
		std::cout << '\n';

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
