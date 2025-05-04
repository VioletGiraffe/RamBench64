#pragma once

#include <stdint.h>
#include <string>
#include <vector>

struct RamInfo {
	std::string bank = "<unknown slot>";
	std::string manufacturer = "<unknown manufacturer>";
	std::string model = "<unknown model>";
	uint64_t capacity = 0;
	uint32_t clock = 0;
	uint32_t moduleMaxSpeed = 0;
	uint32_t ddrStandardNumber = 0;
};

std::vector<RamInfo> queryMemoryInfo() noexcept;
