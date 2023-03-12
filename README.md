# RamBench64
Cross-platform RAM speed benchmark

* Implements memory write, read and copy tests with AVX2 and SSE2.
* Detects supported instructions at runtime.
* Read and write benches have "useful workload" (albeit very simple), not just raw data transfer. Should still be bound by memory latency which is many times longer than the workload needs.
* Performs verification of the data between write and read operations (more as a test for the benchmark code itself, rather than actual memory test, though).
* Copying is almost as fast as MSVC's `memcpy` on Core i5-8500T, but not quite there yet.
