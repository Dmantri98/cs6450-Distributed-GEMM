// worker.cpp
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <algorithm>

struct SharedHeader {
    uint64_t M, N, K;
    uint64_t tile_m, tile_n;
    uint64_t num_tiles_m, num_tiles_n;
};

size_t shared_region_size(const SharedHeader& h) {
    size_t header_bytes = sizeof(SharedHeader);
    size_t a_bytes = sizeof(double) * h.M * h.K;
    size_t b_bytes = sizeof(double) * h.K * h.N;
    size_t c_bytes = sizeof(double) * h.M * h.N;
    size_t flags_bytes = h.num_tiles_m * h.num_tiles_n;
    return header_bytes + a_bytes + b_bytes + c_bytes + flags_bytes;
}

struct SharedPtrs {
    SharedHeader* header;
    double* A;
    double* B;
    double* C;
    uint8_t* flags;
};

SharedPtrs get_ptrs_from_base(void* base) {
    auto* p = static_cast<char*>(base);

    SharedPtrs out;
    out.header = reinterpret_cast<SharedHeader*>(p);
    p += sizeof(SharedHeader);

    const auto& h = *out.header;

    out.A = reinterpret_cast<double*>(p);
    p += sizeof(double) * h.M * h.K;

    out.B = reinterpret_cast<double*>(p);
    p += sizeof(double) * h.K * h.N;

    out.C = reinterpret_cast<double*>(p);
    p += sizeof(double) * h.M * h.N;

    out.flags = reinterpret_cast<uint8_t*>(p);
    return out;
}

// Compute a single C tile: [row0,row1) × [col0,col1)
void compute_tile(const SharedHeader& h,
                  double* A, double* B, double* C,
                  uint64_t row0, uint64_t row1,
                  uint64_t col0, uint64_t col1)
{
    for (uint64_t i = row0; i < row1; ++i) {
        for (uint64_t j = col0; j < col1; ++j) {
            double sum = 0.0;
            for (uint64_t k = 0; k < h.K; ++k) {
                double a = A[i * h.K + k];
                double b = B[k * h.N + j];
                sum += a * b;
            }
            C[i * h.N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: worker <backing_file> <rank> <num_ranks>\n";
        return 1;
    }
    const char* path = argv[1];
    int rank = std::atoi(argv[2]);
    int num_ranks = std::atoi(argv[3]);
    if (rank < 0 || rank >= num_ranks) {
        std::cerr << "invalid rank\n";
        return 1;
    }

    int fd = ::open(path, O_RDWR, 0666);
    if (fd == -1) {
        std::cerr << "open failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    // Map just enough to read header first
    SharedHeader temp_header;
    if (::pread(fd, &temp_header, sizeof(temp_header), 0) != sizeof(temp_header)) {
        std::cerr << "pread header failed\n";
        ::close(fd);
        return 1;
    }

    size_t total_bytes = shared_region_size(temp_header);

    void* base = mmap(nullptr, total_bytes,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        std::cerr << "mmap failed: " << std::strerror(errno) << "\n";
        ::close(fd);
        return 1;
    }
    ::close(fd);

    SharedPtrs ptrs = get_ptrs_from_base(base);
    const SharedHeader& h = *ptrs.header;

    uint64_t total_tiles = h.num_tiles_m * h.num_tiles_n;
    uint64_t tiles_per_rank = (total_tiles + num_ranks - 1) / num_ranks;

    uint64_t start_tile = tiles_per_rank * rank;
    uint64_t end_tile = std::min(start_tile + tiles_per_rank, total_tiles);

    std::cout << "Rank " << rank << " computing tiles ["
              << start_tile << ", " << end_tile << ")\n";

    for (uint64_t tile_idx = start_tile; tile_idx < end_tile; ++tile_idx) {
        uint64_t tile_row = tile_idx / h.num_tiles_n;
        uint64_t tile_col = tile_idx % h.num_tiles_n;

        uint64_t row0 = tile_row * h.tile_m;
        uint64_t row1 = std::min(row0 + h.tile_m, h.M);
        uint64_t col0 = tile_col * h.tile_n;
        uint64_t col1 = std::min(col0 + h.tile_n, h.N);

        compute_tile(h, ptrs.A, ptrs.B, ptrs.C, row0, row1, col0, col1);

        ptrs.flags[tile_idx] = 1;

    }

    munmap(base, total_bytes);
    return 0;
}