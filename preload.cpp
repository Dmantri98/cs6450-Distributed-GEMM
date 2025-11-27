#include <cstddef>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h> // open and close abd ftruncate
#include <cerrno>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>

struct SharedHeader {
    uint64_t M, N, K;
    uint64_t tile_m, tile_n;
    uint64_t num_tiles_m, num_tiles_n;
    // A -> MxK
    // B -> KxN
    // C -> MxN
};

size_t shared_region_size(const SharedHeader& h) {
    size_t header_bytes = sizeof(SharedHeader);
    size_t a_bytes = sizeof(double) * h.M * h.K;
    size_t b_bytes = sizeof(double) * h.K * h.N;
    size_t c_bytes = sizeof(double) * h.M * h.N;
    size_t flags_bytes = h.num_tiles_m * h.num_tiles_n; // uint8_t each

    return header_bytes + a_bytes + b_bytes + c_bytes + flags_bytes;
}

struct SharedPtrs {
    SharedHeader* header;
    double* A;
    double* B;
    double* C;
    uint8_t* flags; // one per tile
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
    // p += flags_bytes; // not strictly needed

    return out;
}

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

int main(int argc, char** argv){
    const char* path = argv[1];
    
    SharedHeader header{};
    header.M = 1024;
    header.N = 1024;
    header.K = 1024;
    header.tile_m = 256;
    header.tile_n = 256;
    header.num_tiles_m = (header.M + header.tile_m - 1) / header.tile_m;
    header.num_tiles_n = (header.N + header.tile_n - 1) / header.tile_n; // ceiling func

    size_t total_bytes = shared_region_size(header);
    std::cout << "shared region bytes: " << total_bytes << "\n";

    int fd = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1) {
        std::cerr << "open sad: " << std::strerror(errno) << "\n";
        return 1;
    }

    if (ftruncate(fd, total_bytes) == -1) {
        std::cerr << "ftruncate cant set the fd: "<< fd << "to total bytes" << total_bytes << std::strerror(errno) << "\n";
        ::close(fd);
        return 1;
    }

    void* base = mmap(nullptr, total_bytes,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        std::cerr << "mmap failed: " << std::strerror(errno) << "\n";
        ::close(fd);
        return 1;
    }
    ::close(fd); 

    // Write header at beginning
    auto* header_ptr = reinterpret_cast<SharedHeader*>(base);
    *header_ptr = header;

    SharedPtrs ptrs = get_ptrs_from_base(base);

    // Initialize A and B in-place in shared memory
    // I just use some dummy values here
    for (uint64_t i = 0; i < header.M; ++i) {
        for (uint64_t k = 0; k < header.K; ++k) {
            ptrs.A[i * header.K + k] = static_cast<double>((i + k) % 100) / 10.0;
        }
    }
    for (uint64_t k = 0; k < header.K; ++k) {
        for (uint64_t j = 0; j < header.N; ++j) {
            ptrs.B[k * header.N + j] = static_cast<double>((k - j + 1000) % 50);
        }
    }

    std::fill(ptrs.C, ptrs.C + header.M * header.N, 0.0);
    std::fill(ptrs.flags,
              ptrs.flags + header.num_tiles_m * header.num_tiles_n,
              uint8_t{0});

    // msync(base, total_bytes, MS_SYNC);

    std::cout << "Preload doneeeee.\n";

    // Keep mapping until exit, then unmap
    // munmap(base, total_bytes);
    return 0;

}