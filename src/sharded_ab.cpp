// compute_numa_fork.cpp
//
// Single-launch version:
//   - Run once, parent forks N worker processes.
//   - Parent sets up shared C and B_pool mappings (CXL-like pool) and zeros them.
//   - Children load their A and B shards into NUMA-local memory,
//     copy their B shard into the shared KxN B_pool, then compute their
//     block of C into the shared MxN C mapping.
//
// Expected files:
//   A.meta: "M K" on first line
//   B.meta: "K N" on first line
//   A.bin:  M*K floats, row-major
//   B.bin:  K*N floats, column-major (col 0, then col 1, ...)
//
// Usage example:
//   ./compute_numa_fork \
//     --nprocs 2 \
//     --a A.bin --a_meta A.meta \
//     --b B_colmajor.bin --b_meta B.meta \
//     --pool_node 1
//
// Compile:
//   g++ -O2 -std=c++20 compute_numa_fork.cpp -lnuma -o compute_numa_fork

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstddef>
#include <algorithm>
#include <cerrno>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include <numa.h>

struct Dim2 {
    std::size_t rows;
    std::size_t cols;
};

bool parse_dims_from_meta(const std::string& path, Dim2& dims) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: could not open meta file: " << path << "\n";
        return false;
    }
    std::size_t r, c;
    if (!(in >> r >> c)) {
        std::cerr << "Error: failed to read dims from meta file: " << path << "\n";
        return false;
    }
    dims.rows = r;
    dims.cols = c;
    return true;
}

bool file_size_matches(const std::string& path, std::size_t expected_bytes) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) {
        std::cerr << "Error: stat failed for " << path << ": "
                  << std::strerror(errno) << "\n";
        return false;
    }
    if (static_cast<std::size_t>(st.st_size) != expected_bytes) {
        std::cerr << "Error: file size mismatch for " << path
                  << " (expected " << expected_bytes
                  << " bytes, got " << static_cast<std::size_t>(st.st_size) << ")\n";
        return false;
    }
    return true;
}

// Read exactly 'bytes' from fd at file offset 'offset' into 'dst' using pread.
bool pread_all(int fd, void* dst, std::size_t bytes, std::size_t offset) {
    std::size_t total_read = 0;
    char* ptr = static_cast<char*>(dst);

    while (total_read < bytes) {
        ssize_t n = pread(fd,
                          ptr + total_read,
                          bytes - total_read,
                          static_cast<off_t>(offset + total_read));
        if (n < 0) {
            std::cerr << "Error: pread failed: " << std::strerror(errno) << "\n";
            return false;
        }
        if (n == 0) {
            std::cerr << "Error: unexpected EOF during pread\n";
            return false;
        }
        total_read += static_cast<std::size_t>(n);
    }
    return true;
}

// Compute this rank's block of C using:
//   A_local: local_rows x K, row-major
//   B_local: K x local_cols, column-major
//   C_global: M x N, row-major (shared)
// Rows: [row_start_global, row_end_global)
// Cols: [col_start_global, col_end_global)
void worker_compute_block(int rank,
                          std::size_t M,
                          std::size_t N,
                          std::size_t K,
                          std::size_t row_start_global,
                          std::size_t row_end_global,
                          std::size_t col_start_global,
                          std::size_t col_end_global,
                          const float* A_local,
                          const float* B_local,
                          float* C_global)
{
    std::size_t local_rows = (row_end_global > row_start_global)
                             ? (row_end_global - row_start_global)
                             : 0;
    std::size_t local_cols = (col_end_global > col_start_global)
                             ? (col_end_global - col_start_global)
                             : 0;

    if (local_rows == 0 || local_cols == 0) {
        std::cout << "Rank " << rank << ": no work (local_rows=" << local_rows
                  << ", local_cols=" << local_cols << ")\n";
        return;
    }

    std::cout << "Rank " << rank << ": computing rows [" << row_start_global
              << ", " << row_end_global << "), columns ["
              << col_start_global << ", " << col_end_global << ")\n";

    for (std::size_t i_local = 0; i_local < local_rows; ++i_local) {
        std::size_t i_global = row_start_global + i_local;
        for (std::size_t j_local = 0; j_local < local_cols; ++j_local) {
            std::size_t j_global = col_start_global + j_local;

            float acc = 0.0f;

            // A_local: row-major, A_local[i_local*K + k]
            // B_local: column-major, B_local[j_local*K + k]
            for (std::size_t k = 0; k < K; ++k) {
                float a_ik = A_local[i_local * K + k];
                float b_kj = B_local[j_local * K + k];
                acc += a_ik * b_kj;
            }

            C_global[i_global * N + j_global] = acc;
        }
    }
}

// Worker process: choose NUMA node, load local A/B shards, publish B to B_pool, compute C block.
void worker_process(int rank,
                    int nprocs,
                    const std::string& a_path,
                    const std::string& b_path,
                    std::size_t M,
                    std::size_t K,
                    std::size_t N,
                    float* C_global,
                    float* B_pool)
{
    int max_node = numa_max_node();
    int num_nodes = max_node + 1;
    int local_node = rank % num_nodes;

    std::cout << "Rank " << rank << ": binding to NUMA node " << local_node << "\n";

    // Pin this process to local_node (CPU + memory policy)
    numa_run_on_node(local_node);
    numa_set_preferred(local_node);

    // Row shard for A
    std::size_t rows_per = (M + nprocs - 1) / nprocs;
    std::size_t row_start = static_cast<std::size_t>(rank) * rows_per;
    std::size_t row_end   = std::min<std::size_t>(M, row_start + rows_per);
    std::size_t local_rows = (row_end > row_start) ? (row_end - row_start) : 0;

    // Column shard for B
    std::size_t cols_per = (N + nprocs - 1) / nprocs;
    std::size_t col_start = static_cast<std::size_t>(rank) * cols_per;
    std::size_t col_end   = std::min<std::size_t>(N, col_start + cols_per);
    std::size_t local_cols = (col_end > col_start) ? (col_end - col_start) : 0;

    std::cout << "Rank " << rank << ": A rows [" << row_start << ", " << row_end << "), "
              << "B cols [" << col_start << ", " << col_end << ")\n";

    // Open A.bin and B.bin
    int fdA = open(a_path.c_str(), O_RDONLY);
    if (fdA < 0) {
        std::cerr << "Rank " << rank << ": open failed for " << a_path
                  << ": " << std::strerror(errno) << "\n";
        _exit(1);
    }

    int fdB = open(b_path.c_str(), O_RDONLY);
    if (fdB < 0) {
        std::cerr << "Rank " << rank << ": open failed for " << b_path
                  << ": " << std::strerror(errno) << "\n";
        close(fdA);
        _exit(1);
    }

    // Allocate local A/B shards on local_node
    std::size_t bytesA_local = local_rows * K * sizeof(float);
    std::size_t bytesB_local = local_cols * K * sizeof(float); // K x local_cols, col-major

    float* A_local = nullptr;
    float* B_local = nullptr;

    if (bytesA_local > 0) {
        A_local = static_cast<float*>(numa_alloc_onnode(bytesA_local, local_node));
        if (!A_local) {
            std::cerr << "Rank " << rank << ": numa_alloc_onnode failed for A_local\n";
            close(fdA);
            close(fdB);
            _exit(1);
        }
    }
    if (bytesB_local > 0) {
        B_local = static_cast<float*>(numa_alloc_onnode(bytesB_local, local_node));
        if (!B_local) {
            std::cerr << "Rank " << rank << ": numa_alloc_onnode failed for B_local\n";
            if (A_local) numa_free(A_local, bytesA_local);
            close(fdA);
            close(fdB);
            _exit(1);
        }
    }

    // Read A shard: contiguous rows in row-major A.bin
    if (bytesA_local > 0) {
        std::size_t offsetA_bytes = row_start * K * sizeof(float);
        if (!pread_all(fdA, A_local, bytesA_local, offsetA_bytes)) {
            std::cerr << "Rank " << rank << ": reading A shard failed\n";
            if (A_local) numa_free(A_local, bytesA_local);
            if (B_local) numa_free(B_local, bytesB_local);
            close(fdA);
            close(fdB);
            _exit(1);
        }
    }

    // Read B shard: contiguous columns in column-major B.bin
    if (bytesB_local > 0) {
        std::size_t offsetB_bytes = col_start * K * sizeof(float);
        if (!pread_all(fdB, B_local, bytesB_local, offsetB_bytes)) {
            std::cerr << "Rank " << rank << ": reading B shard failed\n";
            if (A_local) numa_free(A_local, bytesA_local);
            if (B_local) numa_free(B_local, bytesB_local);
            close(fdA);
            close(fdB);
            _exit(1);
        }
    }

    close(fdA);
    close(fdB);

    // Publish this rank's B columns into shared B_pool (K x N, column-major)
    // B_local: K x local_cols, column-major
    // B_pool:  K x N,           column-major
    if (local_cols > 0) {
        for (std::size_t j_local = 0; j_local < local_cols; ++j_local) {
            std::size_t j_global = col_start + j_local;
            float* dst_col = B_pool + j_global * K;
            const float* src_col = B_local + j_local * K;
            std::memcpy(dst_col, src_col, K * sizeof(float));
        }
        std::cout << "Rank " << rank << ": published B columns ["
                  << col_start << ", " << col_end << ") into B_pool\n";
    }

    // For now, compute C only for this rank's row/col block using A_local/B_local
    worker_compute_block(rank,
                         M, N, K,
                         row_start, row_end,
                         col_start, col_end,
                         A_local ? A_local : nullptr,
                         B_local ? B_local : nullptr,
                         C_global);

    if (A_local) numa_free(A_local, bytesA_local);
    if (B_local) numa_free(B_local, bytesB_local);

    std::cout << "Rank " << rank << ": finished worker_process\n";
    _exit(0);
}

int main(int argc, char** argv) {
    std::string a_path;
    std::string b_path;
    std::string a_meta_path;
    std::string b_meta_path;
    int nprocs = 1;
    int pool_node = 0; // NUMA node for the shared "CXL pool"

    // Simple CLI parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto next_arg = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Error: missing value after " << arg << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--a") {
            a_path = next_arg();
        } else if (arg == "--b") {
            b_path = next_arg();
        } else if (arg == "--a_meta") {
            a_meta_path = next_arg();
        } else if (arg == "--b_meta") {
            b_meta_path = next_arg();
        } else if (arg == "--nprocs") {
            nprocs = std::atoi(next_arg());
        } else if (arg == "--pool_node") {
            pool_node = std::atoi(next_arg());
        } else {
            std::cerr << "Warning: unrecognized argument ignored: " << arg << "\n";
        }
    }

    if (a_path.empty() || b_path.empty() ||
        a_meta_path.empty() || b_meta_path.empty() ||
        nprocs <= 0) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " --nprocs P\n"
                  << "  --a A.bin --a_meta A.meta\n"
                  << "  --b B_colmajor.bin --b_meta B.meta\n"
                  << "  [--pool_node N]\n";
        return 1;
    }

    if (numa_available() < 0) {
        std::cerr << "Error: NUMA is not available on this system\n";
        return 1;
    }

    int max_node = numa_max_node();
    if (pool_node < 0 || pool_node > max_node) {
        std::cerr << "Error: pool_node out of range 0.." << max_node << "\n";
        return 1;
    }

    std::cout << "Parent: nprocs=" << nprocs
              << " pool_node=" << pool_node << "\n";

    Dim2 dimsA, dimsB;
    if (!parse_dims_from_meta(a_meta_path, dimsA)) return 1;
    if (!parse_dims_from_meta(b_meta_path, dimsB)) return 1;

    std::size_t M = dimsA.rows;
    std::size_t K = dimsA.cols;
    std::size_t K2 = dimsB.rows;
    std::size_t N = dimsB.cols;

    if (K != K2) {
        std::cerr << "Error: inner dimensions mismatch: A is "
                  << M << "x" << K << ", B is "
                  << K2 << "x" << N << "\n";
        return 1;
    }

    std::cout << "Parent: A " << M << " x " << K
              << ", B " << K << " x " << N << "\n";

    const std::size_t bytesA = M * K * sizeof(float);
    const std::size_t bytesB = K * N * sizeof(float);

    if (!file_size_matches(a_path, bytesA)) return 1;
    if (!file_size_matches(b_path, bytesB)) return 1;

    // ------------------------------------------------------------------
    // Parent: create and mmap shared C.bin (M x N, row-major)
    // ------------------------------------------------------------------
    const std::string c_path = "C.bin";
    int fdC = open(c_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fdC < 0) {
        std::cerr << "Error: open failed for " << c_path << ": "
                  << std::strerror(errno) << "\n";
        return 1;
    }

    const std::size_t bytesC = M * N * sizeof(float);
    if (ftruncate(fdC, static_cast<off_t>(bytesC)) != 0) {
        std::cerr << "Error: ftruncate failed for C: "
                  << std::strerror(errno) << "\n";
        close(fdC);
        return 1;
    }

    void* c_map = mmap(nullptr, bytesC,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED,
                       fdC, 0);
    if (c_map == MAP_FAILED) {
        std::cerr << "Error: mmap failed for C: "
                  << std::strerror(errno) << "\n";
        close(fdC);
        return 1;
    }
    float* C_global = static_cast<float*>(c_map);

    if (numa_tonode_memory(C_global, bytesC, pool_node) != 0) {
        std::cerr << "Warning: numa_tonode_memory failed for C: "
                  << std::strerror(errno) << "\n";
    }

    std::memset(C_global, 0, bytesC);
    std::cout << "Parent: C.bin created, mapped, zeroed\n";

    // ------------------------------------------------------------------
    // Parent: create and mmap shared B_pool.bin (K x N, column-major)
    // ------------------------------------------------------------------
    const std::string bpool_path = "B_pool.bin";
    int fdBpool = open(bpool_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fdBpool < 0) {
        std::cerr << "Error: open failed for " << bpool_path << ": "
                  << std::strerror(errno) << "\n";
        munmap(c_map, bytesC);
        close(fdC);
        return 1;
    }

    const std::size_t bytesB_pool = K * N * sizeof(float);
    if (ftruncate(fdBpool, static_cast<off_t>(bytesB_pool)) != 0) {
        std::cerr << "Error: ftruncate failed for B_pool: "
                  << std::strerror(errno) << "\n";
        munmap(c_map, bytesC);
        close(fdC);
        close(fdBpool);
        return 1;
    }

    void* bpool_map = mmap(nullptr, bytesB_pool,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED,
                           fdBpool, 0);
    if (bpool_map == MAP_FAILED) {
        std::cerr << "Error: mmap failed for B_pool: "
                  << std::strerror(errno) << "\n";
        munmap(c_map, bytesC);
        close(fdC);
        close(fdBpool);
        return 1;
    }
    float* B_pool = static_cast<float*>(bpool_map);

    if (numa_tonode_memory(B_pool, bytesB_pool, pool_node) != 0) {
        std::cerr << "Warning: numa_tonode_memory failed for B_pool: "
                  << std::strerror(errno) << "\n";
    }

    std::memset(B_pool, 0, bytesB_pool);
    std::cout << "Parent: B_pool.bin created, mapped, zeroed\n";

    // ------------------------------------------------------------------
    // Fork worker processes
    // ------------------------------------------------------------------
    std::vector<pid_t> children;
    children.reserve(nprocs);

    for (int rank = 0; rank < nprocs; ++rank) {
        pid_t pid = fork();
        if (pid < 0) {
            std::cerr << "Error: fork failed: " << std::strerror(errno) << "\n";
            // Best-effort cleanup: wait for any already-forked children
            for (pid_t cpid : children) {
                int status;
                waitpid(cpid, &status, 0);
            }
            munmap(c_map, bytesC);
            munmap(bpool_map, bytesB_pool);
            close(fdC);
            close(fdBpool);
            return 1;
        }

        if (pid == 0) {
            // Child: run worker_process and exit
            worker_process(rank, nprocs,
                           a_path, b_path,
                           M, K, N,
                           C_global, B_pool);
            // worker_process calls _exit
        }

        // Parent
        children.push_back(pid);
    }

    // Parent: wait for all children
    for (pid_t cpid : children) {
        int status = 0;
        pid_t w = waitpid(cpid, &status, 0);
        if (w < 0) {
            std::cerr << "Warning: waitpid failed: "
                      << std::strerror(errno) << "\n";
        }
    }

    std::cout << "Parent: all workers finished. C.bin and B_pool.bin ready.\n";

    // Cleanup mappings
    munmap(c_map, bytesC);
    munmap(bpool_map, bytesB_pool);
    close(fdC);
    close(fdBpool);

    return 0;
}
