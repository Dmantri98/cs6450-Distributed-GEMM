// compute_numa_fork.cpp
//
// Single-launch version with shard table:
//   - Tracks which process owns which shard of B in shared memory
//   - Each worker fetches all B shards to compute its full row block of C

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

// Compute this rank's row block of C using a single B shard
void worker_compute_block(int rank,
                          std::size_t M,
                          std::size_t N,
                          std::size_t K,
                          std::size_t row_start_global,
                          std::size_t row_end_global,
                          const float* A_local,
                          float* C_global,
                          float* B_shard,
                          std::size_t col_start,
                          std::size_t col_end)
{
    std::size_t local_rows = (row_end_global > row_start_global)
                             ? (row_end_global - row_start_global)
                             : 0;
    std::size_t local_cols = (col_end > col_start) ? (col_end - col_start) : 0;

    if (local_rows == 0 || local_cols == 0) {
        std::cout << "Rank " << rank << ": no work (local_rows=" << local_rows
                  << ", local_cols=" << local_cols << ")\n";
        return;
    }

    std::cout << "Rank " << rank << ": computing rows [" << row_start_global
              << ", " << row_end_global << ") for columns [" << col_start
              << ", " << col_end << ")\n";

    for (std::size_t i_local = 0; i_local < local_rows; ++i_local) {
        std::size_t i_global = row_start_global + i_local;

        for (std::size_t j_local = 0; j_local < local_cols; ++j_local) {
            std::size_t j_global = col_start + j_local;
            float acc = 0.0f;

            for (std::size_t k = 0; k < K; ++k) {
                float a_ik = A_local[i_local * K + k];
                float b_kj = B_shard[j_local * K + k]; // B is column-major
                acc += a_ik * b_kj;
            }

            C_global[i_global * N + j_global] = acc;
        }
    }
}


// Worker process: bind to NUMA node, load local A, compute C using on-demand B shards
void worker_process(int rank,
                    int nprocs,
                    const std::string& a_path,
                    std::size_t M,
                    std::size_t K,
                    std::size_t N,
                    float* C_global)
{
    int max_node = numa_max_node();
    int num_nodes = max_node + 1;
    int local_node = rank % num_nodes;

    std::cout << "Rank " << rank << ": binding to NUMA node " << local_node << "\n";

    numa_run_on_node(local_node);
    numa_set_preferred(local_node);

    // Row shard for A
    std::size_t rows_per = (M + nprocs - 1) / nprocs;
    std::size_t row_start = rank * rows_per;
    std::size_t row_end   = std::min<std::size_t>(M, row_start + rows_per);
    std::size_t local_rows = (row_end > row_start) ? (row_end - row_start) : 0;

    std::cout << "Rank " << rank << ": A rows [" << row_start << ", " << row_end << ")\n";

    // Open A.bin
    int fdA = open(a_path.c_str(), O_RDONLY);
    if (fdA < 0) {
        std::cerr << "Rank " << rank << ": failed to open A\n";
        _exit(1);
    }

    // Allocate local A
    float* A_local = nullptr;
    if (local_rows > 0)
        A_local = static_cast<float*>(numa_alloc_onnode(local_rows * K * sizeof(float), local_node));

    // Read A shard
    if (local_rows > 0) {
        std::size_t offsetA_bytes = row_start * K * sizeof(float);
        pread_all(fdA, A_local, local_rows * K * sizeof(float), offsetA_bytes);
    }
    close(fdA);

    // Loop over all B shards on-demand
    std::size_t cols_per = (N + nprocs - 1) / nprocs;
    for (int owner = 0; owner < nprocs; ++owner) {
        std::size_t col_start = owner * cols_per;
        std::size_t col_end   = std::min<std::size_t>(N, col_start + cols_per);
        std::size_t local_cols = (col_end > col_start) ? (col_end - col_start) : 0;
        if (local_cols == 0) continue;

        std::string filename = "B_shard_" + std::to_string(owner) + ".bin";
        int fdB = open(filename.c_str(), O_RDONLY);
        if (fdB < 0) {
            std::cerr << "Rank " << rank << ": cannot open B shard " << filename << "\n";
            _exit(1);
        }

        size_t bytes = K * local_cols * sizeof(float);
        void* map = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fdB, 0);
        close(fdB);
        if (map == MAP_FAILED) {
            std::cerr << "Rank " << rank << ": mmap failed for " << filename << "\n";
            _exit(1);
        }

        float* B_shard = static_cast<float*>(map);

        // Compute partial C block
        worker_compute_block(rank, M, N, K, row_start, row_end, A_local, C_global, B_shard, col_start, col_end);

        munmap(B_shard, bytes);
    }

    if (A_local) numa_free(A_local, local_rows * K * sizeof(float));

    std::cout << "Rank " << rank << ": finished worker_process\n";
    _exit(0);
}



int main(int argc, char** argv) {
    std::string a_path, b_path, a_meta_path, b_meta_path;
    int nprocs = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_arg = [&]() -> const char* { return argv[++i]; };
        if (arg == "--a") a_path = next_arg();
        else if (arg == "--b") b_path = next_arg();
        else if (arg == "--a_meta") a_meta_path = next_arg();
        else if (arg == "--b_meta") b_meta_path = next_arg();
        else if (arg == "--nprocs") nprocs = std::atoi(next_arg());
    }

    Dim2 dimsA, dimsB;
    parse_dims_from_meta(a_meta_path, dimsA);
    parse_dims_from_meta(b_meta_path, dimsB);

    std::size_t M = dimsA.rows;
    std::size_t K = dimsA.cols;
    std::size_t N = dimsB.cols;

    // Create shared C
    int fdC = open("C.bin", O_CREAT | O_RDWR | O_TRUNC, 0666);
    ftruncate(fdC, M*N*sizeof(float));
    float* C_global = static_cast<float*>(mmap(nullptr, M*N*sizeof(float),
                                               PROT_READ | PROT_WRITE,
                                               MAP_SHARED, fdC, 0));
    std::memset(C_global, 0, M*N*sizeof(float));

    // --- Pre-shard B to disk ---
    int fdB = open(b_path.c_str(), O_RDONLY);
    if (fdB < 0) {
        std::cerr << "Failed to open B file\n";
        return 1;
    }

    for (int rank = 0; rank < nprocs; ++rank) {
        std::size_t cols_per = (N + nprocs - 1) / nprocs;
        std::size_t col_start = rank * cols_per;
        std::size_t col_end   = std::min(N, col_start + cols_per);
        std::size_t local_cols = col_end - col_start;

        if (local_cols == 0) continue;

        float* B_local = new float[local_cols * K];
        std::size_t offset_bytes = col_start * K * sizeof(float);
        pread_all(fdB, B_local, local_cols*K*sizeof(float), offset_bytes);

        std::string shard_filename = "B_shard_" + std::to_string(rank) + ".bin";
        int fdShard = open(shard_filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0666);
        if (fdShard < 0) {
            std::cerr << "Failed to create shard file: " << shard_filename << "\n";
            delete[] B_local;
            continue;
        }
        write(fdShard, B_local, local_cols*K*sizeof(float));
        close(fdShard);
        delete[] B_local;
    }
    close(fdB);

    // Fork workers
    std::vector<pid_t> children;
    for (int rank = 0; rank < nprocs; ++rank) {
        pid_t pid = fork();
        if (pid == 0) {
            worker_process(rank, nprocs, a_path, M, K, N, C_global);
        }
        children.push_back(pid);
    }

    for (pid_t cpid : children) {
        waitpid(cpid, nullptr, 0);
    }

    std::cout << "Parent: all workers finished\n";

    munmap(C_global, M*N*sizeof(float));
    close(fdC);

    return 0;
}
