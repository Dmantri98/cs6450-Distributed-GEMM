// convert_dense_mtx_to_bin.cpp
//
// Usage:
//   ./mtx2bin input.mtx output.bin metadata.txt
//
// output.bin:  row-major float32 data, M * N elements
// metadata.txt: single line "M N\n"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "../include/mmio.h"   // make sure mmio.h / mmio.c are in your include path


struct MatrixDims {
    int M;
    int N;
};

// Read dense Matrix Market (array format), values in column-major order,
// return as a double buffer in column-major layout.
static std::vector<double> read_mtx_dense_col_major(const std::string &filename,
                                                    MatrixDims &dims) {
    FILE *f = std::fopen(filename.c_str(), "r");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        std::fclose(f);
        throw std::runtime_error("Could not process Matrix Market banner");
    }

    if (!mm_is_matrix(matcode) || !mm_is_array(matcode)) {
        std::fclose(f);
        throw std::runtime_error("File is not a dense 'array' Matrix Market matrix");
    }

    if (!(mm_is_real(matcode) || mm_is_integer(matcode))) {
        std::fclose(f);
        throw std::runtime_error("Only real or integer matrices are supported (no complex)");
    }

    int M, N;
    if (mm_read_mtx_array_size(f, &M, &N) != 0) {
        std::fclose(f);
        throw std::runtime_error("Could not read matrix dimensions");
    }

    dims.M = M;
    dims.N = N;

    const std::size_t total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    std::vector<double> data_col_major(total);

    // Matrix Market array format:
    //   values are listed one per line, scanning DOWN each column.
    // So the k-th value read is at column-major index k.
    for (std::size_t k = 0; k < total; ++k) {
        double val;
        if (std::fscanf(f, "%lf", &val) != 1) {
            std::fclose(f);
            throw std::runtime_error("Unexpected EOF or parse error while reading values");
        }
        data_col_major[k] = val;
    }

    std::fclose(f);
    return data_col_major;
}

// Convert column-major double buffer to row-major float32 and write out.
static void write_row_major_f32_bin(const std::string &bin_filename,
                                    const std::vector<double> &col_major,
                                    const MatrixDims &dims) {
    const int M = dims.M;
    const int N = dims.N;
    const std::size_t total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

    std::vector<float> row_major(total);

    // col-major index: j * M + i
    // row-major index: i * N + j
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            std::size_t col_idx = static_cast<std::size_t>(j) * M + i;
            std::size_t row_idx = static_cast<std::size_t>(i) * N + j;
            row_major[row_idx] = static_cast<float>(col_major[col_idx]);
        }
    }

    std::ofstream ofs(bin_filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open output bin file: " + bin_filename);
    }

    ofs.write(reinterpret_cast<const char *>(row_major.data()),
              static_cast<std::streamsize>(total * sizeof(float)));

    if (!ofs) {
        throw std::runtime_error("Error while writing bin file: " + bin_filename);
    }
}

// Write metadata file: single line "M N\n"
static void write_metadata(const std::string &meta_filename, const MatrixDims &dims) {
    std::ofstream meta(meta_filename);
    if (!meta) {
        throw std::runtime_error("Cannot open metadata file: " + meta_filename);
    }
    meta << dims.M << " " << dims.N << "\n";
    if (!meta) {
        throw std::runtime_error("Error while writing metadata file: " + meta_filename);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "Usage: %s input.mtx output.bin metadata.txt\n"
                     "  input.mtx    : dense Matrix Market (array) file\n"
                     "  output.bin   : row-major float32 data (M*N elements)\n"
                     "  metadata.txt : text file with 'M N' on one line\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    const std::string input_mtx  = argv[1];
    const std::string output_bin = argv[2];
    const std::string meta_file  = argv[3];

    try {
        MatrixDims dims;
        auto col_major = read_mtx_dense_col_major(input_mtx, dims);
        write_row_major_f32_bin(output_bin, col_major, dims);
        write_metadata(meta_file, dims);
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
