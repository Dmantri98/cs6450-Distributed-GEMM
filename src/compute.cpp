#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>


int main(int argc, char** argv) {
    // 1. Create or open shared file
    int fd = open("/path/to/swcc_file.bin", O_CREAT | O_RDWR, 0666);

    // 2. Allocate size (truncate file)
    size_t size = TOTAL_HWCC_BYTES + TOTAL_SWCC_BYTES;
    ftruncate(fd, size);

    // 3. mmap file
    void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // 4. Initialize HWcc metadata structures (mark Empty, set offsets)
    init_hwcc(base);

    // 5. Fill A and B tiles into SWcc region
    float* A = get_A_pointer(base);
    float* B = get_B_pointer(base);
    fill_random(A, A_size);
    fill_random(B, B_size);

    // 6. Mark all A and B entries as Ready in HWcc
    set_state_ready(hwcc_table_for_A);
    set_state_ready(hwcc_table_for_B);

    // 7. Done
    munmap(base, size);
    close(fd);
    return 0;
}
