Compile command for preprocessing:

g++ -O3 -std=c++17 pre_process.cpp mmio.c -o mtx2bin

Example Usage on medium sized matrix:

./mtx2bin ../dense_mtx_examples/dense_100x100.mtx ../dense_mtx_examples/med.bin ../dense_mtx_examples/med_metadata.txt