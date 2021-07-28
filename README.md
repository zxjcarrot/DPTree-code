# DPTree-code
Source code of the VLDB 2020 paper: [DPTree: Differential Indexing for Persistent Memory](http://www.vldb.org/pvldb/vol13/p421-zhou.pdf)

# Building

DPTree depends on a few libraries:
* tcmalloc
* stx-btree
* Intel's TBB library

To install the dependencies on Ubuntu 18.04, run following commands:

```bash
sudo apt install stx-btree-dev
sudo apt install libgoogle-perftools-dev
sudo apt install libbtbb-dev
```
Once the dependencies are installed, you can compile the tests and benchmarks using:
```bash
mkdir build
cd build
cmake ..
make -j 10
```
This produces `dptree` and `concur_dptree` executables for benchmarks under the build directory.
Checkout the source code of `test/dptree.cxx` or `test/concur_dptree.cxx` on how to use them.