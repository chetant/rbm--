CC=gcc
CXX=g++
NVCC=nvcc
NVCC_GCC_PATH=--compiler-bindir /home/saiko/bin/cuda/gcc

GPERF_CFLAGS=-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
# CFLAGS=-O3
CFLAGS=-g

CUDA_INCLUDE_FLAGS=-I/home/saiko/bin/cuda/include
GPERF_INCLUDE_FLAGS=-I/home/saiko/bin/gperf/include
INCLUDE_FLAGS=$(CUDA_INCLUDE_FLAGS)

CXXFLAGS=-std=c++0x $(INCLUDE_FLAGS)
NVCCFLAGS=$(INCLUDE_FLAGS)

GPERF_LDFLAGS=-L/home/saiko/bin/gperf/lib -lprofiler -ltcmalloc
CUDA_LDFLAGS=-L/home/saiko/bin/cuda/lib -lcurand -lcudart -lcublas

LDFLAGS=$(CUDA_LDFLAGS) -lboost_serialization -lboost_iostreams -lboost_filesystem -lboost_thread -lboost_system -lstdc++

all: bin/rbm

clean:
	rm -f obj/*
	rm -f bin/*

bin/rbm: obj/rbm.o obj/mnist.o obj/cuda_utils.cu.o
	$(CXX) -o $@ $^ $(LDFLAGS) 

obj/rbm.o: src/rbm.cpp src/rbm.hpp src/mnist.hpp src/gnuplot_i.hpp src/cuda_utils.hpp
obj/%.cu.o: src/%.cu src/%.hpp
	$(NVCC) -c $(NVCC_GCC_PATH) $(CFLAGS) $(NVCCFLAGS) -o $@ $<
obj/%.o: src/%.cpp src/%.hpp
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) -o $@ $<

.PHONY: all clean
