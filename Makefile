CC=gcc
CXX=g++
GPERF_CFLAGS=-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
# CFLAGS=-O3
CFLAGS=-g
GPERF_CXXFLAGS=-I/home/saiko/bin/gperf/include
CXXFLAGS=-std=c++0x
GPERF_LDFLAGS=-L/home/saiko/bin/gperf/lib -lprofiler -ltcmalloc
LDFLAGS=-lboost_serialization -lboost_iostreams -lboost_filesystem -lboost_thread -lboost_system -lstdc++

all: bin/rbm

clean:
	rm -f obj/*
	rm -f bin/*

bin/rbm: obj/rbm.o obj/mnist.o
	$(CXX) -o $@ $^ $(LDFLAGS) 

obj/rbm.o: src/rbm.cpp src/rbm.hpp src/mnist.hpp src/gnuplot_i.hpp
obj/%.o: src/%.cpp src/%.hpp
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) -o $@ $<

.PHONY: all clean
