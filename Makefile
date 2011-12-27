CC=gcc
CXX=g++
CFLAGS=-fopenmp
CXXFLAGS=-I/home/saiko/c++/armadillo-2.4.2/include
LDFLAGS=-fopenmp -lgomp -llapack -lblas -L/home/saiko/c++/armadillo-2.4.2 -larmadillo -lstdc++ 

all: bin/rbm

clean:
	rm -f obj/*
	rm -f bin/*

bin/rbm: obj/rbm.o
	$(CXX) -o $@ $^ $(LDFLAGS) 

obj/%.o: src/%.cpp src/%.hpp
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) -o $@ $<

.PHONY: all clean
