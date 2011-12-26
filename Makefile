CC=gcc
CXX=g++
CFLAGS=
CXXFLAGS=-I/home/saiko/c++/armadillo-2.4.2/include
LDFLAGS=-L/home/saiko/c++/armadillo-2.4.2 -larmadillo

all: bin/rbm

clean:
	rm obj/*
	rm bin/*

bin/rbm: obj/rbm.o
	$(CXX) -o $@ $(LDFLAGS) $^

obj/%.o: src/%.cpp src/%.hpp
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) -o $@ $<

.PHONY: all clean
