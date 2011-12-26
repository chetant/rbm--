CC=gcc
CXX=g++
LDFLAGS=

all: bin/rbm

bin/rbm: obj/rbm.o
	$(CXX) -o $@ $(LDFLAGS) $^

obj/%.o: src/%.cpp
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

.PHONY: all
