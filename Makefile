.PHONY: clean

CXX?=g++

INCLUDE=-I. -Iblaze -Isketch/include -Isketch -Isketch/libpopcnt

WARNINGS=-Wall -Wextra -Wno-unused-function

CXXFLAGS+= -O3 -march=native $(INCLUDE) -std=c++17 -fopenmp -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 $(WARNINGS)

all: psi2sketches parsepsi psi2sketches8bit psi2sketches32bit

%: src/%.cpp $(wildcard src/*.h)
	$(CXX) $(CXXFLAGS) -lz $< -o $@

psi2sketches8bit: src/psi2sketches.cpp $(wildcard src/*.h)
	$(CXX) $(CXXFLAGS) -lz $< -o $@ -DREGISTERTYPE=uint8_t

psi2sketches32bit: src/psi2sketches.cpp $(wildcard src/*.h)
	$(CXX) $(CXXFLAGS) -lz $< -o $@ -DREGISTERTYPE=uint32_t

clean:
	rm psi2sketches parsepsi
