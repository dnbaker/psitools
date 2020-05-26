.PHONY: clean

CXX?=g++

INCLUDE=-I. -Iblaze -Isketch/include -Isketch -Isketch/libpopcnt

WARNINGS=-Wall -Wextra

CXXFLAGS+= -O3 -march=native $(INCLUDE) -std=c++17 -fopenmp -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 $(WARNINGS)

all: psi2sketches parsepsi

%: src/%.cpp $(wildcard src/*.h)
	$(CXX) $(CXXFLAGS) -lz $< -o $@

clean:
	rm psi2sketches parsepsi
