.PHONY: clean

CXX?=g++

INCLUDE=-I. -Iblaze -Isketch/include -Isketch -Isketch/libpopcnt

CXXFLAGS+= -O3 -march=native $(INCLUDE) -std=c++17 -fopenmp

all: psi2sketches parsepsi

%: src/%.cpp $(wildcard src/*.h)
	$(CXX) $(CXXFLAGS) -lz $< -o $@

clean:
	rm psi2sketches parsepsi
