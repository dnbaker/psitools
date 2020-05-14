CXX?=g++

INCLUDE=-I. -Iblaze

CXXFLAGS+= -O3 -march=native $(INCLUDE) -std=c++17

%: src/%.cpp
	$(CXX) $(CXXFLAGS) -lz $< -o $@
