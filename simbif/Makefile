CXX = g++
CXXFLAGS = -O3 -march=native -flto -std=c++17 -Wall -Wextra
TARGETS = sb sa bench

.PHONY: all clean run bench-run

all: $(TARGETS)

sb: sb.cpp
	$(CXX) $(CXXFLAGS) sb.cpp -o sb

sa: sa.cpp
	$(CXX) $(CXXFLAGS) sa.cpp -o sa

bench: bench.cpp
	$(CXX) $(CXXFLAGS) bench.cpp -o bench

clean:
	rm -f $(TARGETS)

run: sb
	./sb

bench-run: bench
	./bench
