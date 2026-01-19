# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -fopenmp
LDFLAGS = -fopenmp -lstdc++fs

# Targets
TARGET = corpus_miner
SRCS = main.cpp corpus_miner.cpp signal_handler.cpp
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
