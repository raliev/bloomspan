# Настройки компилятора
CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -pthread -flto -Wall -Wextra

# Если планируете использовать OpenMP для параллельных циклов, 
# раскомментируйте строку ниже и добавьте -fopenmp в CXXFLAGS
# CXXFLAGS += -fopenmp

# Название бинарного файла
TARGET = corpus_miner

# Директории
SRCS = main.cpp corpus_miner.cpp signal_handler.cpp
OBJS = $(SRCS:.cpp=.o)

# Основная цель
all: $(TARGET)

# Сборка исполняемого файла
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	@echo "--------------------------------------------------"
	@echo "Build complete: ./$(TARGET)"
	@echo "Optimization: -O3, Link-Time Optimization (LTO) enabled"
	@echo "--------------------------------------------------"

# Компиляция объектных файлов
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Очистка
clean:
	rm -f $(OBJS) $(TARGET)

# Режим отладки
debug: CXXFLAGS = -std=c++20 -g -pthread -Wall
debug: clean all

.PHONY: all clean debug
