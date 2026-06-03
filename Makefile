CXX = g++

# Cờ biên dịch: Bật tối ưu (-O3), hiện cảnh báo (-Wall) và trỏ tới thư viện Xilinx
XILINX_INC = /tools/Xilinx/Vivado/2020.2/include
CXXFLAGS = -O3 -Wall -I$(XILINX_INC) -I./src

SRC_DIR = src
TB_DIR = tb
SOURCES = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(TB_DIR)/*.cpp)
EXECUTABLE = csim_run

all: $(EXECUTABLE)
$(EXECUTABLE): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)