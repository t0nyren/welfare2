CXX = g++
CXXFLAGS = -Wall -g -O3
#-std=gnu++98 -fPIC

LD_FLAGS = -Llib/cv -Llib/vl -lvl -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_flann -lopencv_objdetect
INCLUDE_FLAGS = -Iinclude/cv -Iinclude/vl -Iinclude/intraface
INTRAFACE_LIB = lib/intraface/libintraface.a
#LIBFLAGS = -fopenmp

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

OBJECTS =	$(BUILD_DIR)/mblbp-detect.o \
			$(BUILD_DIR)/binary_model_file.o \
			$(BUILD_DIR)/detector.o \
			$(BUILD_DIR)/encoder.o \
			$(BUILD_DIR)/recognizer.o \

TARGET1 = $(BIN_DIR)/buildModel
TARGET2 = $(BIN_DIR)/loadModel
TARGET3 = $(BIN_DIR)/buildModelSHM
TARGET4 = $(BIN_DIR)/loadModelSHM
TARGET5 = $(BIN_DIR)/faceDetect

.PHONY: all clean

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5)
	
$(TARGET1) : $(OBJECTS) $(BUILD_DIR)/buildModel.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) $(INTRAFACE_LIB)

$(TARGET2) : $(OBJECTS) $(BUILD_DIR)/loadModel.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) $(INTRAFACE_LIB)
	
$(TARGET3) : $(OBJECTS) $(BUILD_DIR)/buildModelSHM.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) $(INTRAFACE_LIB)

$(TARGET4) : $(OBJECTS) $(BUILD_DIR)/loadModelSHM.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) $(INTRAFACE_LIB)	

$(TARGET5) : $(OBJECTS) $(BUILD_DIR)/faceDetect.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) $(INTRAFACE_LIB)	
	
$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(INCLUDE_FLAGS)
clean:
	$(RM) $(TARGET) $(OBJECTS)
