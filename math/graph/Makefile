TARGET=a.out
FLAGS=-Wall -Werror -std=c++11
SOURCES=test.cpp graph.hpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	g++ $(FLAGS) test.cpp -o $(TARGET)
clean:
	rm -f $(TARGET)
