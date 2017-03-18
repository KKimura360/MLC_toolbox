INC=-I../Tools/c++
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread

all: clean fastXML_train fastXML_test

fastXML_train:
	g++ -o fastXML_train $(CXXFLAGS) $(INC) fastXML_train.cpp fastXML.cpp $(LIBFLAGS)

fastXML_test:
	g++ -o fastXML_test $(CXXFLAGS) $(INC) fastXML_test.cpp fastXML.cpp $(LIBFLAGS)

clean:
	rm -f fastXML_train fastXML_test

