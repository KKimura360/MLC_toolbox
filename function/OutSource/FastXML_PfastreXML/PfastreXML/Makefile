SRC=../FastXML
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread
CXXINCPATH=../Tools/c++
INC=-I$(CXXINCPATH) -I$(SRC)

all: clean PfastreXML_train PfastreXML_test

PfastreXML_train:
	$(CXX) -o PfastreXML_train $(CXXFLAGS) $(INC) PfastreXML_train.cpp $(SRC)/fastXML.cpp $(LIBFLAGS)

PfastreXML_test:
	$(CXX) -o PfastreXML_test $(CXXFLAGS) $(INC) PfastreXML_test.cpp $(SRC)/fastXML.cpp $(LIBFLAGS)

clean:
	rm -f PfastreXML_train PfastreXML_test
