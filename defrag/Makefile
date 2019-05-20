INC=-I../Tools/c++
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread

all: clean defrag_clustering defrag_agglomeration

defrag_clustering:
	$(CXX) -o defrag_clustering $(CXXFLAGS) $(INC) defrag_clustering.cpp defrag.cpp $(LIBFLAGS)

defrag_agglomeration:
	$(CXX) -o defrag_agglomeration $(CXXFLAGS) $(INC) defrag_agglomeration.cpp defrag.cpp $(LIBFLAGS)

clean:
	rm -f defrag_clustering defrag_agglomeration
