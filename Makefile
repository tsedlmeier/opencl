CFLAGS = `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4`
% : %.cpp
	g++ $(CFLAGS) -std=c++17 -Wno-psabi -o $@ $< $(LIBS)
