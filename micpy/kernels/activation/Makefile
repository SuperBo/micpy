SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

activation.o: $(OBJECTS)
	xild -r -o $@ $^

%.o: %.c
	icc -I../ -Wall -mmic -O2 -ipo -fPIC -fopenmp -c -o $@ $<

clean:
	rm *.o
