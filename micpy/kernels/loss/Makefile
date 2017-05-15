SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

loss.o: $(OBJECTS)
	xild -r -o $@ $^

%.o: %.c
	icc -I../ -std=c99 -Wall -mmic -O2 -ipo -fPIC -fopenmp -c -o $@ $<

clean:
	rm *.o
