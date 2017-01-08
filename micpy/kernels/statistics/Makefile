SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

statistics.o: $(OBJECTS)
	$(AR) rcs $@ $^

%.o: %.c
	icc -I../ -Wall -mmic -fPIC -fopenmp -c -o $@ $<

clean:
	rm *.o
