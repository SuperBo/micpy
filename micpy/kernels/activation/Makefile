SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

activation.o: $(OBJECTS)
	$(AR) rcs $@ $^

%.o: %.c
	icc -I../ -Wall -mmic -fPIC -fopenmp -c -o $@ $<

clean:
	rm *.o
