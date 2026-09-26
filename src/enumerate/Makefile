FATHOM ?= $(HOME)/ext/Fathom/src
CFLAGS = -O3 -march=native -fopenmp -Wall -Wno-unused-function -I$(FATHOM)
all: tbenum
tbenum: tbenum.c
	$(CC) $(CFLAGS) -o $@ $< -lm
clean:
	rm -f tbenum
