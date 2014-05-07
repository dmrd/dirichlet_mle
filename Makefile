CC = gcc
CFLAGS = -Wall -O3
LIB = -lgsl

all: test_dirichlet

test_dirichlet:
	$(CC) -o $@ $(LIB) $(CFLAGS) test_dirichlet.c dirichlet.c

clean:
	rm test_dirichlet
