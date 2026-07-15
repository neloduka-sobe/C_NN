CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Isrc
LDFLAGS = -lm

SRC = src/engine.c src/nn.c src/main.c
OBJ = $(SRC:.c=.o)
TARGET = micrograd

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
