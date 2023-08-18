.PHONY: all clean

CORE_DIR = Core
CORE_SRC := $(wildcard $(TRIAL_DIR)/*.cpp)
CORE_BINS := $(patsubst $(TRIAL_DIR)/%.cpp, obj/core/%.o, $(TRIAL_SRC))

HEAD_FOLDERS = $(CORE_DIR) Strategies Suppl
HEAD_LINK = $(patsubst %, -I%, $(HEAD_FOLDERS))

CC = g++ # compiler to use
CFLAGS = -c -O3 -I. $(HEAD_LINK) # flags to use at compliation
LINKERFLAG = -lm

all: obj $(CORE_BINS) obj/run.o obj/routines_lib.o 
	${CC} -o run $(CORE_BINS) obj/run.o ${LINKERFLAG}
	${CC} -shared -o routines_lib.so $(CORE_BINS) obj/routines_lib.o ${LINKERFLAG}

obj:
	@echo "Making obj directory..."
	mkdir obj
	mkdir obj/core

obj/run.o: run.cpp
	@echo "Compiling run.cpp..."
	${CC} ${CFLAGS} -o obj/run.o run.cpp

obj/routines_lib.o: routines_lib.cpp
	@echo "Compiling routines_lib.cpp..."
	${CC} ${CFLAGS} -fpic -o obj/routines_lib.o routines_lib.cpp

clean:
	@echo "Cleaning up..."
	rm -rvf obj
	rm run
	rm routines_lib.so