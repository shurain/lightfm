CC=clang++
CFLAGS=-Wall -O3 -std=c++11 -m64

INC=-I include

BINDIR=bin
OBJDIR=build
SRCDIR=src

_OBJS=featurizer.o util.o fm.o
OBJS=$(patsubst %,$(OBJDIR)/%,$(_OBJS))

all: $(BINDIR)/test $(BINDIR)/lightfm

run:
	@$(BINDIR)/lightfm

test:
	@$(BINDIR)/test

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CC) -c $(CFLAGS) $(INC) -o $@ $<

$(BINDIR)/lightfm: $(OBJS) $(OBJDIR)/lightfm.o
	@mkdir -p $(BINDIR)
	$(CC) -o $(BINDIR)/lightfm $^

$(BINDIR)/test: $(OBJS) $(OBJDIR)/test.o
	@mkdir -p $(BINDIR)
	$(CC) -o $(BINDIR)/test $^

clean:
	@rm -rf $(OBJDIR) $(BINDIR)
