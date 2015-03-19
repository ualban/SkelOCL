SUBDIRS := $(wildcard */)

.PHONY : all $(SUBDIRS)

all : $(SUBDIRS)

$(SUBDIRS) :
	erlc -o $@ $@*.erl


