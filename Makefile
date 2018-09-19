SHELL:=bash
default: all

SOURCEDIR:=$(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
BUILDDIR:=build
GENERATOR:=Ninja
MAKE_COMMAND:=ninja
MAKE_FILE:=build.ninja

EXECUTABLE:=ryujin
PARAMETER_FILE:=$(EXECUTABLE).prm

edit: all
	@if [ \! -f $(BUILDDIR)/run/$(PARAMETER_FILE) ]; then cd $(BUILDDIR)/run; ./$(EXECUTABLE); fi
	@vim $(BUILDDIR)/run/$(PARAMETER_FILE)

run: all
	@cd $(BUILDDIR)/run && time ./$(EXECUTABLE)

run_clean:
	@rm -f $(BUILDDIR)/run/*.(vtk|vtu|log|txt|gnuplot)
	@rm -f $(BUILDDIR)/run/*-*.*

.PHONY: default edit run run_clean

##########################################################################

rebuild_cache:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake -G$(GENERATOR) $(SOURCEDIR)

edit_cache:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && ccmake -G$(GENERATOR) $(SOURCEDIR)

$(BUILDDIR)/$(MAKE_FILE):
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake -G$(GENERATOR) $(SOURCEDIR)

release:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake -DCMAKE_BUILD_TYPE=Release -G$(GENERATOR) $(SOURCEDIR)
	@cd $(BUILDDIR) && $(MAKE_COMMAND)

debug:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake -DCMAKE_BUILD_TYPE=Debug -G$(GENERATOR) $(SOURCEDIR)
	@cd $(BUILDDIR) && $(MAKE_COMMAND)

Makefile:
	

%: $(BUILDDIR)/$(MAKE_FILE)
	@cd $(BUILDDIR) && ninja $@

.PHONY: rebuild_cache edit_cache release debug
