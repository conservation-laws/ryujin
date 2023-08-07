##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2023 by the ryujin authors
##

SHELL:=bash
default: all

.PHONY: default

SOURCEDIR:=$(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
BUILDDIR:=build

ifeq (, $(shell which ninja))
  GENERATOR:="Unix Makefiles"
  MAKE_COMMAND:=make
  MAKE_FILE:=Makefile
else
  GENERATOR:=Ninja
  MAKE_COMMAND:=ninja
  MAKE_FILE:=build.ninja
endif

##########################################################################

cleanup_insource:
	@rm -f $(SOURCEDIR)/CMakeCache.txt
	@rm -rf $(SOURCEDIR)/CMakeFiles

rebuild_cache:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake -G$(GENERATOR) $(SOURCEDIR)

edit_cache:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && ccmake -G$(GENERATOR) $(SOURCEDIR)

$(BUILDDIR)/$(MAKE_FILE):
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake -G$(GENERATOR) $(SOURCEDIR)

debug:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake -DCMAKE_BUILD_TYPE=Debug -G$(GENERATOR) $(SOURCEDIR)
	@cd $(BUILDDIR) && $(MAKE_COMMAND)

release:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake -DCMAKE_BUILD_TYPE=Release -G$(GENERATOR) $(SOURCEDIR)
	@cd $(BUILDDIR) && $(MAKE_COMMAND)

Makefile:
	

%: cleanup_insource $(BUILDDIR)/$(MAKE_FILE)
	@cd $(BUILDDIR) && $(MAKE_COMMAND) $@

.PHONY: cleanup_insource rebuild_cache edit_cache debug release

##########################################################################

indent:
	@clang-format -i source/**/*.h source/**/*.cc

.PHONY: indent

##########################################################################
