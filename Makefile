.PHONY: all data benchmark test

all: data benchmark test

data:
	ghgmatch generate

benchmark:
	ghgmatch benchmark

test:
	pytest -q
