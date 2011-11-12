all: patterns network

nonrandom: patterns-nonrandom network

patterns:
	./generate_patterns.py >| patterns

patterns-nonrandom:
	./generate_patterns.py -s 42 >| patterns

network:
	./neuralnet.py patterns

network-nonrandom:
	./neuralnet.py -s 42 patterns

clean:
	-rm -f patterns
