all: patterns network

nonrandom: patterns-nonrandom network

patterns:
	./generate_patterns.py >| patterns

patterns-nonrandom:
	./generate_patterns.py -s 42 >| patterns

patterns-small:
	./generate_patterns.py -n 10 >| patterns

patterns-small-nonrandom:
	./generate_patterns.py -s 42 -n 10 >| patterns

network:
	./neuralnet.py patterns

network-nonrandom:
	./neuralnet.py -s 42 patterns

clean:
	-rm -f patterns
