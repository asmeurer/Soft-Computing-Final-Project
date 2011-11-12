all: patterns network

nonrandom: patterns-nonrandom network-nonrandom

patterns:
	./generate_patterns.py

patterns-nonrandom:
	./generate_patterns.py -s 42

patterns-small:
	./generate_patterns.py -n 10

patterns-small-nonrandom:
	./generate_patterns.py -s 42 -n 10

network:
	./neuralnet.py patterns

network-nonrandom:
	./neuralnet.py -s 42 patterns

clean:
	-rm -f patterns
