all:
	./generate_patterns.py >| patterns
	./neuralnet.py

nonrandom:
	./generate_patterns.py -s 42 >| patterns
	./neuralnet.py -s 42
