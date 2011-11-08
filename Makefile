all:
	./generate_patterns.py >| patterns
	./neuralnet.py
