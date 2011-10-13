all:
	./generate_problems.py >| problems
	./neuralnet.py
