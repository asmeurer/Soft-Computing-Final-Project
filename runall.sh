#!/sw/bin/bash
# We have to run bash >= 4 for the for i in  syntax.

# Run the neural net with [10, 10, 10, 1] ... [80, 80, 80, 1] indefinitely

while true
do
    ./generate_patterns.py patterns
    ./generate_patterns.py patterns_test
    for i in {10..80..10}
    do
        echo "NODES = [$i $i $i 1]"
        ./neuralnet.py patterns -t patterns_test -N $i $i $i 1
    done
done
