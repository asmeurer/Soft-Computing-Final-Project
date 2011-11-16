#!/sw/bin/bash
# We have to run bash >= 4 for the for i in  syntax.

# Run the neural net with [10, 10, 10, 1] ... [80, 80, 80, 1] indefinitely

for i in {1..50}
do
    echo "Iteration $i"
    ./generate_patterns.py patterns
    ./generate_patterns.py patterns_test
    for j in {10..80..10}
    do
        echo "NODES = [$j $j $j 1]"
        ./neuralnet.py patterns -t patterns_test -N $j $j $j 1
    done
done
