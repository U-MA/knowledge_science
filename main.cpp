#include <iostream>
#include <neural_network.h>

int main()
{
    // test set is XOR
    std::vector<std::vector<double>> inputs = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    std::vector<std::vector<double>> trainings = {
        { 0 },
        { 1 },
        { 1 },
        { 0 }
    };

    neural_network nn(2, 2, 1);

    for (int i=0; i < 10000; i++) {
        for (int j=0; j < 4; j++)
            nn.learn(inputs[j], trainings[j]);
    }
    for (int i=0; i < 4; i++) {
        std::cout << inputs[i][0] << ", " << inputs[i][1] << " | " << nn.input(inputs[i])[0] << std::endl;
    }
}
