#ifndef KNOWLEDGE_SCIENCE_NEURAL_NETWORK
#define KNOWLEDGE_SCIENCE_NEURAL_NETWORK

#include <iostream>
#include <vector>

// three layered neural network
class neural_network
{
public:
    neural_network(std::size_t in_size, std::size_t mid_size, std::size_t out_size);

    void learn(const std::vector<double>& in, const std::vector<double>& training);
    std::vector<double> input(const std::vector<double>& in) const;

    void print_weight() const
    {
        for (std::size_t i=1; i <= num_neurons_ * num_neurons_; i++) {
            printf("%2.2g ", weight_[i-1]);
            if (i % num_neurons_ == 0) putchar('\n');
        }
    }

private:
    std::vector<double> transfer(const std::vector<double>& in, std::size_t in_begin,
                                 std::size_t out_begin, std::size_t out_size) const;

    const std::size_t   num_neurons_;
    const std::size_t   in_size_;
    const std::size_t   mid_size_;
    const std::size_t   out_size_;
    std::vector<double> weight_;
};


#endif // KNOWLEDGE_SCIENCE_NEURAL_NETWORK
