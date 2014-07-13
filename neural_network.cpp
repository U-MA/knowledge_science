#include <iostream>
#include <vector>
#include <cmath>

#include <neural_network.h>

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double square_error(double x, double y)
{
    return pow(x-y, 2) / 2.0;
}

neural_network::neural_network(std::size_t in_size, std::size_t mid_size,
                               std::size_t out_size)
    : num_neurons_(in_size+mid_size+out_size),
      in_size_(in_size), mid_size_(mid_size), out_size_(out_size)
{
    srand(2014);
    for (std::size_t i=0; i < num_neurons_ * num_neurons_; i++) {
        weight_.push_back((rand() % 1000) / 1000.0);
    }
}

std::vector<double> neural_network::transfer(const std::vector<double>& in, std::size_t in_begin,
                                             std::size_t out_begin, std::size_t out_size) const
{
    std::vector<double> out_data(out_size);
    for (std::size_t j=0; j < out_size; j++) {
        double sum = .0;
        for (std::size_t i=0; i < in.size(); i++) {
            sum += in[i] * weight_[(i+in_begin) * num_neurons_ + (j+out_begin)];
        }
        out_data[j] = sigmoid(sum);
    }
    return out_data;
}

void neural_network::learn(const std::vector<double>& in, const std::vector<double>& training)
{
    // input layer to middle layer
    std::vector<double> mid_data = transfer(in, 0, in_size_, mid_size_);

    // middle layer to output layer
    std::vector<double> out_data = transfer(mid_data, in_size_, in_size_+mid_size_, out_size_);


    std::vector<double> old_weight = weight_;

    std::vector<double> delta(out_size_);
    for (std::size_t k=0; k < out_size_; k++) {
        delta[k] = - (training[k] - out_data[k]) * out_data[k] * (1 - out_data[k]);
        for (std::size_t j=0; j < mid_size_; j++) {
            weight_[(j+in_size_) * num_neurons_ + (k+in_size_ + mid_size_)] += -  delta[k] * mid_data[j];
        }
    }

    for (std::size_t j=0; j < mid_size_; j++) {
        double delta_j = 0;
        for (std::size_t k=0; k < out_size_; k++) {
            delta_j += delta[k] * old_weight[(j+in_size_) * num_neurons_ + (k+in_size_ + mid_size_)];
        }
        delta_j = delta_j * mid_data[j] * (1 - mid_data[j]);
        for (std::size_t i=0; i < in_size_; i++) {
            weight_[i * num_neurons_ + (j+in_size_)] += - delta_j * in[i];
        }
    }
}

std::vector<double> neural_network::input(const std::vector<double>& in) const
{
    // input layer to middle layer
    std::vector<double> mid_data = transfer(in, 0, in_size_, mid_size_);

    // middle layer to output layer
    return transfer(mid_data, in_size_, in_size_+mid_size_, out_size_);
}
