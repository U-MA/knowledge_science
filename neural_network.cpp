#include <iostream>
#include <vector>
#include <cmath>

double sigmoid(double x)
{
    return 1.0 / 1.0 + exp(-x);
}

double square_error(double x, double y)
{
    return pow(x-y, 2) / 2.0;
}

class neural_network
{
public:
    neural_network(std::size_t in_layer, std::size_t mid_layer, std::size_t out_layer);

    void learn(std::vector<double> in, std::vector<double> teach);
    std::vector<double> check(std::vector<double> in) const;
    std::vector<double> transfer(std::vector<double> in, std::size_t in_begin,
                                 std::size_t out_begin, std::size_t out_size) const;
    double weight(std::size_t i, std::size_t j) const;

    void print_weight() const
    {
        for (std::size_t i=1; i <= num_neurons_ * num_neurons_; i++) {
            printf("%2.2g ", weight_[i]);
            if (i % num_neurons_ == 0) putchar('\n');
        }
    }

private:
    std::size_t         num_neurons_;
    std::size_t         num_in_;
    std::size_t         num_mid_;
    std::size_t         num_out_;
    std::vector<double> weight_;
};

neural_network::neural_network(std::size_t in_layer, std::size_t mid_layer,
                               std::size_t out_layer)
{
    num_in_  = in_layer;
    num_mid_ = mid_layer;
    num_out_ = out_layer;
    num_neurons_ = in_layer + mid_layer + out_layer;
    srand(2014);
    for (std::size_t i=0; i < num_neurons_ * num_neurons_; i++) {
        weight_.push_back((rand() % 1000) / 1000.0);
    }
}

double neural_network::weight(std::size_t i, std::size_t j) const
{
    return weight_[i * num_neurons_ + j];
}

std::vector<double> neural_network::transfer(std::vector<double> in, std::size_t in_begin,
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

void neural_network::learn(std::vector<double> in, std::vector<double> teach)
{
    // 入力層から中間層へ
    std::vector<double> mid_data(transfer(in, 0, num_in_, num_mid_));

    // 中間層から出力層へ
    std::vector<double> out_data(transfer(mid_data, num_in_, num_in_+num_mid_, num_out_));

    ////////////////
    // 重みの更新 //
    ////////////////

    std::vector<double> old_weight(weight_);

    // 出力層から中間層へ
    std::vector<double> delta(num_out_);
    for (std::size_t k=0; k < num_out_; k++) {
        delta[k] = - (teach[k] - out_data[k]) * out_data[k] * (1 - out_data[k]);
        for (std::size_t j=0; j < num_mid_; j++) {
            weight_[(j+num_in_) * num_neurons_ + (k+num_in_ + num_mid_)] += -  delta[k] * mid_data[j];
        }
    }

    // 中間層から入力層へ
    for (std::size_t j=0; j < num_mid_; j++) {
        double delta_j = 0;
        for (std::size_t i=0; i < num_in_; i++) {
            for (std::size_t k=0; k < num_out_; k++) {
                delta_j += delta[k] * old_weight[(j+num_in_) * num_neurons_ + (k+num_in_ + num_mid_)];
            }
            delta_j = delta_j * mid_data[j] * (1 - mid_data[j]);
            weight_[i * num_neurons_ + (j+num_in_)] += - delta_j * mid_data[i];
        }
    }
}

std::vector<double> neural_network::check(std::vector<double> in) const
{
    // 入力層から中間層へ
    std::vector<double> mid_data(transfer(in, 0, num_in_, num_mid_));

    // 中間層から出力層へ
    return transfer(mid_data, num_in_, num_in_+num_mid_, num_out_);
}

// 入力層2, 中間層2, 出力層1の階層型ニューラルネットワーク
int main()
{
    // XORを想定
    std::vector<std::vector<double>> inputs = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    std::vector<std::vector<double>> teaches = {
        { 0 },
        { 1 },
        { 1 },
        { 0 }
    };

    neural_network nn(2, 2, 1);

    for (int i=0; i < 10000; i++) {
        for (int i=0; i < 4; i++)
            nn.learn(inputs[0], teaches[0]);
        std::vector<double> x(nn.check(inputs[0]));
        for (std::size_t i=0; i < x.size(); i++)
            std::cout << x[i] << std::endl;
    }
}
