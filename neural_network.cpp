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

// 入力層2, 中間層2, 出力層1の階層型ニューラルネットワーク
int main()
{
    double weight[] = {
        0, 0, 0.5, 0.5,   0,
        0, 0, 0.5, 0.5,   0,
        0, 0,   0,   0, 0.5,
        0, 0,   0,   0, 0.5,
        0, 0,   0,   0,   0
    };

    // XORを想定
    std::vector<std::vector<double>> inputs = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    std::vector<double> teaches = {
        0,
        1,
        1,
        0
    };

    for (int i=0; i < 4; i++) {
        std::vector<double> input_data(inputs[i]);
        double teach_data = teaches[i];

        // 入力層から中間層へ
        std::vector<double> mid_data(2); // 入力層の出力
        for (int i=0; i < 2; i++) {
            double sum = 0;
            for (int j=2; j < 4; j++) {
                sum += input_data[i] * weight[i * 5 + j];
            }
            mid_data[i] = sigmoid(sum);
        }

        // 中間層から出力層へ
        double out_data = -1;
        double sum = 0;
        for (int j=0; j < 2; j++) {
            sum += mid_data[j+2] * weight[(j+2) * 5 + 4];
        }
        out_data = sigmoid(sum);

        // 誤差が十分小さければ終了
        if (square_error(out_data, teach_data) < 0.001)
            std::cout << out_data << std::endl;

        ////////////////
        // 重みの更新 //
        ////////////////

        double old_weight[25] = {};

        // 出力層から中間層へ
        std::vector<double> delta_jk(2);
        double delta_k = (teach_data - out_data) * out_data * (1 - out_data);
        for (int j=0; j < 2; j++) {
            old_weight[(j+2) * 5 + 4] = weight[(j+2) * 5 + 4];
            weight[(j+2) * 5 + 4] +=  delta_k * mid_data[j];
        }

        // 中間層から入力層へ
        std::vector<double> delta_ij(2);
        for (int i=0; i < 2; i++) {
            for (int j=2; j < 4; j++) {
                old_weight[i * 5 + j] = weight[i * 5 + j];
                weight[i * 5 + j] += - delta_k * old_weight[j * 5 + 4] * mid_data[j-2] *
                                       (1 - mid_data[j-2]) * input_data[i];
            }
        }

        printf("=========================\n");
        for (int i=1; i <= 25; i++) {
            printf("%4.3g ", weight[i-1]);
            if (i % 5 == 0) printf("\n");
        }
    }
}
