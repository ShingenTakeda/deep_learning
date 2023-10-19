#pragma once
#include <vector>
#include <Eigen/Dense>

//Mutithread this!!!!!

//CHECK THIS SHIT OUT: https://github.com/Cr4ckC4t/neural-network-from-scratch

struct Neuron
{
    size_t num_weights;
    std::vector<double> weights;
};

struct Layer
{
    std::vector<Eigen::MatrixXd> backpropagationLayer;
    std::vector<Eigen::MatrixXd> layer;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Neuron> neurons;
};

class NeuralNetework
{
    public:
        NeuralNetework();

        void AddLayer();
        void UpdateLayer();
    private:
        std::vector<Layer> layers;
};
