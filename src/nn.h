#ifndef NN_H
#define NN_H

#include "engine.h"
#include <stdbool.h>

typedef struct Neuron Neuron;
typedef struct Layer Layer;
typedef struct MLP MLP;
typedef struct NeuronNode NeuronNode;
typedef struct LayerNode LayerNode;

typedef struct Neuron {
    ValueNode* w;
    Value* b;
    bool nonlin;
    int nin;
} Neuron;

typedef struct NeuronNode {
    Neuron* value;
    NeuronNode* next;
} NeuronNode;

typedef struct Layer {
    NeuronNode* neurons;
    int nout;
} Layer;

typedef struct LayerNode {
    Layer* value;
    LayerNode* next;
} LayerNode;

typedef struct MLP {
    LayerNode* layers;
    int n_layers;
} MLP;

double rand_double();


// Neuron functions
Neuron* create_neuron(int nin, bool nonlin);
Value call_neuron(Neuron *neuron, ValueNode *x);
ValueNode* neuron_parameters(Neuron *neuron);

// Layer functions
Layer* create_layer(int nin, int nout, bool nonlin);
Value call_layer(Layer *layer, ValueNode *x, int x_size);
ValueNode* layer_parameters(Layer *layer);

// MLP functions
MLP* create_MLP(int nin, int *nouts, int n_layers);
Value call_MLP(MLP *mlp, ValueNode *x, int x_size);
ValueNode* MLP_parameters(MLP *mlp);
#endif
