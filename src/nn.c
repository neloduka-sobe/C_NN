#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <assert.h>

/*
* Return random double between -1 and 1
* Returns: Random double between -1 and 1
*/
double rand_double() {
    return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

/*
* Create a neuron
* Takes: nin - number of inputs, nonlin - true if non-lineary
* Returns: Pointer to a Neuron
*/
Neuron* create_neuron(int nin, bool nonlin) {
    assert(nin > 0);
    Neuron* neuron = (Neuron*)calloc(sizeof(Neuron), 1);
    assert(neuron);
    neuron->w = NULL;
    for (int i = 0; i < nin; i++) {
        Value* v = create_value(rand_double(), NULL);
        neuron->w = add_child(neuron->w, v);
    }
    neuron->b = create_value(rand_double(), NULL);
    neuron->nonlin = nonlin;
    neuron->nin = nin;
    return neuron;
}

/*
* Call a Neuron
* Takes: Pointer to a neuron, pointer to a value to be put into the neuron
* Returns: Value with the evaluated neuron
*/
Value* call_neuron(Neuron* neuron, ValueNode* x) {
    Value* act = neuron->b;
    ValueNode* w_node = neuron->w;
    ValueNode* input_node = x;
    for (int i = 0; i < neuron->nin; i++) {
        act = add(act, mul(w_node->value, input_node->value));
        w_node = w_node->next;
        input_node = input_node->next;
    }
    if (neuron->nonlin) {
        return relu(act);
    } else {
        return act;
    }
}

/*
* Get neuron parameters
* Takes: Pointer to a neuron, number of parameters in a neuron
* Returns: Pointer to parameters of Neuron
*/
ValueNode* neuron_parameters(Neuron* neuron) {
    ValueNode* params = NULL;
    ValueNode* p = neuron->w;
    while (p != NULL) {
        params = add_child(params, p->value);
        p = p->next;
    }
    params = add_child(params, neuron->b);
    return params;
}

/*
* Creates a layer
* Takes: nin - number of inputs, nout - number of outputs, nonlin - if nonlinear
* Returns: Pointer to a layer
*/
Layer* create_layer(int nin, int nout, bool nonlin) {
    Layer* layer = (Layer*)calloc(sizeof(Layer), 1);
    assert(layer);
    layer->neurons = NULL;
    for (int i = 0; i < nout; i++) {
        Neuron* neuron = create_neuron(nin, nonlin);
        NeuronNode* neuron_node = (NeuronNode*)calloc(sizeof(NeuronNode), 1);
        assert(neuron_node);
        neuron_node->value = neuron;
        neuron_node->next = layer->neurons;
        layer->neurons = neuron_node;
    }
    layer->nout = nout;
    return layer;
}

/*
* Call the layer
* Takes: Layer pointer, value for which to evaluate, size of the input
* Returns: Linked list of Values, one per neuron in the layer
*/
ValueNode* call_layer(Layer* layer, ValueNode* x, int x_size) {
    ValueNode* outputs = NULL;
    NeuronNode* neuron_node = layer->neurons;
    while (neuron_node != NULL) {
        Value* neuron_out = call_neuron(neuron_node->value, x);
        outputs = add_child(outputs, neuron_out);
        neuron_node = neuron_node->next;
    }
    return outputs;
}

/*
* Returns layer parameters
* Takes: Pointer to a layer, pointer to number of the parameters
* Returns: Pointer to parameters of a layer
*/
ValueNode* layer_parameters(Layer* layer) {
    ValueNode* params = NULL;
    NeuronNode* p = layer->neurons;
    while (p != NULL) {
        ValueNode* neuron_params = neuron_parameters(p->value);
        ValueNode* node = neuron_params;
        while (node != NULL) {
            params = add_child(params, node->value);
            node = node->next;
        }
        p = p->next;
    }
    return params;
}

/*
* Creates the MLP
* Takes: Number of inputs, array of outputs, number of layers
* Returns: Pointer to MLP object
*/
MLP* create_MLP(int nin, int* nouts, int n_layers) {
    MLP* mlp = (MLP*)calloc(sizeof(MLP), 1);
    assert(mlp);
    mlp->layers = NULL;
    LayerNode* tail = NULL;
    int input_size = nin;
    for (int i = 0; i < n_layers; i++) {
        Layer* layer = create_layer(input_size, nouts[i], i != n_layers - 1);
        LayerNode* layer_node = (LayerNode*)calloc(sizeof(LayerNode), 1);
        assert(layer_node);
        layer_node->value = layer;
        layer_node->next = NULL;
        if (tail == NULL) {
            mlp->layers = layer_node;
        } else {
            tail->next = layer_node;
        }
        tail = layer_node;
        input_size = nouts[i];
    }
    mlp->n_layers = n_layers;
    return mlp;
}

/*
* Calls the MLP
* Takes: Pointer to the MLP, Value for which to evaluate, and size of the input
* Returns: Value for the MLP evaluated for x
*/
Value* call_MLP(MLP* mlp, ValueNode* x, int x_size) {
    ValueNode* input = x;
    LayerNode* layer_node = mlp->layers;
    while (layer_node != NULL) {
        input = call_layer(layer_node->value, input, layer_node->value->nout);
        layer_node = layer_node->next;
    }
    return input->value;
}

/*
* Returns MLP parameters
* Takes: pointer to the mlp, pointer to array of sizes of parameters
* Returns: Values of the parameters of MLP
*/
ValueNode* MLP_parameters(MLP* mlp) {
    ValueNode* params = NULL;
    LayerNode* layer_node = mlp->layers;
    while (layer_node != NULL) {
        ValueNode* layer_params = layer_parameters(layer_node->value);
        ValueNode* node = layer_params;
        while (node != NULL) {
            params = add_child(params, node->value);
            node = node->next;
        }
        layer_node = layer_node->next;
    }
    return params;
}
