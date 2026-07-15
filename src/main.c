#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int main(void) {
    srand((unsigned int)time(NULL));

    int nouts[2] = {4, 1};
    MLP* mlp = create_MLP(2, nouts, 2);

    Value* x1 = create_value(1.0, NULL);
    Value* x2 = create_value(-1.0, NULL);
    ValueNode* x = add_child(NULL, x1);
    x = add_child(x, x2);

    Value* out = call_MLP(mlp, x, 2);
    printf("Output: %f\n", out->data);

    backward(out);
    printf("Gradient wrt x1: %f\n", x1->grad);
    printf("Gradient wrt x2: %f\n", x2->grad);

    return 0;
}

