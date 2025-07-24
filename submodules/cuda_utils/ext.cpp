#include <torch/extension.h>
#include "cuda_utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("accumulate_gaussian_error", &accumulate_gaussian_error);
    m.def("accumulate_gaussian_confidence", &accumulate_gaussian_confidence);
}