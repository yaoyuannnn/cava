%module cmodule
%{
#define SWIG_FILE_WITH_INIT
#include "gamut_map.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {
      (float* input, int in_dim1, int in_dim2, int in_dim3)};
%apply(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {
      (float* result, int out_dim1, int out_dim2, int out_dim3)};
%apply(float* IN_ARRAY2, int DIM1, int DIM2) {
      (float* ctrl_pts, int cp_dim1, int cp_dim2),
      (float* weights, int weight_dim1, int weight_dim2),
      (float* coefs, int coef_dim1, int coef_dim2)
};

%rename (gamut_map) gamut_map_full;
%exception gamut_map_full {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
void gamut_map_full(float* input, int in_dim1, int in_dim2, int in_dim3,
                    float* result, int out_dim1, int out_dim2, int out_dim3,
                    float* ctrl_pts, int cp_dim1, int cp_dim2,
                    float* weights, int weight_dim1, int weight_dim2,
                    float* coefs, int coef_dim1, int coef_dim2) {
    gamut_map(input, in_dim1, in_dim2, in_dim3, result, ctrl_pts, weights, coefs, cp_dim1);
}
%}

%include "gamut_map.h"
