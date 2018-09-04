#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "kernels/pipe_stages.h"

// Get color space transform
float** get_Ts(char* cam_model_path);

// Get white balance transform
float** get_Tw(char* cam_model_path, int wb_index);

// Get combined transforms for checking
float** get_TsTw(char* cam_model_path, int wb_index);

//// Get control points
//vector<vector<float>> get_ctrl_pts(char* cam_model_path, int num_cntrl_pts, bool direction);
//
//// Get weights
//vector<vector<float>> get_weights(char* cam_model_path, int num_cntrl_pts, bool direction);
//
//// Get coeficients
//vector<vector<float>> get_coefs(char* cam_model_path, int num_cntrl_pts, bool direction);
