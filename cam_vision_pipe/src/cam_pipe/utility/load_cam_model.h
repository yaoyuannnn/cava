#ifndef _LOAD_CAM_MODEL_H_
#define _LOAD_CAM_MODEL_H_

// Get color space transform
float *get_Ts(char *cam_model_path);

// Get white balance transform
float *get_Tw(char *cam_model_path, int wb_index);

// Get combined transforms for checking
float *get_TsTw(char *cam_model_path, int wb_index);

// Get control points
float *get_ctrl_pts(char *cam_model_path, int num_cntrl_pts);

// Get weights
float *get_weights(char *cam_model_path, int num_cntrl_pts);

// Get coeficients
float *get_coefs(char *cam_model_path, int num_cntrl_pts);

// Get tone mapping table
float *get_tone_map(char *cam_model_path);

#endif
