#include "load_camera_model.h"

// Get color space transform
float** get_Ts(char* cam_model_path) {
  float **Ts;
  int err = posix_memalign((void **)&Ts, CACHELINE_SIZE, sizeof(float) * 3);
  for (int i = 0; i < 3; i++)
    err |= posix_memalign((void **)&(Ts[i]), CACHELINE_SIZE, sizeof(float) * 3);
  assert(err == 0 && "Failed to allocate memory!");
  char *line;
  char *str;
  float line_data[3];
  size_t len = 0;
  int line_idx = 0;

  // Open file for reading
  char file_name[] = "raw2jpg_transform.txt";
  char file_path[100];
  strcpy(file_path, cam_model_path);
  strcat(file_path, file_name);
  FILE *fp = fopen(file_path, "r");
  if (fp == NULL) {
    printf("Didn't find the camera model file!\n");
    exit(1);
  }

  while (getline(&line, &len, fp) != -1) {
    str = strtok(line, " \n");
    int i = 0;
    while (str != NULL) {
      line_data[i] = atof(str); 
      str = strtok(NULL, " \n");
      i++;
    }

    if (line_idx >= 1 && line_idx <= 3) {
      for (int j = 0; j < 3; j++) {
        Ts[line_idx - 1][j] = line_data[j];
      }
    }
    line_idx = line_idx + 1;
  }
  fclose(fp);
  free(line);
  return Ts;
}

// Get white balance transform
float** get_Tw(char* cam_model_path, int wb_index) {
  float **Tw;
  int err = posix_memalign((void **)&Tw, CACHELINE_SIZE, sizeof(float) * 3);
  for (int i = 0; i < 3; i++)
    err |= posix_memalign((void **)&(Tw[i]), CACHELINE_SIZE, sizeof(float) * 3);
  assert(err == 0 && "Failed to allocate memory!");
  char *line;
  char *str;
  float line_data[3];
  size_t len = 0;
  int line_idx = 0;

  // Calculate base for the white balance transform selected
  // For more details see the camera model readme
  int wb_base  = 8 + 5*(wb_index-1);

  // Open file for reading
  // Open file for reading
  char file_name[] = "raw2jpg_transform.txt";
  char file_path[100];
  strcpy(file_path, cam_model_path);
  strcat(file_path, file_name);
  FILE *fp = fopen(file_path, "r");
  if (fp == NULL) {
    printf("Didn't find the camera model file!\n");
    exit(1);
  }

  // Read a line at a time
  while (getline(&line, &len, fp) != -1) {
    str = strtok(line, " \n");
    int i = 0;
    while (str != NULL) {
      line_data[i] = atof(str); 
      str = strtok(NULL, " \n");
      i++;
    }

    if (line_idx == wb_base) {
      // Convert the white balance vector into a diagaonal matrix
      for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
          if (i==j) { Tw[i][j] = line_data[i]; }
          else      { Tw[i][j] = 0.0;         }
        }
      }
    }
    line_idx = line_idx + 1;
  }
  fclose(fp);
  free(line);
  return Tw;
}


// Get combined transforms for checking
float** get_TsTw(char* cam_model_path, int wb_index) {
  float **TsTw;
  int err = posix_memalign((void **)&TsTw, CACHELINE_SIZE, sizeof(float) * 3);
  for (int i = 0; i < 3; i++)
    err |=
        posix_memalign((void **)&(TsTw[i]), CACHELINE_SIZE, sizeof(float) * 3);
  assert(err == 0 && "Failed to allocate memory!");
  char *line;
  char *str;
  float line_data[3];
  size_t len = 0;
  int line_idx = 0;

  // Calculate base for the white balance transform selected
  // For more details see the camera model readme
  int wb_base  = 5 + 5*(wb_index-1);

  // Open file for reading
  // Open file for reading
  char file_name[] = "raw2jpg_transform.txt";
  char file_path[100];
  strcpy(file_path, cam_model_path);
  strcat(file_path, file_name);
  FILE *fp = fopen(file_path, "r");
  if (fp == NULL) {
    printf("Didn't find the camera model file!\n");
    exit(1);
  }

  // Read a line at a time
  while (getline(&line, &len, fp) != -1) {
    str = strtok(line, " \n");
    int i = 0;
    while (str != NULL) {
      line_data[i] = atof(str); 
      str = strtok(NULL, " \n");
      i++;
    }

    if (line_idx >= wb_base && line_idx <= (wb_base + 2)) {
      for (int j = 0; j < 3; j++) {
        TsTw[line_idx - wb_base][j] = line_data[j];
      }
    }
    line_idx = line_idx + 1;
  }
  fclose(fp);
  free(line);
  return TsTw;
}

//// Get control points
//vector<vector<float>> get_ctrl_pts(char* cam_model_path, int num_cntrl_pts, bool direction) {
//
//  vector<vector<float>> ctrl_pnts;
//  string   line, directionfile;
//  int line_idx = 0;
//
//  // Open file for reading
//  if (direction==1) {
//    // Forward pipeline
//    directionfile = string("raw2jpg_ctrlPoints.txt");
//  } else {
//    // Backward pipeline
//    directionfile = string("jpg2raw_ctrlPoints.txt");
//  }
//  ifstream file(cam_model_path+directionfile);
//
//  // Read a line at a time
//  while(getline(file, line))
//  {
//    vector<float> lineData, temp;
//    stringstream  lineStream(line);
//    float value;
//    // Read one value at a time from the line
//    while(lineStream >> value)
//    {
//      lineData.push_back(value);
//    }
//    if (line_idx>=1 && line_idx<=num_cntrl_pts) {
//      for (int i=0; i<3; i++) {
//        temp.push_back(lineData[i]);
//      }
//      ctrl_pnts.push_back(temp);
//    }
//    line_idx = line_idx + 1;
//  }
//
//  return ctrl_pnts;
//
//}
//
//// Get weights
//vector<vector<float>> get_weights(char* cam_model_path, int num_cntrl_pts, bool direction) {
//
//  vector<vector<float>> weights;
//  string   line, directionfile;
//  int line_idx = 0;
//
//  // Open file for reading
//  if (direction==1) {
//    // Forward pipeline
//    directionfile = string("raw2jpg_coefs.txt");
//  } else {
//    // Backward pipeline
//    directionfile = string("jpg2raw_coefs.txt");
//  }
//  ifstream file(cam_model_path+directionfile);
//
//  // Read a line at a time
//  while(getline(file, line))
//  {
//    vector<float> lineData, temp;
//    stringstream  lineStream(line);
//    float value;
//    // Read one value at a time from the line
//    while(lineStream >> value)
//    {
//     lineData.push_back(value);
//    }
//    if (line_idx>=1 && line_idx<=num_cntrl_pts) {
//      for (int i=0; i<3; i++) {
//        temp.push_back(lineData[i]);
//      }
//      weights.push_back(temp);
//    }
//    line_idx = line_idx + 1;
//  }
//
//  return weights;
//
//}
//
//// Get coeficients
//vector<vector<float>> get_coefs(char* cam_model_path, int num_cntrl_pts, bool direction) {
//
//  vector<vector<float>> coefs;
//  string   line, directionfile;
//  int line_idx = 0;
//
//  // Open file for reading
//  if (direction==1) {
//    // Forward pipeline
//    directionfile = string("raw2jpg_coefs.txt");
//  } else {
//    // Backward pipeline
//    directionfile = string("jpg2raw_coefs.txt");
//  }
//  ifstream file(cam_model_path+directionfile);
//
//  // Read a line at a time
//  while(getline(file, line))
//  {
//    vector<float> lineData, temp;
//    stringstream  lineStream(line);
//    float value;
//   // Read one value at a time from the line
//    while(lineStream >> value)
//    {
//      lineData.push_back(value);
//    }
//    if (line_idx>=(num_cntrl_pts+1) && line_idx<=(num_cntrl_pts+4)) {
//      for (int i=0; i<3; i++) {
//        temp.push_back(lineData[i]);
//      }
//      coefs.push_back(temp);
//    }
//    line_idx = line_idx + 1;
//  }
//
//  return coefs;
//
//}
