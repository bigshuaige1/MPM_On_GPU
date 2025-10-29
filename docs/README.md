# MPM on GPU: A Minimized GPU Implementation of Material Point Method Simulator

This repository contains a GPU implementation of the Material Point Method (MPM) simulator. It is designed to be minimal, efficient, and easily portable across different GPU platforms. A version ported to Moore Threads GPUs is available in the musa branch. For algorithmic details, please refer to the original MPM paper: [here](https://yzhu.io/publication/mpmmls2018siggraph/).



## Main functions

### Compatible Platforms

The GPU-based MPM simulator is implemented using CUDA, ensuring compatibility with NVIDIA GPUs. In the musa branch, it is reimplemented using MUSA, the GPU computing language developed by Moore Threads, which serves a similar role to CUDA. It can be easily migrated to other GPU platforms as well.


### Physics Simulation Core

- **Material Point Method (MPM)**: Implements a complete MPM algorithm for simulating the physical behavior of deformable objects
- **SVD Decomposition**: Integrates a 3x3 matrix singular value decomposition algorithm for handling large deformations


### Simulation Pipeline
- **P2G (Particle to Grid)**: Transfers particle information to the grid
- **Grid Update**: Updates grid velocities and forces
- **G2P (Grid to Particle)**: Transfers grid information back to particles
- **Time Integration**: Uses explicit time integration methods


### Input/Output
- **Input**: Supports 3D model files in OBJ format (`two_dragons.obj`)
- **Output**: Generates OBJ files for each frame to the `res/` directory
- **Performance Monitoring**: Outputs detailed performance statistics to `res/timings.json`



## File Structure

```
MPM/
├── mpm.cu           
├── svd3.h             
├── two_dragons.obj     
├── docs
│   └── README.md      
└── res/                
    ├── res_1.obj       
    ├── res_2.obj      
    └── ...
    └── timings.json    
```



## Compilation 
```bash
nvcc -o mpm mpm.cu
```

```bash
./mpm
```



## Output Specifications

### 1. Console Output
- Particle loading information
- Simulation progress and performance statistics
- Execution time per step

### 2. File Output
- **OBJ file**: Particle position data per frame
- **JSON file**: Detailed performance statistics

### 3. Performance Statistics
- Execution time per step
- Timing analysis by phase
- Total runtime