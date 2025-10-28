#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <iomanip>
#include "svd3.h"

const float total_volume = 0.567921f;
const float density = 1000.f;
const float E = 5e3, nu = 0.4f;
const float total_mass = total_volume * density;

const float h_one_over_dx = 128.f;
const float h_dx = 1.f / h_one_over_dx;
const float h_dt = 1e-4;
const float h_mu = E / (2.f * (1.f + nu));
const float h_lambda = E * nu / ((1.f + nu) * (1.f - 2.f * nu));
const float h_gravity = -1.f;
const int h_n_grid = 150;
const int h_num_steps = 60;            
const int h_substeps_per_step = 240;  

const int num_grid_cells = h_n_grid * h_n_grid * h_n_grid;

__device__ __constant__ float one_over_dx;
__device__ __constant__ float dx;
__device__ __constant__ float dt;
__device__ __constant__ float mu;
__device__ __constant__ float lambda;
__device__ __constant__ float gravity;
__device__ __constant__ float m_p;
__device__ __constant__ float V_p;
__device__ __constant__ int n_grid;
__device__ __constant__ int num_p;
__device__ __constant__ int num_g;

int read_from_obj(const std::string& obj_path, std::vector<float>& x_p);
void init_solver(float *&x_p, const std::vector<float>& h_x_p, float *&v_p, float *&F_p, float *&B_p, float *&m_i, float *&v_i, float *&f_i, int h_num_p, int h_num_g);
void substep(float *x_p, float *v_p, float *F_p, float *B_p, float *m_i, float *v_i, float *f_i, int h_num_p, int h_num_g);
void export_to_obj(const std::string& obj_path, float *d_x_p, int num_p);

// #include <filesystem>
int main() {
    // std::cout << std::filesystem::current_path() << std::endl;
    std::cout << "cuda" << std::endl;
    std::vector<float> h_x_p;
    int h_num_p = read_from_obj("./two_dragons.obj", h_x_p);
    // int h_num_p = read_from_obj("assets/two_dragons.obj", h_x_p);

    float *x_p;
    float *v_p;
    float *F_p;
    float *B_p;
    float *m_i;
    float *v_i;
    float *f_i;
    init_solver(x_p, h_x_p, v_p, F_p, B_p, m_i, v_i, f_i, h_num_p, num_grid_cells);

    std::cout << "Start simulate" << std::endl;
    std::cout << "Total steps: " << h_num_steps << ", substeps per step: " << h_substeps_per_step << std::endl;

    cudaEvent_t step_start, step_end;
    cudaEventCreate(&step_start);
    cudaEventCreate(&step_end);

    std::vector<float> step_times_ms;
    step_times_ms.reserve(h_num_steps);
    double total_time_ms_accum = 0.0;

    for (int frame = 0; frame < h_num_steps; frame++) {
        cudaEventRecord(step_start, 0);

        for (int s = 0; s < h_substeps_per_step; s++) {
            substep(x_p, v_p, F_p, B_p, m_i, v_i, f_i, h_num_p, num_grid_cells);
            cudaDeviceSynchronize();
        }

        cudaEventRecord(step_end, 0);
        cudaEventSynchronize(step_end);
        float step_ms = 0.f;
        cudaEventElapsedTime(&step_ms, step_start, step_end);

        export_to_obj("res/res_" + std::to_string(frame + 1) + ".obj", x_p, h_num_p);
        std::printf("step %d/%d time: %.3f ms (substeps: %d)\n", frame + 1, h_num_steps, step_ms, h_substeps_per_step);

        step_times_ms.push_back(step_ms);
        total_time_ms_accum += static_cast<double>(step_ms);
    }

    cudaEventDestroy(step_start);
    cudaEventDestroy(step_end);

    // 输出总耗时到控制台
    std::printf("Total runtime (sum of steps): %.3f ms\n", static_cast<float>(total_time_ms_accum));

    // 将统计信息写入 JSON 文件
    {
        std::ofstream json_file("res/timings.json");
        if (json_file.is_open()) {
            json_file << std::fixed << std::setprecision(3);
            json_file << "{\n";
            json_file << "  \"total_steps\": " << h_num_steps << ",\n";
            json_file << "  \"substeps_per_step\": " << h_substeps_per_step << ",\n";
            json_file << "  \"total_time_ms\": " << total_time_ms_accum << ",\n";
            json_file << "  \"steps\": [\n";
            for (int i = 0; i < (int)step_times_ms.size(); ++i) {
                json_file << "    { \"step\": " << (i + 1) << ", \"time_ms\": " << step_times_ms[i] << " }";
                if (i + 1 < (int)step_times_ms.size()) json_file << ",";
                json_file << "\n";
            }
            json_file << "  ]\n";
            json_file << "}\n";
            json_file.close();
            std::cout << "Timings written to res/timings.json" << std::endl;
        } else {
            std::cerr << "Failed to open res/timings.json for writing" << std::endl;
        }
    }

    return 0;
}

int read_from_obj(const std::string& obj_path, std::vector<float>& x_p) {
    std::ifstream file(obj_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open obj file: " << obj_path << std::endl;
        std::exit(0);
    }
    x_p.reserve(100000 * 3);
    std::string line;
    int particle_idx = 0;
    while (std::getline(file, line)) {
        if (line.rfind("v ", 0) == 0) {
            std::stringstream ss(line.substr(2));
            float x, y, z;
            ss >> x >> y >> z;
            x_p[particle_idx * 3 + 0] = x;
            x_p[particle_idx * 3 + 1] = y;
            x_p[particle_idx * 3 + 2] = z;
            particle_idx++;
        }
    }
    std::cout << "Loaded " << particle_idx << " particles" << std::endl;
    return particle_idx;
}

__global__ void init_particles(float *x_p, float * v_p, float *F_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_p) {
        if (x_p[i * 3 + 2] < 0.5f) {
            v_p[i * 3 + 2] = 0.5f;
        } else {
            v_p[i * 3 + 2] = -0.5f;
        }
        F_p[i * 9 + 0] = 1.f;
        F_p[i * 9 + 4] = 1.f;
        F_p[i * 9 + 8] = 1.f;
    }
}

__device__ __forceinline__ int fetch_grid_index(int i, int j, int k) {
    return i * n_grid * n_grid + j * n_grid + k;
}

__global__ void reset_grids(float *m_i, float *v_i, float *f_i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_g) {
        m_i[idx] = 0.f;
        v_i[idx * 3 + 0] = 0.f;
        v_i[idx * 3 + 1] = 0.f;
        v_i[idx * 3 + 2] = 0.f;
        f_i[idx * 3 + 0] = 0.f;
        f_i[idx * 3 + 1] = 0.f;
        f_i[idx * 3 + 2] = 0.f;
    }
}

__global__ void p2g(float *particles_position, float *particles_velocity, float *particles_deformation_gradient, float *B_p, float *m_i, float * v_i, float *f_i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_p) {
        float x_p[3];
        x_p[0] = particles_position[idx * 3 + 0];
        x_p[1] = particles_position[idx * 3 + 1];
        x_p[2] = particles_position[idx * 3 + 2];

        int smallest_node[3];
        smallest_node[0] = x_p[0] * one_over_dx - 0.5f;
        smallest_node[1] = x_p[1] * one_over_dx - 0.5f;
        smallest_node[2] = x_p[2] * one_over_dx - 0.5f;

        float fx[3];
        fx[0] = x_p[0] * one_over_dx - smallest_node[0];
        fx[1] = x_p[1] * one_over_dx - smallest_node[1];
        fx[2] = x_p[2] * one_over_dx - smallest_node[2];

        float w[3][3], dw[3][3];
        for (int i = 0; i < 3; i++) {
            float d0 = fx[i];
            w[i][0] = 0.5f * (1.5f - d0) * (1.5f - d0);
            w[i][1] = 0.75f - (d0 - 1.f) * (d0 - 1.f);
            w[i][2] = 0.5f * (d0 - 0.5f) * (d0 - 0.5f);

            dw[i][0] = d0 - 1.5f;
            dw[i][1] = 2.f * (1.f - d0);
            dw[i][2] = d0 - 0.5f;
        }
        dw[0][0] *=  one_over_dx;
        dw[0][1] *=  one_over_dx;
        dw[0][2] *=  one_over_dx;
        dw[1][0] *=  one_over_dx;
        dw[1][1] *=  one_over_dx;
        dw[1][2] *=  one_over_dx;
        dw[2][0] *=  one_over_dx;
        dw[2][1] *=  one_over_dx;
        dw[2][2] *=  one_over_dx;

        // svd
        float F_p[9], U[9], sig[3], V[9];
        F_p[0] = particles_deformation_gradient[idx * 9 + 0];
        F_p[1] = particles_deformation_gradient[idx * 9 + 1];
        F_p[2] = particles_deformation_gradient[idx * 9 + 2];
        F_p[3] = particles_deformation_gradient[idx * 9 + 3];
        F_p[4] = particles_deformation_gradient[idx * 9 + 4];
        F_p[5] = particles_deformation_gradient[idx * 9 + 5];
        F_p[6] = particles_deformation_gradient[idx * 9 + 6];
        F_p[7] = particles_deformation_gradient[idx * 9 + 7];
        F_p[8] = particles_deformation_gradient[idx * 9 + 8];
        svd(F_p[0], F_p[3], F_p[6], F_p[1], F_p[4], F_p[7], F_p[2], F_p[5], F_p[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], sig[0], sig[1], sig[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
        float J = sig[0] * sig[1] * sig[2];
        float P_hat[3];
        P_hat[0] = 2.f * mu * (sig[0] - 1.f) + lambda * (J - 1.f) * sig[1] * sig[2];
        P_hat[1] = 2.f * mu * (sig[1] - 1.f) + lambda * (J - 1.f) * sig[0] * sig[2];
        P_hat[2] = 2.f * mu * (sig[2] - 1.f) + lambda * (J - 1.f) * sig[0] * sig[1];
        float P[9];
        P[0] = P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6];
        P[1] = P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6];
        P[2] = P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
        P[3] = P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7];
        P[4] = P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7];
        P[5] = P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
        P[6] = P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8];
        P[7] = P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8];
        P[8] = P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];
        float contribution[9];
        contribution[0] = (P[0] * F_p[0] + P[3] * F_p[3] + P[6] * F_p[6]) * V_p;
        contribution[1] = (P[1] * F_p[0] + P[4] * F_p[3] + P[7] * F_p[6]) * V_p;
        contribution[2] = (P[2] * F_p[0] + P[5] * F_p[3] + P[8] * F_p[6]) * V_p;
        contribution[3] = (P[0] * F_p[1] + P[3] * F_p[4] + P[6] * F_p[7]) * V_p;
        contribution[4] = (P[1] * F_p[1] + P[4] * F_p[4] + P[7] * F_p[7]) * V_p;
        contribution[5] = (P[2] * F_p[1] + P[5] * F_p[4] + P[8] * F_p[7]) * V_p;
        contribution[6] = (P[0] * F_p[2] + P[3] * F_p[5] + P[6] * F_p[8]) * V_p;
        contribution[7] = (P[1] * F_p[2] + P[4] * F_p[5] + P[7] * F_p[8]) * V_p;
        contribution[8] = (P[2] * F_p[2] + P[5] * F_p[5] + P[8] * F_p[8]) * V_p;

        float v_p[3];
        v_p[0] = particles_velocity[idx * 3 + 0];
        v_p[1] = particles_velocity[idx * 3 + 1];
        v_p[2] = particles_velocity[idx * 3 + 2];

        float C_p[9];
        C_p[0] = B_p[idx * 9 + 0] * (4.f * one_over_dx * one_over_dx);
        C_p[1] = B_p[idx * 9 + 1] * (4.f * one_over_dx * one_over_dx);
        C_p[2] = B_p[idx * 9 + 2] * (4.f * one_over_dx * one_over_dx);
        C_p[3] = B_p[idx * 9 + 3] * (4.f * one_over_dx * one_over_dx);
        C_p[4] = B_p[idx * 9 + 4] * (4.f * one_over_dx * one_over_dx);
        C_p[5] = B_p[idx * 9 + 5] * (4.f * one_over_dx * one_over_dx);
        C_p[6] = B_p[idx * 9 + 6] * (4.f * one_over_dx * one_over_dx);
        C_p[7] = B_p[idx * 9 + 7] * (4.f * one_over_dx * one_over_dx);
        C_p[8] = B_p[idx * 9 + 8] * (4.f * one_over_dx * one_over_dx);

        float weight_gradient[3], x_ip[3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int index = fetch_grid_index(smallest_node[0] + i, smallest_node[1] + j, smallest_node[2] + k);

                    float weight = w[0][i] * w[1][j] *w[2][k];
                    weight_gradient[0] = dw[0][i] * w[1][j] * w[2][k];
                    weight_gradient[1] = w[0][i] * dw[1][j] * w[2][k];
                    weight_gradient[2] = w[0][i] * w[1][j] * dw[2][k];

                    x_ip[0] = ((float)i - fx[0]) * dx;
                    x_ip[1] = ((float)j - fx[1]) * dx;
                    x_ip[2] = ((float)k - fx[2]) * dx;

                    atomicAdd(m_i + index, m_p * weight);
                    atomicAdd(v_i + index * 3 + 0, m_p * weight * (v_p[0] + (C_p[0] * x_ip[0] + C_p[3] * x_ip[1] + C_p[6] * x_ip[2])));
                    atomicAdd(v_i + index * 3 + 1, m_p * weight * (v_p[1] + (C_p[1] * x_ip[0] + C_p[4] * x_ip[1] + C_p[7] * x_ip[2])));
                    atomicAdd(v_i + index * 3 + 2, m_p * weight * (v_p[2] + (C_p[2] * x_ip[0] + C_p[5] * x_ip[1] + C_p[8] * x_ip[2])));
                    atomicAdd(f_i + index * 3 + 0, -(contribution[0] * weight_gradient[0] + contribution[3] * weight_gradient[1] + contribution[6] * weight_gradient[2]));
                    atomicAdd(f_i + index * 3 + 1, -(contribution[1] * weight_gradient[0] + contribution[4] * weight_gradient[1] + contribution[7] * weight_gradient[2]));
                    atomicAdd(f_i + index * 3 + 2, -(contribution[2] * weight_gradient[0] + contribution[5] * weight_gradient[1] + contribution[8] * weight_gradient[2]));
                }
            }
        }
    }
}

__global__ void update_grid(float *m_i, float * v_i, float *f_i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_g) {
        if (m_i[idx] > 1e-6) {
            // if (m_i[idx] > m_p * 25) {
            //     printf("%d, %f\n", idx, m_i[idx]);
            // }
            float one_over_mass = 1.f / m_i[idx];

            float new_v_i[3];
            new_v_i[0] = v_i[idx * 3 + 0] * one_over_mass + f_i[idx * 3 + 0] * dt * one_over_mass;
            new_v_i[1] = v_i[idx * 3 + 1] * one_over_mass + gravity * dt + f_i[idx * 3 + 1] * dt * one_over_mass;
            new_v_i[2] = v_i[idx * 3 + 2] * one_over_mass + f_i[idx * 3 + 2] * dt * one_over_mass;

            int i = idx / (n_grid * n_grid);
            int j = (idx / n_grid) % n_grid;
            int k = idx % n_grid;
            if (i < 3 && new_v_i[0] < 0.f) {
                new_v_i[0] = 0.f;
            }
            if (i > n_grid - 3 && new_v_i[0] > 0.f) {
                new_v_i[0] = 0.f;
            }
            if (j < 3 && new_v_i[1] < 0.f) {
                new_v_i[1] = 0.f;
            }
            if (j > n_grid - 3 && new_v_i[1] > 0.f) {
                new_v_i[1] = 0.f;
            }
            if (k < 3 && new_v_i[2] < 0.f) {
                new_v_i[2] = 0.f;
            }
            if (k > n_grid - 3 && new_v_i[2] > 0.f) {
                new_v_i[2] = 0.f;
            }

            v_i[idx * 3 + 0] = new_v_i[0];
            v_i[idx * 3 + 1] = new_v_i[1];
            v_i[idx * 3 + 2] = new_v_i[2];
            // if (idx == 488279) {
            //     printf("%f %f %f\n", new_v_i[0], new_v_i[1], new_v_i[2]);
            // }
        }
    }
}

__global__ void g2p(float *grid_velocity, float *particles_position, float *particles_velocity, float *F_p, float *B_p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_p) {
        float x_p[3];
        x_p[0] = particles_position[idx * 3 + 0];
        x_p[1] = particles_position[idx * 3 + 1];
        x_p[2] = particles_position[idx * 3 + 2];

        int smallest_node[3];
        smallest_node[0] = x_p[0] * one_over_dx - 0.5f;
        smallest_node[1] = x_p[1] * one_over_dx - 0.5f;
        smallest_node[2] = x_p[2] * one_over_dx - 0.5f;

        float fx[3];
        fx[0] = x_p[0] * one_over_dx - smallest_node[0];
        fx[1] = x_p[1] * one_over_dx - smallest_node[1];
        fx[2] = x_p[2] * one_over_dx - smallest_node[2];

        float w[3][3], dw[3][3];
        for (int i = 0; i < 3; i++) {
            float d0 = fx[i];
            w[i][0] = 0.5f * (1.5f - d0) * (1.5f - d0);
            w[i][1] = 0.75f - (d0 - 1.f) * (d0 - 1.f);
            w[i][2] = 0.5f * (d0 - 0.5f) * (d0 - 0.5f);

            dw[i][0] = d0 - 1.5f;
            dw[i][1] = 2.f * (1.f - d0);
            dw[i][2] = d0 - 0.5f;
        }
        dw[0][0] *=  one_over_dx;
        dw[0][1] *=  one_over_dx;
        dw[0][2] *=  one_over_dx;
        dw[1][0] *=  one_over_dx;
        dw[1][1] *=  one_over_dx;
        dw[1][2] *=  one_over_dx;
        dw[2][0] *=  one_over_dx;
        dw[2][1] *=  one_over_dx;
        dw[2][2] *=  one_over_dx;

        float new_v_p[3] = {0.f, 0.f, 0.f}, new_B_p[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, tmp_F_p[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float weight_gradient[3], x_ip[3], v_i[3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int index = fetch_grid_index(smallest_node[0] + i, smallest_node[1] + j, smallest_node[2] + k);

                    float weight = w[0][i] * w[1][j] *w[2][k];
                    weight_gradient[0] = dw[0][i] * w[1][j] * w[2][k];
                    weight_gradient[1] = w[0][i] * dw[1][j] * w[2][k];
                    weight_gradient[2] = w[0][i] * w[1][j] * dw[2][k];

                    x_ip[0] = ((float)i - fx[0]) * dx;
                    x_ip[1] = ((float)j - fx[1]) * dx;
                    x_ip[2] = ((float)k - fx[2]) * dx;

                    v_i[0] = grid_velocity[index * 3 + 0];
                    v_i[1] = grid_velocity[index * 3 + 1];
                    v_i[2] = grid_velocity[index * 3 + 2];

                    new_v_p[0] += v_i[0] * weight;
                    new_v_p[1] += v_i[1] * weight;
                    new_v_p[2] += v_i[2] * weight;
                    // new_B_p[0] += weight * v_i[0] * x_ip[0];
                    // new_B_p[1] += weight * v_i[1] * x_ip[0];
                    // new_B_p[2] += weight * v_i[2] * x_ip[0];
                    // new_B_p[3] += weight * v_i[0] * x_ip[1];
                    // new_B_p[4] += weight * v_i[1] * x_ip[1];
                    // new_B_p[5] += weight * v_i[2] * x_ip[1];
                    // new_B_p[6] += weight * v_i[0] * x_ip[2];
                    // new_B_p[7] += weight * v_i[1] * x_ip[2];
                    // new_B_p[8] += weight * v_i[2] * x_ip[2];
                    // tmp_F_p[0] += v_i[0] * weight_gradient[0];
                    // tmp_F_p[1] += v_i[1] * weight_gradient[0];
                    // tmp_F_p[2] += v_i[2] * weight_gradient[0];
                    // tmp_F_p[3] += v_i[0] * weight_gradient[1];
                    // tmp_F_p[4] += v_i[1] * weight_gradient[1];
                    // tmp_F_p[5] += v_i[2] * weight_gradient[1];
                    // tmp_F_p[6] += v_i[0] * weight_gradient[2];
                    // tmp_F_p[7] += v_i[1] * weight_gradient[2];
                    // tmp_F_p[8] += v_i[2] * weight_gradient[2];

                    new_B_p[0] += weight * v_i[0] * x_ip[0];
                    new_B_p[1] += weight * v_i[0] * x_ip[1];
                    new_B_p[2] += weight * v_i[0] * x_ip[2];
                    new_B_p[3] += weight * v_i[1] * x_ip[0];
                    new_B_p[4] += weight * v_i[1] * x_ip[1];
                    new_B_p[5] += weight * v_i[1] * x_ip[2];
                    new_B_p[6] += weight * v_i[2] * x_ip[0];
                    new_B_p[7] += weight * v_i[2] * x_ip[1];
                    new_B_p[8] += weight * v_i[2] * x_ip[2];
                    tmp_F_p[0] += v_i[0] * weight_gradient[0];
                    tmp_F_p[1] += v_i[0] * weight_gradient[1];
                    tmp_F_p[2] += v_i[0] * weight_gradient[2];
                    tmp_F_p[3] += v_i[1] * weight_gradient[0];
                    tmp_F_p[4] += v_i[1] * weight_gradient[1];
                    tmp_F_p[5] += v_i[1] * weight_gradient[2];
                    tmp_F_p[6] += v_i[2] * weight_gradient[0];
                    tmp_F_p[7] += v_i[2] * weight_gradient[1];
                    tmp_F_p[8] += v_i[2] * weight_gradient[2];

                    // if (idx == 45000) {
                    //     printf("%d: %f, (%f %f %f), (%f %f %f, %f %f %f, %f %f %f)\n", index, weight, weight_gradient[0], weight_gradient[1], weight_gradient[2], tmp_F_p[0], tmp_F_p[1], tmp_F_p[2], tmp_F_p[3], tmp_F_p[4], tmp_F_p[5], tmp_F_p[6], tmp_F_p[7], tmp_F_p[8]);
                    // }
                }
            }
        }

        particles_velocity[idx * 3 + 0] = new_v_p[0];
        particles_velocity[idx * 3 + 1] = new_v_p[1];
        particles_velocity[idx * 3 + 2] = new_v_p[2];

        B_p[idx * 9 + 0] = new_B_p[0];
        B_p[idx * 9 + 1] = new_B_p[1];
        B_p[idx * 9 + 2] = new_B_p[2];
        B_p[idx * 9 + 3] = new_B_p[3];
        B_p[idx * 9 + 4] = new_B_p[4];
        B_p[idx * 9 + 5] = new_B_p[5];
        B_p[idx * 9 + 6] = new_B_p[6];
        B_p[idx * 9 + 7] = new_B_p[7];
        B_p[idx * 9 + 8] = new_B_p[8];
        
        tmp_F_p[0] *= dt;
        tmp_F_p[1] *= dt;
        tmp_F_p[2] *= dt;
        tmp_F_p[3] *= dt;
        tmp_F_p[4] *= dt;
        tmp_F_p[5] *= dt;
        tmp_F_p[6] *= dt;
        tmp_F_p[7] *= dt;
        tmp_F_p[8] *= dt;
        tmp_F_p[0] += 1.f;
        tmp_F_p[4] += 1.f;
        tmp_F_p[8] += 1.f;
        float old_F_p[9];
        old_F_p[0] = F_p[idx * 9 + 0];
        old_F_p[1] = F_p[idx * 9 + 1];
        old_F_p[2] = F_p[idx * 9 + 2];
        old_F_p[3] = F_p[idx * 9 + 3];
        old_F_p[4] = F_p[idx * 9 + 4];
        old_F_p[5] = F_p[idx * 9 + 5];
        old_F_p[6] = F_p[idx * 9 + 6];
        old_F_p[7] = F_p[idx * 9 + 7];
        old_F_p[8] = F_p[idx * 9 + 8];
        F_p[idx * 9 + 0] = tmp_F_p[0] * old_F_p[0] + tmp_F_p[3] * old_F_p[1] + tmp_F_p[6] * old_F_p[2];
        F_p[idx * 9 + 1] = tmp_F_p[1] * old_F_p[0] + tmp_F_p[4] * old_F_p[1] + tmp_F_p[7] * old_F_p[2];
        F_p[idx * 9 + 2] = tmp_F_p[2] * old_F_p[0] + tmp_F_p[5] * old_F_p[1] + tmp_F_p[8] * old_F_p[2];
        F_p[idx * 9 + 3] = tmp_F_p[0] * old_F_p[3] + tmp_F_p[3] * old_F_p[4] + tmp_F_p[6] * old_F_p[5];
        F_p[idx * 9 + 4] = tmp_F_p[1] * old_F_p[3] + tmp_F_p[4] * old_F_p[4] + tmp_F_p[7] * old_F_p[5];
        F_p[idx * 9 + 5] = tmp_F_p[2] * old_F_p[3] + tmp_F_p[5] * old_F_p[4] + tmp_F_p[8] * old_F_p[5];
        F_p[idx * 9 + 6] = tmp_F_p[0] * old_F_p[6] + tmp_F_p[3] * old_F_p[7] + tmp_F_p[6] * old_F_p[8];
        F_p[idx * 9 + 7] = tmp_F_p[1] * old_F_p[6] + tmp_F_p[4] * old_F_p[7] + tmp_F_p[7] * old_F_p[8];
        F_p[idx * 9 + 8] = tmp_F_p[2] * old_F_p[6] + tmp_F_p[5] * old_F_p[7] + tmp_F_p[8] * old_F_p[8];
        // if (idx == 45000) {
        //     printf("(%f %f %f, %f %f %f, %f %f %f)\n", F_p[0], F_p[1], F_p[2], F_p[3], F_p[4], F_p[5], F_p[6], F_p[7], F_p[8]);
        // }

        particles_position[idx * 3 + 0] += new_v_p[0] * dt;
        particles_position[idx * 3 + 1] += new_v_p[1] * dt;
        particles_position[idx * 3 + 2] += new_v_p[2] * dt;
    }
}

void init_solver(float *&x_p, const std::vector<float>& h_x_p, float *&v_p, float *&F_p, float *&B_p, float *&m_i, float *&v_i, float *&f_i, int h_num_p, int h_num_g) {
    float h_m_p = total_mass / h_num_p;
    float h_V_p = total_volume / h_num_p;
    cudaMemcpyToSymbol(one_over_dx, &h_one_over_dx, 4);
    cudaMemcpyToSymbol(dx, &h_dx, 4);
    cudaMemcpyToSymbol(dt, &h_dt, 4);
    cudaMemcpyToSymbol(mu, &h_mu, 4);
    cudaMemcpyToSymbol(lambda, &h_lambda, 4);
    cudaMemcpyToSymbol(gravity, &h_gravity, 4);
    cudaMemcpyToSymbol(m_p, &h_m_p, 4);
    cudaMemcpyToSymbol(V_p, &h_V_p, 4);
    cudaMemcpyToSymbol(n_grid, &h_n_grid, 4);
    cudaMemcpyToSymbol(num_p, &h_num_p, 4);
    cudaMemcpyToSymbol(num_g, &h_num_g, 4);

    cudaMalloc(&x_p, h_num_p * 3 * 4);
    cudaMemcpy(x_p, h_x_p.data(), h_num_p * 3 * 4, cudaMemcpyHostToDevice);

    cudaMalloc(&v_p, h_num_p * 3 * 4);
    cudaMalloc(&F_p, h_num_p * 9 * 4);
    init_particles<<<(h_num_p + 255) / 256, 256>>>(x_p,  v_p, F_p);
    cudaMalloc(&B_p, h_num_p * 9 * 4);

    cudaMalloc(&m_i, h_num_g * 4);
    cudaMalloc(&v_i, h_num_g * 3 * 4);
    cudaMalloc(&f_i, h_num_g * 3 * 4);
}

void substep(float *x_p, float *v_p, float *F_p, float *B_p, float *m_i, float *v_i, float *f_i, int h_num_p, int h_num_g) {
    cudaEvent_t e_start, e_after_reset, e_after_p2g, e_after_update, e_after_g2p;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_after_reset);
    cudaEventCreate(&e_after_p2g);
    cudaEventCreate(&e_after_update);
    cudaEventCreate(&e_after_g2p);

    cudaEventRecord(e_start, 0);

    reset_grids<<<(h_num_g + 511) / 512, 512>>>(m_i, v_i, f_i);
    cudaEventRecord(e_after_reset, 0);

    p2g<<<(h_num_p + 255) / 256, 256>>>(x_p, v_p, F_p, B_p, m_i, v_i, f_i);
    cudaEventRecord(e_after_p2g, 0);

    update_grid<<<(h_num_g + 511) / 512, 512>>>(m_i, v_i, f_i);
    cudaEventRecord(e_after_update, 0);

    g2p<<<(h_num_p + 255) / 256, 256>>>(v_i, x_p, v_p, F_p, B_p);
    cudaEventRecord(e_after_g2p, 0);

    cudaEventSynchronize(e_after_g2p);

    float t_reset_ms = 0.f, t_p2g_ms = 0.f, t_update_ms = 0.f, t_g2p_ms = 0.f, t_total_ms = 0.f;
    cudaEventElapsedTime(&t_reset_ms, e_start, e_after_reset);
    cudaEventElapsedTime(&t_p2g_ms, e_after_reset, e_after_p2g);
    cudaEventElapsedTime(&t_update_ms, e_after_p2g, e_after_update);
    cudaEventElapsedTime(&t_g2p_ms, e_after_update, e_after_g2p);
    cudaEventElapsedTime(&t_total_ms, e_start, e_after_g2p);

    std::printf("substep ms: reset=%.3f, p2g=%.3f, update=%.3f, g2p=%.3f, total=%.3f\n",
                t_reset_ms, t_p2g_ms, t_update_ms, t_g2p_ms, t_total_ms);

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_after_reset);
    cudaEventDestroy(e_after_p2g);
    cudaEventDestroy(e_after_update);
    cudaEventDestroy(e_after_g2p);
}

void export_to_obj(const std::string& obj_path, float *d_x_p, int num_p) {
    float *x_p = (float*)malloc(num_p * 3 * 4);
    cudaMemcpy(x_p, d_x_p, num_p * 3 * 4, cudaMemcpyDeviceToHost);

    std::ofstream file(obj_path);
    for (int i = 0; i < num_p; i++) {
        file << "v " << x_p[i * 3 + 0] << " " << x_p[i * 3 + 1] << " " << x_p[i * 3 + 2] << std::endl;
    }
    file.close();
}