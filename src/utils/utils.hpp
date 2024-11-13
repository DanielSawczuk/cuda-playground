#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>

inline void check_cuda_error(const CUresult err, const char *file, int line) {
  if (err != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorString(err, &msg);
    std::cerr << "CUDA error in file '" << file << "' at line " << line << ": "
              << msg << ".\n";
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) check_cuda_error((call), __FILE__, __LINE__)

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

template <typename dtype>
double dot_product(const std::vector<dtype> &vec1,
                   const std::vector<dtype> &vec2) {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Vectors must have the same size!");
  }

  return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
}

template <typename dtype> bool has_nan(const std::vector<dtype> &vec) {
  return std::any_of(vec.begin(), vec.end(),
                     [](dtype val) { return std::isnan(val); });
}

template <typename dtype> bool is_zero_vector(const std::vector<dtype> &vec) {
  return std::all_of(vec.begin(), vec.end(), [](float i) { return i == 0; });
}

template <typename dtype>
bool cosine_similarity(const std::vector<dtype> &calculated,
                       const std::vector<dtype> &gold,
                       double threshold = 0.99999) {
  if (calculated.size() != gold.size()) {
    std::cout << "Provided tensors have different sizes: " << calculated.size()
              << " vs " << gold.size() << std::endl;
    return false;
  }

  double dot_product_value = dot_product(calculated, gold);
  double norm_calculated = dot_product(calculated, calculated);
  double norm_gold = dot_product(gold, gold);

  norm_calculated = std::sqrt(norm_calculated);
  norm_gold = std::sqrt(norm_gold);

  double cosine_similarity = dot_product_value / (norm_calculated * norm_gold);
  double trimmed_cosine_similarity =
      std::min(1.0, std::max(-1.0, cosine_similarity));
  double angle = std::acos(trimmed_cosine_similarity) / M_PI;

  std::cout << "Cosine similarity: " << trimmed_cosine_similarity << " (>"
            << threshold << ")"
            << " (angle: " << angle << " [degrees])" << std::endl;

  if (trimmed_cosine_similarity < threshold) {
    std::cout << "Cosine similarity is below the threshold!" << std::endl;
    return false;
  }
  return true;
}

template <typename dtype>
bool tensors_close(const std::vector<dtype> &calculated,
                   const std::vector<dtype> &gold, double atol = 1e-8,
                   double rtol = 1e-5) {
  if (calculated.size() != gold.size()) {
    std::cout << "Provided tensors have different sizes: " << calculated.size()
              << " vs " << gold.size() << std::endl;
    return false;
  }

  double max_diff = 0.0;
  size_t max_diff_idx = 0;
  double max_rel_diff = 0.0;
  size_t max_rel_diff_idx = 0;

  bool passed = true;
  for (size_t i = 0; i < calculated.size(); ++i) {
    double diff = std::fabs(calculated[i] - gold[i]);
    double rel_diff = diff / std::fabs(gold[i]);
    if (diff > max_diff) {
      max_diff = diff;
      max_diff_idx = i;
    }
    if (rel_diff > max_rel_diff) {
      max_rel_diff = rel_diff;
      max_rel_diff_idx = i;
    }
  }

  std::cout << "Max absolute difference: " << max_diff << " at index "
            << max_diff_idx << " (values: " << calculated[max_diff_idx]
            << " vs " << gold[max_diff_idx] << ")" << std::endl;
  std::cout << "Max relative difference: " << max_rel_diff << " at index "
            << max_rel_diff_idx << " (values: " << calculated[max_rel_diff_idx]
            << " vs " << gold[max_rel_diff_idx] << ")" << std::endl;

  const size_t print_max = 20;
  size_t incorrect_elements = 0;
  for (size_t i = 0; i < calculated.size(); ++i) {
    double diff = std::fabs(calculated[i] - gold[i]);
    double max_allowed_diff = atol + rtol * std::fabs(gold[i]);

    if (diff > max_allowed_diff) {
      passed = false;
      incorrect_elements++;

      if (incorrect_elements <= print_max) {
        std::cout << "Element " << i << " is not close: " << calculated[i]
                  << " vs " << gold[i]
                  << " (difference and max allowed difference: " << diff
                  << " > " << max_allowed_diff << ")" << std::endl;
      }
    }
  }
  if (!passed) {
    std::cout << "Total number of incorrect elements: " << incorrect_elements
              << "/" << calculated.size() << std::endl;
  }
  return passed;
}

template <typename dtype>
bool compare_tensors(const std::vector<dtype> &calculated,
                     const std::vector<dtype> &gold, double atol = 1e-8,
                     double rtol = 1e-5, double cossim_threshold = 0.99999) {
  std::cout << "Comparing tensors..." << std::endl;
  if (calculated.size() != gold.size()) {
    std::cout << "Provided tensors have different sizes: " << calculated.size()
              << " vs " << gold.size() << std::endl;
    return false;
  }

  if (has_nan(calculated)) {
    std::cout << "Calculated values contain NaNs!" << std::endl;
    return false;
  }

  if (has_nan(gold)) {
    std::cout << "Gold values contain NaNs!" << std::endl;
    return false;
  }

  if (is_zero_vector(calculated)) {
    std::cout << "Calculated vector is a zero vector!" << std::endl;
    return false;
  }

  if (is_zero_vector(gold)) {
    std::cout << "Gold vector is a zero vector!" << std::endl;
    return false;
  }

  bool passed = true;
  passed &= cosine_similarity(calculated, gold, cossim_threshold);
  passed &= tensors_close(calculated, gold, atol, rtol);
  return passed;
}

template <typename dtype>
std::vector<dtype> generate_random_vector(size_t size, double min_value = 0,
                                          double max_value = 1) {
  std::vector<dtype> vec(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min_value, max_value);

  for (size_t i = 0; i < size; ++i) {
    vec[i] = static_cast<dtype>(dis(gen));
  }

  return vec;
}
