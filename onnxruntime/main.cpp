// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//
#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <numeric>
#include <vector>

class OrtInfer {
 public:
  OrtInfer(const char *model_file) : session_(nullptr) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env, model_file, session_options);
  }
  void RunInfer();

 private:
  Ort::Session session_;
};

void OrtInfer::RunInfer() {
  // std::vector<int64_t> input_dims = {1, 3, 32, 620};
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  std::vector<int64_t> input_dims = {1, 3, 32, 630};
  int64_t input_size = std::accumulate(input_dims.begin(), input_dims.end(), 1,
                                       std::multiplies<int64_t>());
  std::vector<float> input_data(input_size, 0);

  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session_.GetInputCount();
  std::vector<const char *> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char *input_name = session_.GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++) {
      if (input_node_dims[j] == -1) {
        input_node_dims[j] = input_dims[j];
      }
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }
  }

  std::vector<const char *> output_node_names = {"logit"};

  // create input tensor object from data values
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_size, input_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors =
      session_.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                   &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
  printf("Done!\n");
}

int main(int argc, char *argv[]) {
  OrtInfer ort_infer("init.onnx");
  ort_infer.RunInfer();
}
