#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#define min(x, y) (((x) < (y))? (x) : (y))

void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes);

void fill_window_cuda(int *edgeToColumn, int *blockPartition, int *nodePointer,
                      int *edgeList, int blockSize_h, int blockSize_w,
                      int num_nodes);

void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes);


std::tuple<torch::Tensor, int, torch::Tensor>
seg_sort_dequ_fs(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *blocknum, int * vectornum,
              int *row_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, int, float>
preprocess_gpu_fs(torch::Tensor edgeList_tensor, torch::Tensor nodePointer_tensor,
               int num_nodes, int blockSize_h, int blockSize_w,
               torch::Tensor blockPartition_tensor,
               torch::Tensor edgeToColumn_tensor,
               torch::Tensor edgeToRow_tensor) {
  // input tensors.
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto options_gpu_unit8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  auto edgeList = edgeList_tensor.data<int>();
  auto blockPartition = blockPartition_tensor.data<int>();
  auto row_window_offset_tensor =
      torch::zeros({blockPartition_tensor.size(0) + 1}, options_gpu);
  auto row_window_offset = row_window_offset_tensor.data<int>();
  auto edgeToColumn = edgeToColumn_tensor.data<int>();
  //#NNZ
  auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, options_gpu);
  auto blocknum = torch::zeros({1}, options_gpu);
  auto vectornum = torch::zeros({1}, options_gpu);
  auto block_num = blocknum.data<int>();
  auto vector_num = vectornum.data<int>();
  auto edgeToRow = edgeToRow_tensor.data<int>();
  auto nodePointer = nodePointer_tensor.data<int>();
  auto seg_out = seg_out_tensor.data<int>();
  auto start = std::chrono::high_resolution_clock::now();
  fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
  int block_counter = 0;

  //Step1. 求出seg_out，即每个非零元自己所属于的window
  fill_segment_cuda(nodePointer, seg_out, blockSize_h, blockSize_w, num_nodes);
  
  //Step2. 找到每个元素在window中的new column 
  auto tuple_tensor_blockcnt = seg_sort_dequ_fs(
      seg_out, edgeList, nodePointer, edgeToColumn, edgeToRow, blockPartition,
      block_num, vector_num, row_window_offset, blockSize_h, blockSize_w, num_nodes,
      edgeList_tensor.size(0), blockPartition_tensor.size(0));
      
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count()
            << " seconds\n";
  // auto tcblock_offset_tensor = std::get<0>(tuple_tensor_blockcnt);
  // auto tcblock_rowid_tensor = std::get<1>(tuple_tensor_blockcnt);
  // auto tcblocktile_id_tensor = std::get<2>(tuple_tensor_blockcnt);
  auto sparse_AToX_index_tensor = std::get<0>(tuple_tensor_blockcnt);
  block_counter = std::get<1>(tuple_tensor_blockcnt);
  auto tcvalues_tensor = std::get<2>(tuple_tensor_blockcnt);
  printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter,
         block_counter * 8 * 16);
  
  // 1. Window内有多少block
  // 2. 当前block对应的window --- 负载均衡
  // 3. 每个block内的偏移
  // 4. 每个block中非零元个数
  // 5. block列索引
float elapsed_time = static_cast<float>(elapsed_seconds.count());
  return std::make_tuple(row_window_offset_tensor,
                         sparse_AToX_index_tensor, block_counter, elapsed_time*1000);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gpu_fs", &preprocess_gpu_fs, "Preprocess Step on (CUDA)");
}