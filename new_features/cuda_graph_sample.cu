#include "../common/common.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <vector>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

typedef struct callBackData {
    const char *fn_name;
    double *data;
} callBackData_t;

__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       size_t outputSize) {
    __shared__ double tmp[THREADS_PER_BLOCK];

    cg::thread_block cta = cg::this_thread_block();
    size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    double temp_sum = 0.0;
    for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
        temp_sum += (double)inputVec[i];
    }
    tmp[cta.thread_rank()] = temp_sum;

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    double beta = temp_sum;
    double temp;

    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        if (tile32.thread_rank() < i) {
            temp = tmp[cta.thread_rank() + i];
            beta += temp;
            tmp[cta.thread_rank()] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
        beta = 0.0;
        for (int i = 0; i < cta.size(); i += tile32.size()) {
            beta += tmp[i];
        }
        outputVec[blockIdx.x] = beta;
    }
}

__global__ void reduceFinal(double *inputVec, double *result,
                            size_t inputSize) {
    __shared__ double tmp[THREADS_PER_BLOCK];

    cg::thread_block cta = cg::this_thread_block();
    size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    double temp_sum = 0.0;
    for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
        temp_sum += (double)inputVec[i];
    }
    tmp[cta.thread_rank()] = temp_sum;

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    // do reduction in shared mem
    if ((blockDim.x >= 512) && (cta.thread_rank() < 256)) {
        tmp[cta.thread_rank()] = temp_sum =
            temp_sum + tmp[cta.thread_rank() + 256];
    }

    cg::sync(cta);

    if ((blockDim.x >= 256) && (cta.thread_rank() < 128)) {
        tmp[cta.thread_rank()] = temp_sum =
            temp_sum + tmp[cta.thread_rank() + 128];
    }

    cg::sync(cta);

    if ((blockDim.x >= 128) && (cta.thread_rank() < 64)) {
        tmp[cta.thread_rank()] = temp_sum =
            temp_sum + tmp[cta.thread_rank() + 64];
    }

    cg::sync(cta);

    if (cta.thread_rank() < 32) {
        // Fetch final intermediate sum from 2nd warp
        if (blockDim.x >= 64)
            temp_sum += tmp[cta.thread_rank() + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            temp_sum += tile32.shfl_down(temp_sum, offset);
        }
    }
    // write result for this block to global mem
    if (cta.thread_rank() == 0)
        result[0] = temp_sum;
}

void init_input(float *a, size_t size) {
    for (size_t i = 0; i < size; i++)
        a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void CUDART_CB myHostNodeCallback(void *data) {
    // Check status of GPU after stream operations are done
    callBackData_t *tmp = (callBackData_t *)(data);
    // CHECK(tmp->status);

    double *result = (double *)(tmp->data);
    char *function = (char *)(tmp->fn_name);
    printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
    *result = 0.0; // reset the result
}

void cudaGraphsManual(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                      double *result_d, size_t inputSize, size_t numOfBlocks) {
    // 创建graph的流
    cudaStream_t streamForGraph;
    cudaGraph_t graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t memcpyNode, kernelNode, memsetNode;
    double result_h = 0.0;

    CHECK(cudaStreamCreate(&streamForGraph));

    cudaKernelNodeParams kernelNodeParams = {0};
    cudaMemcpy3DParms memcpyParams = {0};
    cudaMemsetParams memsetParams = {0};

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(
        inputVec_h, sizeof(float) * inputSize, inputSize, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(
        inputVec_d, sizeof(float) * inputSize, inputSize, 1);
    memcpyParams.extent = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;

    memsetParams.dst = (void *)outputVec_d;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(float); // elementSize can be max 4 bytes
    memsetParams.width = numOfBlocks * 2;
    memsetParams.height = 1;

    CHECK(cudaGraphCreate(&graph, 0));
    CHECK(cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
    CHECK(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

    nodeDependencies.push_back(memsetNode);
    nodeDependencies.push_back(memcpyNode);

    void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d,
                           &inputSize, &numOfBlocks};

    kernelNodeParams.func = (void *)reduce;
    kernelNodeParams.gridDim = dim3(numOfBlocks, 1, 1);
    kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = (void **)kernelArgs;
    kernelNodeParams.extra = NULL;

    CHECK(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                 nodeDependencies.size(), &kernelNodeParams));

    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = result_d;
    memsetParams.value = 0;
    memsetParams.elementSize = sizeof(float);
    memsetParams.width = 2;
    memsetParams.height = 1;
    CHECK(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

    nodeDependencies.push_back(memsetNode);

    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams.func = (void *)reduceFinal;
    kernelNodeParams.gridDim = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    void *kernelArgs2[3] = {(void *)&outputVec_d, (void *)&result_d,
                            &numOfBlocks};
    kernelNodeParams.kernelParams = kernelArgs2;
    kernelNodeParams.extra = NULL;

    CHECK(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                 nodeDependencies.size(), &kernelNodeParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    memset(&memcpyParams, 0, sizeof(memcpyParams));

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
    memcpyParams.extent = make_cudaExtent(sizeof(double), 1, 1);
    memcpyParams.kind = cudaMemcpyDeviceToHost;
    CHECK(cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                 nodeDependencies.size(), &memcpyParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(memcpyNode);

    cudaGraphNode_t hostNode;
    cudaHostNodeParams hostParams = {0};
    hostParams.fn = myHostNodeCallback;
    callBackData_t hostFnData;
    hostFnData.data = &result_h;
    hostFnData.fn_name = "cudaGraphsManual";
    hostParams.userData = &hostFnData;

    CHECK(cudaGraphAddHostNode(&hostNode, graph, nodeDependencies.data(),
                               nodeDependencies.size(), &hostParams));

    cudaGraphNode_t *nodes = NULL;
    size_t numNodes = 0;
    CHECK(cudaGraphGetNodes(graph, nodes, &numNodes));
    printf("\nNum of nodes in the graph created manually = %zu\n", numNodes);

    cudaGraphExec_t graphExec;
    CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    // 绘制CUDAGraph
    CHECK(cudaGraphDebugDotPrint(graph, "graph.dot",
                                 cudaGraphDebugDotFlagsVerbose));

    cudaGraph_t clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    CHECK(cudaGraphClone(&clonedGraph, graph));
    CHECK(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        CHECK(cudaGraphLaunch(graphExec, streamForGraph));
    }

    CHECK(cudaStreamSynchronize(streamForGraph));

    printf("Cloned Graph Output.. \n");
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        CHECK(cudaGraphLaunch(clonedGraphExec, streamForGraph));
    }
    CHECK(cudaStreamSynchronize(streamForGraph));

    CHECK(cudaGraphExecDestroy(graphExec));
    CHECK(cudaGraphExecDestroy(clonedGraphExec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaGraphDestroy(clonedGraph));
    CHECK(cudaStreamDestroy(streamForGraph));
}

void cudaGraphsUsingStreamCapture(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
    cudaStream_t stream1, stream2, stream3, streamForGraph;
    cudaEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
    cudaGraph_t graph;
    double result_h = 0.0;

    CHECK(cudaStreamCreate(&stream1));
    CHECK(cudaStreamCreate(&stream2));
    CHECK(cudaStreamCreate(&stream3));
    CHECK(cudaStreamCreate(&streamForGraph));

    CHECK(cudaEventCreate(&forkStreamEvent));
    CHECK(cudaEventCreate(&memsetEvent1));
    CHECK(cudaEventCreate(&memsetEvent2));

    CHECK(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    CHECK(cudaEventRecord(forkStreamEvent, stream1));
    CHECK(cudaStreamWaitEvent(stream2, forkStreamEvent, 0));
    CHECK(cudaStreamWaitEvent(stream3, forkStreamEvent, 0));

    CHECK(cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize,
                          cudaMemcpyDefault, stream1));

    CHECK(
        cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2));

    CHECK(cudaEventRecord(memsetEvent1, stream2));

    CHECK(cudaMemsetAsync(result_d, 0, sizeof(double), stream3));
    CHECK(cudaEventRecord(memsetEvent2, stream3));

    CHECK(cudaStreamWaitEvent(stream1, memsetEvent1, 0));

    reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
        inputVec_d, outputVec_d, inputSize, numOfBlocks);

    CHECK(cudaStreamWaitEvent(stream1, memsetEvent2, 0));

    reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d,
                                                      numOfBlocks);
    CHECK(cudaMemcpyAsync(&result_h, result_d, sizeof(double),
                          cudaMemcpyDefault, stream1));

    callBackData_t hostFnData = {0};
    hostFnData.data = &result_h;
    hostFnData.fn_name = "cudaGraphsUsingStreamCapture";
    cudaHostFn_t fn = myHostNodeCallback;
    CHECK(cudaLaunchHostFunc(stream1, fn, &hostFnData));
    CHECK(cudaStreamEndCapture(stream1, &graph));

    cudaGraphNode_t *nodes = NULL;
    size_t numNodes = 0;
    CHECK(cudaGraphGetNodes(graph, nodes, &numNodes));
    printf(
        "\nNum of nodes in the graph created using stream capture API = %zu\n",
        numNodes);

    cudaGraphExec_t graphExec;
    CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    cudaGraph_t clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    CHECK(cudaGraphClone(&clonedGraph, graph));
    CHECK(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        CHECK(cudaGraphLaunch(graphExec, streamForGraph));
    }

    CHECK(cudaStreamSynchronize(streamForGraph));

    printf("Cloned Graph Output.. \n");
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        CHECK(cudaGraphLaunch(clonedGraphExec, streamForGraph));
    }

    CHECK(cudaStreamSynchronize(streamForGraph));

    CHECK(cudaGraphExecDestroy(graphExec));
    CHECK(cudaGraphExecDestroy(clonedGraphExec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaGraphDestroy(clonedGraph));
    CHECK(cudaStreamDestroy(stream1));
    CHECK(cudaStreamDestroy(stream2));
    CHECK(cudaStreamDestroy(streamForGraph));
}

int main(int argc, char **argv) {
    size_t size = 1 << 24; // number of elements to reduce
    size_t maxBlocks = 512;

    // This will pick the best possible CUDA capable device
    int devID = 0;

    printf("元素个数：%zu elements\n", size);
    printf("threads per block  = %d\n", THREADS_PER_BLOCK);
    printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

    float *inputVec_d = NULL, *inputVec_h = NULL;
    double *outputVec_d = NULL, *result_d;

    // host侧分配
    CHECK(cudaMallocHost(&inputVec_h, sizeof(float) * size));
    CHECK(cudaMalloc(&inputVec_d, sizeof(float) * size));
    CHECK(cudaMalloc(&outputVec_d, sizeof(double) * maxBlocks));
    CHECK(cudaMalloc(&result_d, sizeof(double)));

    // 随机初始化向量
    init_input(inputVec_h, size);

    cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size,
                     maxBlocks);
    cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d,
                                 size, maxBlocks);

    CHECK(cudaFree(inputVec_d));
    CHECK(cudaFree(outputVec_d));
    CHECK(cudaFree(result_d));
    CHECK(cudaFreeHost(inputVec_h));
    return EXIT_SUCCESS;
}
