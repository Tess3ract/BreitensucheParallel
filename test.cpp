int *pinned_startingNode;
cudaMallocHost((void**)&pinned_startingNode, sizeInt);
checkHostAllocation(pinned_startingNode, "pinned_startingNode");
memcpy(pinned_startingNode, &startingNode, sizeInt);
cudaMemPrefetchAsync(pinned_startingNode, sizeInt, cudaCpuDeviceId, stream);
int *dev_inQ = queueData.inQ;
cudaMemcpy(dev_inQ, pinned_startingNode, sizeInt, cudaMemcpyHostToDevice);



cudaMemPrefetchAsync(queueData.counterIn, sizeInt, cudaCpuDeviceId, stream);
cudaMemPrefetchAsync(queueData.counterOut, sizeInt, cudaCpuDeviceId, stream);
cudaMemPrefetchAsync(queueData.inQ, sizeInt, cudaCpuDeviceId, stream);
cudaMemPrefetchAsync(queueData.outQ, sizeInt, cudaCpuDeviceId, stream);



cudaMemcpy(queueData.counterIn, host_counterIn, sizeInt, cudaMemcpyHostToDevice);
cudaMemcpy(queueData.counterOut, host_counterOut, sizeInt, cudaMemcpyHostToDevice);
cudaMemcpy(host_counterIn, queueData.counterIn, sizeInt, cudaMemcpyDeviceToHost);
cudaMemcpy(host_counterOut, queueData.counterOut, sizeInt, cudaMemcpyDeviceToHost);



int *host_counterIn;
cudaMallocManaged((void**)&host_counterIn, sizeInt);
checkCudaError(cudaMemset(host_counterIn, 0, sizeInt), "cudaMemset host_counterIn failed");
int *host_counterOut;
cudaMallocManaged((void**)&host_counterOut, sizeInt);
checkCudaError(cudaMemset(host_counterOut, 0, sizeInt), "cudaMemset host_counterOut failed");
