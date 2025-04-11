# SPP Assignment 2

## Report

### Preamble - System Specifications and Dataset used

#### System Specifications

- CPU: Intel(R) Core(TM) i7-13700H CPU @ 2.40GHz
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU - but had many issues with CUDA. Hence, used CPU for the assignment.
- RAM: 16 GB
- OS: WSL (Windows Subsystem for Linux) with Ubuntu 22.04

#### Dataset

The dataset used is a small subset of the COCO dataset called coco128 containing 128 images.

### Task 1: Inference obtained from the YOLOv5 Models

Non pre-trained inference.

- YOLOv5n: time to run: 8s
- YOLOv5s: time to run: 14.1s
- YOLOv5m: time to run: 28.4s
- YOLOv5l: time to run: 52.9s
- YOLOv5x: time to run: 1m 28.3s

Pre-trained models.

- YOLOv5n: time to run: 5.8s
- YOLOv5s: time to run: 10s
- YOLOv5m: time to run: 24.2s
- YOLOv5l: time to run: 42.2s
- YOLOv5x: time to run: 1m 13s

Since the pre-trained models are already trained on the COCO dataset, they are expected to perform better than the non pre-trained models. The inference time is also lower for the pre-trained models. Henceforth we will be using the pretrained models for the remaining tasks.

#### Latency and Throughput Comparison Table for Pre-trained Model

| Model    | Average Latency | Average Throughput | Average Inference Time |
|----------|-----------------|--------------------| --------|
| YOLOv5n  | 54.27 ms        | 20.58 FPS          | 45.65 ms        |
| YOLOv5s  | 87.79 ms        | 12.05 FPS          | 83.09 ms        |
| YOLOv5m  | 189.66 ms       | 5.60 FPS          | 185.05 ms       |
| YOLOv5l  | 341.42 ms       | 3.05 FPS          | 336.72 ms       |
| YOLOv5x  | 580.42 ms       | 1.80 FPS          | 573.85 ms       |

### Task 2: Model information, GFLOPS and roofline analysis

#### Model information and GFLOPS

| Model    | Total Parameters | Model Size | GFLOPS| GFLOPS/sec |
|----------|------------------|------------|-----|--------|
| YOLOv5n  | 1,867,405        | 124.14 MB  | 2.25 |50   |
| YOLOv5s  | 7,225,885        | 240.19 MB  | 8.24 |99.1   |
| YOLOv5m  | 21,172,173       | 469.24 MB  | 24.48 |132.2  |
| YOLOv5l  | 46,533,693       | 796.37 MB  | 54.57 |162.1  |
| YOLOv5x  | 86,705,005       | 1235.18 MB | 102.83 |179.2 |

Peak GFLOPS for my machine is $2.5 * 20 * 1.25 * 8 = 500$ GFLOPS. (Assuming 1.25 CPI, 20 threads and AVX2 vectorization).

Memory Bandwidth of my system - $18$ GB/s (calculated in the last assignment)

And,
$$
\text{Utilization} = \frac{\text{Actual GFLOPS}}{\text{Peak GFLOPS}}
$$

Where Actual GFLOPS is the GFLOPS obtained from the model and Peak GFLOPS is the peak GFLOPS of the machine.

Hence, comparing the Utilization for each model, we obtain the following table:

| Model    | Actual GFLOPS/sec | Utilization |
|----------|---------------|-------------|
| YOLOv5n  | 50           | 0.1       |
| YOLOv5s  | 99.1          | 0.198      |
| YOLOv5m  | 132.2        | 0.264       |
| YOLOv5l  | 162.1        | 0.324       |
| YOLOv5x  | 179.2        | 0.358      |

#### Roofline analysis

| Model    | Operational intensity | OI * Memory Bandwidth | Bound Type |
|----------|---------------|-------------|---------|
| YOLOv5n  | 17.2          | 314.8       |  Memory Bound       |
| YOLOv5s  | 32.7          | 598.4      |    Compute Bound      |
| YOLOv5m  | 49.8          | 911.3       |    Compute  Bound    |
| YOLOv5l  | 65.3          | 1195       |   Compute  Bound        |
| YOLOv5x  | 79.4          | 1453      |    Compute  Bound        |

The analysis here is a little decieveing, since we are assuming that the whole model fits into our cache, which is just simply impossible. Hence we observe such a discrepancy between calculated GFLOPS/sec and actual GFLOPS/sec (since we are considering no conflict misses in the cache).

#### Bonus task: Extending above analysis per layer

### Task 3: Code Profiling & Hotspot Identification

Used tools: `PyTorch Profiler` and `line_profile`

`PyTorch Profiler` Results:

`YOLOv5n`

**CPU Profiler**
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                     aten::conv2d         0.70%      32.182ms        67.68%        3.123s     406.596us       4.93 Gb           0 b          7680
                aten::convolution         0.61%      28.316ms        66.98%        3.090s     402.406us       4.93 Gb           0 b          7680
               aten::_convolution         1.24%      57.425ms        66.37%        3.062s     398.719us       4.93 Gb           0 b          7680
         aten::mkldnn_convolution        63.57%        2.933s        65.07%        3.002s     391.534us       4.93 Gb     490.00 Kb          7668
                      aten::silu_        11.50%     530.552ms        11.50%     530.552ms      72.718us           0 b           0 b          7296
                        aten::cat         6.44%     297.131ms         7.10%     327.737ms     124.567us       3.00 Gb       3.00 Gb          2631
                 aten::max_pool2d         0.04%       2.060ms         3.62%     167.044ms     435.011us      55.80 Mb    -110.59 Mb           384
    aten::max_pool2d_with_indices         3.58%     164.984ms         3.58%     164.984ms     429.647us     166.64 Mb     166.64 Mb           384
                      aten::copy_         1.97%      90.947ms         1.97%      90.947ms      13.915us           0 b           0 b          6536
                    aten::sigmoid         1.44%      66.268ms         1.44%      66.268ms     172.573us     774.62 Mb     774.62 Mb           384
                      aten::clone         0.12%       5.383ms         1.42%      65.547ms      47.122us     874.23 Mb           0 b          1391
                 aten::contiguous         0.05%       2.282ms         1.42%      65.341ms      57.417us     874.12 Mb           0 b          1138
                        aten::mul         1.15%      53.172ms         1.32%      60.714ms      28.802us      72.96 Mb      72.95 Mb          2108
                         aten::to         0.14%       6.485ms         1.16%      53.310ms      10.851us     444.42 Mb           0 b          4913
                      aten::empty         1.07%      49.486ms         1.07%      49.486ms       2.569us       5.79 Gb       5.79 Gb         19262
                   aten::_to_copy         0.31%      14.429ms         1.01%      46.825ms      14.412us     444.42 Mb         212 b          3249
         aten::upsample_nearest2d         0.37%      17.203ms         0.92%      42.404ms     165.639us     222.19 Mb     122.58 Mb           256
                        aten::div         0.57%      26.270ms         0.75%      34.825ms      30.548us     444.51 Mb     444.50 Mb          1140
                     aten::narrow         0.27%      12.448ms         0.68%      31.324ms       7.868us           0 b           0 b          3981
                        aten::add         0.64%      29.650ms         0.64%      29.650ms      15.548us     444.23 Mb     444.23 Mb          1907
                    aten::type_as         0.01%     412.070us         0.57%      26.153ms     204.320us     444.38 Mb           0 b           128
                      aten::slice         0.42%      19.467ms         0.55%      25.467ms       3.798us           0 b           0 b          6705
                     aten::select         0.45%      20.848ms         0.53%      24.512ms       3.706us           0 b           0 b          6615
                      aten::index         0.24%      10.904ms         0.46%      21.175ms      28.196us       2.46 Mb       2.36 Mb           751
                aten::as_strided_         0.46%      21.113ms         0.46%      21.113ms       2.665us           0 b           0 b          7921
                         aten::gt         0.33%      15.061ms         0.39%      17.831ms      70.477us       2.28 Mb       2.28 Mb           253
                     aten::arange         0.19%       8.960ms         0.38%      17.430ms       9.662us     386.02 Kb      28.21 Kb          1804
                 torchvision::nms         0.09%       4.033ms         0.35%      16.161ms     133.562us      44.14 Kb    -226.22 Kb           121
                 aten::as_strided         0.30%      13.677ms         0.30%      13.677ms       0.751us           0 b           0 b         18206
                      aten::stack         0.05%       2.101ms         0.27%      12.624ms      38.255us       5.22 Mb           0 b           330
                        aten::sub         0.15%       6.753ms         0.24%      11.194ms      10.384us      15.77 Mb      15.77 Mb          1078
                       aten::sort         0.14%       6.466ms         0.22%       9.973ms      41.210us     132.42 Kb      44.14 Kb           242
                 aten::empty_like         0.09%       4.028ms         0.15%       7.102ms       5.623us     874.22 Mb         780 b          1263
                    aten::nonzero         0.13%       6.121ms         0.15%       6.937ms      27.420us      96.59 Kb      96.59 Kb           253
                    aten::argsort         0.02%     830.702us         0.15%       6.751ms      55.795us      45.62 Kb     -19.11 Kb           121
              aten::empty_strided         0.14%       6.273ms         0.14%       6.273ms       1.791us     444.53 Mb     444.53 Mb          3502
                   aten::meshgrid         0.07%       3.447ms         0.14%       6.251ms      18.944us           0 b           0 b           330
                    aten::resize_         0.13%       6.150ms         0.13%       6.150ms       0.717us    1004.80 Kb    1004.80 Kb          8582
                        aten::pow         0.11%       5.215ms         0.12%       5.574ms      14.515us      18.23 Mb      18.23 Mb           384
                     aten::expand         0.07%       3.075ms         0.09%       4.078ms       3.089us           0 b           0 b          1320
                        aten::max         0.09%       3.939ms         0.09%       3.939ms      31.513us      78.68 Kb      78.68 Kb           125
           aten::split_with_sizes         0.06%       2.583ms         0.07%       3.457ms       9.002us           0 b           0 b           384
                 aten::index_put_         0.02%     708.566us         0.07%       3.394ms      13.258us         -16 b         -16 b           256
                       aten::view         0.07%       3.263ms         0.07%       3.263ms       1.099us           0 b           0 b          2970
                      aten::zeros         0.04%       1.652ms         0.07%       3.196ms       8.638us      49.66 Kb       1.33 Kb           370
                       aten::sub_         0.02%       1.117ms         0.06%       2.942ms      11.491us         136 b        -752 b           256
           aten::_index_put_impl_         0.04%       2.026ms         0.06%       2.685ms      10.490us           0 b           0 b           256
                    aten::reshape         0.04%       1.847ms         0.06%       2.538ms       2.491us           0 b           0 b          1019
                       aten::ones         0.03%       1.399ms         0.05%       2.423ms      10.013us      66.34 Kb         240 b           242
                aten::thnn_conv2d         0.00%      52.821us         0.05%       2.401ms     200.045us     920.00 Kb           0 b            12
       aten::_slow_conv2d_forward         0.04%       1.984ms         0.05%       2.348ms     195.643us     920.00 Kb      -3.96 Mb            12
                    aten::permute         0.03%       1.294ms         0.04%       1.996ms       5.197us           0 b           0 b           384
                  aten::unsqueeze         0.03%       1.333ms         0.04%       1.725ms       2.614us           0 b           0 b           660
                     aten::clamp_         0.03%       1.581ms         0.04%       1.657ms       3.237us           0 b           0 b           512
                       aten::div_         0.01%     625.094us         0.03%       1.394ms      10.894us          52 b        -408 b           128
                      aten::fill_         0.03%       1.225ms         0.03%       1.225ms       3.312us           0 b           0 b           370
                     aten::unbind         0.01%     333.823us         0.02%       1.005ms       7.849us           0 b           0 b           128
                      aten::zero_         0.01%     495.466us         0.02%     995.011us       2.689us           0 b           0 b           370
                       aten::mul_         0.02%     780.596us         0.02%     780.596us       6.245us           0 b           0 b           125
                 aten::lift_fresh         0.01%     331.039us         0.01%     331.039us       0.431us           0 b           0 b           768
                aten::result_type         0.01%     241.124us         0.01%     241.124us       0.628us           0 b           0 b           384
                      aten::alias         0.00%     181.093us         0.00%     181.093us       1.415us           0 b           0 b           128
                    aten::detach_         0.00%      68.018us         0.00%      68.018us       0.531us           0 b           0 b           128
          aten::_nnpack_available         0.00%      52.944us         0.00%      52.944us       4.412us           0 b           0 b            12
                     aten::detach         0.00%      14.525us         0.00%      14.525us       2.421us           0 b           0 b             6
                         [memory]         0.00%       0.000us         0.00%       0.000us       0.000us     -11.13 Gb     -11.13 Gb         26389
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 4.614s

`YOLOv5s`

**CPU Profiler**
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                     aten::conv2d         0.48%      50.814ms        79.77%        8.389s       1.092ms       9.11 Gb           0 b          7680
                aten::convolution         0.42%      43.709ms        79.28%        8.338s       1.086ms       9.11 Gb           0 b          7680
               aten::_convolution         0.92%      96.941ms        78.87%        8.294s       1.080ms       9.11 Gb           0 b          7680
         aten::mkldnn_convolution        76.73%        8.069s        77.95%        8.197s       1.067ms       9.11 Gb           0 b          7680
                      aten::silu_         7.86%     826.585ms         7.86%     826.585ms     113.293us           0 b           0 b          7296
                        aten::cat         2.99%     314.697ms         3.40%     357.847ms     135.960us       4.48 Gb       4.48 Gb          2632
                 aten::max_pool2d         0.03%       3.336ms         3.11%     327.508ms     852.884us     111.09 Mb    -222.19 Mb           384
    aten::max_pool2d_with_indices         3.08%     324.172ms         3.08%     324.172ms     844.198us     333.28 Mb     333.28 Mb           384
                      aten::copy_         1.29%     135.407ms         1.29%     135.407ms      20.575us           0 b           0 b          6581
                      aten::empty         0.92%      97.264ms         0.92%      97.264ms       5.036us      10.06 Gb      10.06 Gb         19314
                 aten::contiguous         0.03%       2.946ms         0.85%      89.209ms      77.037us     973.58 Mb           0 b          1158
                      aten::clone         0.06%       6.599ms         0.85%      88.874ms      62.942us     973.74 Mb           0 b          1412
                        aten::mul         0.69%      72.751ms         0.79%      82.617ms      39.007us      72.98 Mb      72.97 Mb          2118
                         aten::to         0.07%       7.020ms         0.75%      78.410ms      15.924us     444.43 Mb           0 b          4924
                   aten::_to_copy         0.16%      16.820ms         0.68%      71.390ms      21.899us     444.43 Mb         168 b          3260
                    aten::sigmoid         0.62%      65.672ms         0.62%      65.672ms     171.020us     774.62 Mb     774.62 Mb           384
         aten::upsample_nearest2d         0.23%      23.699ms         0.60%      63.345ms     247.441us     444.38 Mb     245.35 Mb           256
                        aten::div         0.44%      46.166ms         0.54%      56.863ms      49.705us     444.56 Mb     444.55 Mb          1144
                        aten::add         0.47%      49.619ms         0.47%      49.619ms      25.924us     870.15 Mb     870.15 Mb          1914
                    aten::type_as         0.01%     560.952us         0.43%      44.948ms     351.156us     444.38 Mb           0 b           128
                     aten::narrow         0.17%      18.201ms         0.42%      43.989ms      11.036us           0 b           0 b          3986
                      aten::slice         0.24%      25.597ms         0.32%      33.239ms       4.916us           0 b           0 b          6762
                aten::as_strided_         0.31%      32.725ms         0.31%      32.725ms       4.125us           0 b           0 b          7934
                     aten::select         0.25%      26.621ms         0.30%      31.124ms       4.675us           0 b           0 b          6658
                     aten::arange         0.11%      11.629ms         0.23%      24.008ms      13.162us     477.08 Kb      31.42 Kb          1824
                      aten::index         0.12%      12.142ms         0.22%      23.640ms      31.024us       3.47 Mb       3.34 Mb           762
                      aten::stack         0.03%       3.399ms         0.18%      19.237ms      58.293us       5.22 Mb           0 b           330
                         aten::gt         0.15%      15.780ms         0.18%      18.730ms      73.739us       2.29 Mb       2.29 Mb           254
                 torchvision::nms         0.05%       4.932ms         0.18%      18.598ms     147.599us      66.91 Kb    -342.89 Kb           126
                 aten::as_strided         0.16%      17.202ms         0.16%      17.202ms       0.939us           0 b           0 b         18327
                        aten::sub         0.10%       9.995ms         0.15%      15.752ms      14.451us      15.81 Mb      15.81 Mb          1090
                 aten::empty_like         0.05%       5.673ms         0.10%      10.887ms       8.479us     973.72 Mb         424 b          1284
                        aten::pow         0.10%      10.195ms         0.10%      10.586ms      27.567us      18.23 Mb      18.23 Mb           384
                    aten::resize_         0.09%       9.516ms         0.09%       9.516ms       1.108us     207.12 Kb     207.12 Kb          8592
                       aten::sort         0.05%       4.845ms         0.08%       8.916ms      35.383us     200.72 Kb      66.91 Kb           252
              aten::empty_strided         0.08%       8.282ms         0.08%       8.282ms       2.357us     444.59 Mb     444.59 Mb          3514
                    aten::nonzero         0.06%       6.775ms         0.07%       7.654ms      30.133us     140.45 Kb     140.45 Kb           254
                   aten::meshgrid         0.04%       4.051ms         0.07%       7.536ms      22.835us           0 b           0 b           330
                    aten::argsort         0.00%     434.016us         0.05%       5.773ms      45.819us      67.98 Kb     -31.30 Kb           126
                     aten::expand         0.04%       4.022ms         0.05%       5.297ms       4.013us           0 b           0 b          1320
           aten::split_with_sizes         0.03%       3.460ms         0.04%       4.533ms      11.804us           0 b           0 b           384
                 aten::index_put_         0.01%     867.710us         0.04%       3.841ms      15.003us           0 b           0 b           256
                       aten::view         0.04%       3.746ms         0.04%       3.746ms       1.273us           0 b           0 b          2942
                      aten::zeros         0.02%       1.804ms         0.03%       3.626ms       9.541us      75.27 Kb         744 b           380
                       aten::sub_         0.01%       1.309ms         0.03%       3.496ms      13.657us          76 b        -872 b           256
                        aten::max         0.03%       3.265ms         0.03%       3.265ms      25.913us     110.31 Kb     110.31 Kb           126
                       aten::ones         0.02%       1.894ms         0.03%       3.003ms      12.411us      66.34 Kb         448 b           242
           aten::_index_put_impl_         0.02%       2.216ms         0.03%       2.973ms      11.614us           0 b           0 b           256
                    aten::reshape         0.02%       2.094ms         0.03%       2.831ms       2.781us           0 b           0 b          1018
                    aten::permute         0.02%       1.651ms         0.03%       2.640ms       6.874us           0 b           0 b           384
                  aten::unsqueeze         0.02%       1.612ms         0.02%       2.110ms       3.198us           0 b           0 b           660
                     aten::clamp_         0.02%       1.743ms         0.02%       1.825ms       3.565us           0 b           0 b           512
                       aten::div_         0.01%     700.852us         0.01%       1.548ms      12.093us          44 b        -424 b           128
                      aten::fill_         0.01%       1.343ms         0.01%       1.343ms       3.630us           0 b           0 b           370
                      aten::zero_         0.01%     542.759us         0.01%       1.145ms       3.013us           0 b           0 b           380
                     aten::unbind         0.00%     386.086us         0.01%       1.096ms       8.559us           0 b           0 b           128
                       aten::mul_         0.01%     918.188us         0.01%     918.188us       7.287us           0 b           0 b           126
                 aten::lift_fresh         0.00%     400.428us         0.00%     400.428us       0.521us           0 b           0 b           768
                aten::result_type         0.00%     279.989us         0.00%     279.989us       0.729us           0 b           0 b           384
                      aten::alias         0.00%     195.637us         0.00%     195.637us       1.528us           0 b           0 b           128
                    aten::detach_         0.00%      75.253us         0.00%      75.253us       0.588us           0 b           0 b           128
                         [memory]         0.00%       0.000us         0.00%       0.000us       0.000us     -17.48 Gb     -17.48 Gb         26451
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 10.517s

`YOLOv5m`

**CPU Profiler**

---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                     aten::conv2d         0.37%      84.098ms        86.94%       19.851s       1.891ms      16.76 Gb           0 b         10496
                aten::convolution         0.28%      63.349ms        86.58%       19.767s       1.883ms      16.76 Gb           0 b         10496
               aten::_convolution         0.51%     117.308ms        86.30%       19.704s       1.877ms      16.76 Gb           0 b         10496
         aten::mkldnn_convolution        84.65%       19.327s        85.78%       19.586s       1.866ms      16.76 Gb           0 b         10496
                      aten::silu_         6.56%        1.498s         6.56%        1.498s     148.120us           0 b           0 b         10112
                 aten::max_pool2d         0.01%       2.979ms         2.15%     491.325ms       1.279ms     166.64 Mb    -333.28 Mb           384
    aten::max_pool2d_with_indices         2.14%     488.346ms         2.14%     488.346ms       1.272ms     499.92 Mb     499.92 Mb           384
                        aten::cat         1.38%     314.089ms         1.57%     358.755ms     136.357us       5.97 Gb       5.97 Gb          2631
                      aten::empty         0.92%     209.370ms         0.92%     209.370ms       8.396us      17.81 Gb      17.81 Gb         24936
                      aten::copy_         0.49%     110.914ms         0.49%     110.914ms      16.908us           0 b           0 b          6560
                        aten::add         0.47%     106.873ms         0.47%     106.873ms      38.074us       2.51 Gb       2.51 Gb          2807
                      aten::clone         0.03%       6.773ms         0.38%      86.301ms      61.337us       1.05 Gb           0 b          1407
                 aten::contiguous         0.01%       2.597ms         0.38%      86.012ms      74.534us       1.05 Gb           0 b          1154
         aten::upsample_nearest2d         0.14%      32.338ms         0.35%      80.695ms     315.214us     666.56 Mb     368.13 Mb           256
                        aten::mul         0.30%      68.287ms         0.34%      77.951ms      36.839us      72.99 Mb      72.98 Mb          2116
                    aten::sigmoid         0.30%      68.386ms         0.30%      68.386ms     178.087us     774.62 Mb     774.62 Mb           384
                         aten::to         0.03%       6.971ms         0.24%      54.570ms      11.098us     444.43 Mb           0 b          4917
                aten::as_strided_         0.22%      50.129ms         0.22%      50.129ms       4.664us           0 b           0 b         10749
                   aten::_to_copy         0.07%      16.059ms         0.21%      47.599ms      14.632us     444.43 Mb          24 b          3253
                     aten::narrow         0.08%      18.548ms         0.20%      45.486ms      11.414us           0 b           0 b          3985
                      aten::slice         0.12%      26.443ms         0.15%      34.825ms       5.166us           0 b           0 b          6741
                        aten::div         0.11%      24.324ms         0.15%      34.067ms      29.883us     444.58 Mb     444.57 Mb          1140
                     aten::select         0.11%      25.517ms         0.13%      30.071ms       4.529us           0 b           0 b          6639
                      aten::index         0.05%      11.975ms         0.10%      23.402ms      30.833us       3.90 Mb       3.75 Mb           759
                    aten::type_as         0.00%     723.080us         0.10%      22.679ms     177.179us     444.38 Mb           0 b           128
                     aten::arange         0.05%      10.845ms         0.09%      21.423ms      11.771us     512.55 Kb      33.25 Kb          1820
                         aten::gt         0.07%      16.781ms         0.09%      19.579ms      77.389us       2.29 Mb       2.29 Mb           253
                 aten::as_strided         0.08%      18.350ms         0.08%      18.350ms       1.004us           0 b           0 b         18282
                 torchvision::nms         0.02%       5.121ms         0.08%      17.764ms     142.114us      75.77 Kb    -388.34 Kb           125
                      aten::stack         0.01%       2.525ms         0.08%      17.544ms      53.162us       5.22 Mb           0 b           330
                        aten::sub         0.05%      10.726ms         0.07%      16.408ms      15.109us      15.83 Mb      15.83 Mb          1086
                 aten::empty_like         0.02%       5.551ms         0.06%      12.671ms       9.907us       1.05 Gb       1.11 Kb          1279
                    aten::resize_         0.05%      12.300ms         0.05%      12.300ms       1.078us     223.03 Kb     223.03 Kb         11406
                        aten::pow         0.04%       9.120ms         0.04%       9.525ms      24.804us      18.23 Mb      18.23 Mb           384
                       aten::sort         0.02%       5.000ms         0.04%       9.018ms      36.073us     227.32 Kb      75.77 Kb           250
              aten::empty_strided         0.04%       8.054ms         0.04%       8.054ms       2.297us     444.61 Mb     444.61 Mb          3506
                    aten::nonzero         0.03%       6.905ms         0.03%       7.867ms      31.093us     158.28 Kb     158.28 Kb           253
                   aten::meshgrid         0.02%       3.917ms         0.03%       7.357ms      22.294us           0 b           0 b           330
                    aten::argsort         0.00%     410.487us         0.03%       6.027ms      48.215us      76.05 Kb     -37.34 Kb           125
                     aten::expand         0.02%       4.031ms         0.02%       5.312ms       4.024us           0 b           0 b          1320
           aten::split_with_sizes         0.01%       3.407ms         0.02%       4.528ms      11.791us           0 b           0 b           384
                 aten::index_put_         0.00%     821.698us         0.02%       3.901ms      15.236us           0 b           0 b           256
                       aten::view         0.02%       3.880ms         0.02%       3.880ms       1.326us           0 b           0 b          2926
                      aten::zeros         0.01%       1.915ms         0.02%       3.652ms       9.661us      85.25 Kb         560 b           378
                        aten::max         0.02%       3.479ms         0.02%       3.479ms      27.830us     123.76 Kb     123.76 Kb           125
                       aten::sub_         0.01%       1.275ms         0.01%       3.260ms      12.736us          68 b        -888 b           256
           aten::_index_put_impl_         0.01%       2.356ms         0.01%       3.079ms      12.027us           0 b           0 b           256
                    aten::permute         0.01%       1.851ms         0.01%       3.078ms       8.016us           0 b           0 b           384
                       aten::ones         0.01%       1.518ms         0.01%       2.598ms      10.737us      66.34 Kb           0 b           242
                    aten::reshape         0.01%       1.897ms         0.01%       2.597ms       2.559us           0 b           0 b          1015
                  aten::unsqueeze         0.01%       1.636ms         0.01%       2.136ms       3.237us           0 b           0 b           660
                     aten::clamp_         0.01%       1.740ms         0.01%       1.822ms       3.558us           0 b           0 b           512
                       aten::div_         0.00%     686.445us         0.01%       1.526ms      11.926us          24 b        -464 b           128
                      aten::fill_         0.01%       1.253ms         0.01%       1.253ms       3.385us           0 b           0 b           370
                      aten::zero_         0.00%     599.847us         0.01%       1.147ms       3.034us           0 b           0 b           378
                     aten::unbind         0.00%     381.165us         0.00%       1.120ms       8.751us           0 b           0 b           128
                       aten::mul_         0.00%       1.018ms         0.00%       1.018ms       8.147us           0 b           0 b           125
                 aten::lift_fresh         0.00%     344.569us         0.00%     344.569us       0.449us           0 b           0 b           768
                aten::result_type         0.00%     292.682us         0.00%     292.682us       0.762us           0 b           0 b           384
                      aten::alias         0.00%     198.929us         0.00%     198.929us       1.554us           0 b           0 b           128
                    aten::detach_         0.00%     113.341us         0.00%     113.341us       0.885us           0 b           0 b           128
                         [memory]         0.00%       0.000us         0.00%       0.000us       0.000us     -28.54 Gb     -28.54 Gb         30081
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 22.832s

`YOLOv5l`

**CPU Profiler**
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                     aten::conv2d         0.31%     121.546ms        90.68%       35.718s       2.683ms      26.72 Gb           0 b         13312
                aten::convolution         0.21%      82.663ms        90.37%       35.596s       2.674ms      26.72 Gb           0 b         13312
               aten::_convolution         0.37%     146.725ms        90.16%       35.514s       2.668ms      26.72 Gb           0 b         13312
         aten::mkldnn_convolution        88.83%       34.990s        89.79%       35.367s       2.657ms      26.72 Gb           0 b         13312
                      aten::silu_         5.00%        1.968s         5.00%        1.968s     152.209us           0 b           0 b         12928
                 aten::max_pool2d         0.01%       3.737ms         1.51%     595.459ms       1.551ms     222.19 Mb    -444.38 Mb           384
    aten::max_pool2d_with_indices         1.50%     591.722ms         1.50%     591.722ms       1.541ms     666.56 Mb     666.56 Mb           384
                        aten::cat         0.89%     351.431ms         1.00%     395.685ms     150.336us       7.45 Gb       7.45 Gb          2632
                      aten::empty         0.78%     308.682ms         0.78%     308.682ms      10.095us      27.87 Gb      27.87 Gb         30578
                        aten::add         0.45%     175.578ms         0.45%     175.578ms      47.377us       5.01 Gb       5.01 Gb          3706
                      aten::copy_         0.40%     156.572ms         0.40%     156.572ms      23.792us           0 b           0 b          6581
         aten::upsample_nearest2d         0.10%      38.567ms         0.34%     132.341ms     516.956us     888.75 Mb     490.90 Mb           256
                 aten::contiguous         0.01%       3.143ms         0.24%      95.991ms      82.894us       1.14 Gb           0 b          1158
                      aten::clone         0.02%       6.344ms         0.24%      95.393ms      67.559us       1.15 Gb           0 b          1412
                        aten::mul         0.15%      58.078ms         0.17%      66.395ms      31.348us      72.99 Mb      72.99 Mb          2118
                aten::as_strided_         0.16%      64.633ms         0.16%      64.633ms       4.764us           0 b           0 b         13566
                    aten::sigmoid         0.15%      59.362ms         0.15%      59.362ms     154.588us     774.62 Mb     774.62 Mb           384
                         aten::to         0.02%       5.917ms         0.12%      48.841ms       9.919us     444.43 Mb           0 b          4924
                     aten::narrow         0.04%      17.594ms         0.11%      45.084ms      11.310us           0 b           0 b          3986
                   aten::_to_copy         0.04%      13.879ms         0.11%      42.924ms      13.167us     444.43 Mb         104 b          3260
                      aten::slice         0.07%      26.454ms         0.09%      34.509ms       5.103us           0 b           0 b          6762
                        aten::div         0.06%      25.477ms         0.09%      34.021ms      29.738us     444.60 Mb     444.59 Mb          1144
                     aten::select         0.06%      22.885ms         0.07%      27.044ms       4.062us           0 b           0 b          6658
                    aten::type_as         0.00%     637.690us         0.05%      21.166ms     165.358us     444.38 Mb           0 b           128
                      aten::index         0.03%      10.676ms         0.05%      20.670ms      27.126us       4.21 Mb       4.04 Mb           762
                     aten::arange         0.02%       9.639ms         0.05%      19.273ms      10.566us     542.86 Kb      34.58 Kb          1824
                         aten::gt         0.04%      16.253ms         0.05%      18.583ms      73.161us       2.29 Mb       2.29 Mb           254
                 aten::as_strided         0.04%      17.062ms         0.04%      17.062ms       0.931us           0 b           0 b         18327
                 torchvision::nms         0.01%       4.706ms         0.04%      16.924ms     134.321us      83.35 Kb    -427.18 Kb           126
                    aten::resize_         0.04%      15.695ms         0.04%      15.695ms       1.103us     236.85 Kb     236.85 Kb         14224
                      aten::stack         0.01%       2.339ms         0.04%      15.558ms      47.145us       5.22 Mb           0 b           330
                        aten::sub         0.02%       8.305ms         0.03%      13.317ms      12.218us      15.84 Mb      15.84 Mb          1090
                 aten::empty_like         0.01%       5.546ms         0.03%      12.389ms       9.649us       1.15 Gb       1.24 Kb          1284
                       aten::sort         0.01%       4.630ms         0.02%       8.225ms      32.641us     250.05 Kb      83.35 Kb           252
              aten::empty_strided         0.02%       6.933ms         0.02%       6.933ms       1.973us     444.63 Mb     444.63 Mb          3514
                        aten::pow         0.02%       6.521ms         0.02%       6.883ms      17.925us      18.23 Mb      18.23 Mb           384
                    aten::nonzero         0.02%       6.081ms         0.02%       6.844ms      26.943us     172.25 Kb     172.25 Kb           254
                   aten::meshgrid         0.01%       3.712ms         0.02%       6.801ms      20.610us           0 b           0 b           330
                    aten::argsort         0.00%     395.973us         0.01%       5.438ms      43.157us      83.46 Kb     -41.46 Kb           126
                     aten::expand         0.01%       3.476ms         0.01%       4.581ms       3.470us           0 b           0 b          1320
           aten::split_with_sizes         0.01%       3.004ms         0.01%       3.936ms      10.249us           0 b           0 b           384
                 aten::index_put_         0.00%     791.184us         0.01%       3.551ms      13.870us           0 b           0 b           256
                       aten::view         0.01%       3.449ms         0.01%       3.449ms       1.178us           0 b           0 b          2928
                      aten::zeros         0.00%       1.715ms         0.01%       3.269ms       8.602us      93.77 Kb           0 b           380
                       aten::sub_         0.00%       1.090ms         0.01%       2.839ms      11.090us          64 b        -896 b           256
                    aten::permute         0.00%       1.669ms         0.01%       2.819ms       7.342us           0 b           0 b           384
                       aten::ones         0.00%       1.679ms         0.01%       2.766ms      11.429us      66.34 Kb           0 b           242
           aten::_index_put_impl_         0.01%       2.055ms         0.01%       2.760ms      10.780us           0 b           0 b           256
                    aten::reshape         0.00%       1.770ms         0.01%       2.421ms       2.378us           0 b           0 b          1018
                        aten::max         0.01%       2.292ms         0.01%       2.292ms      18.187us     133.35 Kb     133.35 Kb           126
                  aten::unsqueeze         0.00%       1.457ms         0.00%       1.898ms       2.876us           0 b           0 b           660
                     aten::clamp_         0.00%       1.653ms         0.00%       1.726ms       3.371us           0 b           0 b           512
                       aten::div_         0.00%     622.232us         0.00%       1.410ms      11.019us          40 b        -432 b           128
                      aten::fill_         0.00%       1.231ms         0.00%       1.231ms       3.326us           0 b           0 b           370
                     aten::unbind         0.00%     417.735us         0.00%       1.088ms       8.501us           0 b           0 b           128
                      aten::zero_         0.00%     506.702us         0.00%       1.010ms       2.658us           0 b           0 b           380
                       aten::mul_         0.00%     896.195us         0.00%     896.195us       7.113us           0 b           0 b           126
                 aten::lift_fresh         0.00%     273.455us         0.00%     273.455us       0.356us           0 b           0 b           768
                aten::result_type         0.00%     265.221us         0.00%     265.221us       0.691us           0 b           0 b           384
                      aten::alias         0.00%     186.951us         0.00%     186.951us       1.461us           0 b           0 b           128
                    aten::detach_         0.00%      71.038us         0.00%      71.038us       0.555us           0 b           0 b           128
                         [memory]         0.00%       0.000us         0.00%       0.000us       0.000us     -42.76 Gb     -42.76 Gb         33867
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 39.389s

`YOLOv5x`

**CPU Profiler**
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                     aten::conv2d         0.27%     186.451ms        92.77%       64.950s       4.027ms      39.00 Gb           0 b         16128
                aten::convolution         0.17%     121.121ms        92.50%       64.764s       4.016ms      39.00 Gb           0 b         16128
               aten::_convolution         0.28%     197.819ms        92.33%       64.643s       4.008ms      39.00 Gb           0 b         16128
         aten::mkldnn_convolution        91.14%       63.811s        92.05%       64.445s       3.996ms      39.00 Gb           0 b         16128
                      aten::silu_         4.12%        2.882s         4.12%        2.882s     183.030us           0 b           0 b         15744
                 aten::max_pool2d         0.01%       4.336ms         1.06%     739.255ms       1.925ms     277.73 Mb    -555.47 Mb           384
    aten::max_pool2d_with_indices         1.05%     734.919ms         1.05%     734.919ms       1.914ms     833.20 Mb     833.20 Mb           384
                      aten::empty         0.77%     538.325ms         0.77%     538.325ms      14.867us      40.24 Gb      40.24 Gb         36210
                        aten::cat         0.63%     444.024ms         0.70%     492.049ms     186.949us       8.93 Gb       8.93 Gb          2632
                        aten::add         0.48%     335.008ms         0.48%     335.008ms      72.796us       8.34 Gb       8.34 Gb          4602
                      aten::copy_         0.26%     182.920ms         0.26%     182.920ms      27.795us           0 b           0 b          6581
         aten::upsample_nearest2d         0.07%      46.357ms         0.23%     159.652ms     623.640us       1.08 Gb     613.67 Mb           256
                 aten::contiguous         0.00%       2.891ms         0.15%     108.453ms      93.656us       1.24 Gb           0 b          1158
                      aten::clone         0.01%       7.276ms         0.15%     108.415ms      76.781us       1.24 Gb           0 b          1412
                aten::as_strided_         0.13%      91.929ms         0.13%      91.929ms       5.612us           0 b           0 b         16382
                        aten::mul         0.09%      62.384ms         0.10%      72.105ms      34.044us      73.00 Mb      72.99 Mb          2118
                         aten::to         0.01%       7.066ms         0.09%      63.598ms      12.916us     444.44 Mb           0 b          4924
                    aten::sigmoid         0.09%      60.868ms         0.09%      60.868ms     158.510us     774.62 Mb     774.62 Mb           384
                   aten::_to_copy         0.02%      15.813ms         0.08%      56.532ms      17.341us     444.44 Mb          36 b          3260
                        aten::div         0.06%      39.318ms         0.07%      48.955ms      42.793us     444.60 Mb     444.59 Mb          1144
                     aten::narrow         0.03%      19.316ms         0.07%      48.813ms      12.246us           0 b           0 b          3986
                      aten::slice         0.04%      28.801ms         0.05%      37.412ms       5.533us           0 b           0 b          6762
                    aten::type_as         0.00%     700.320us         0.05%      32.131ms     251.027us     444.38 Mb           0 b           128
                     aten::select         0.04%      26.672ms         0.05%      31.531ms       4.736us           0 b           0 b          6658
                      aten::index         0.02%      11.637ms         0.03%      22.706ms      29.798us       4.28 Mb       4.11 Mb           762
                     aten::arange         0.02%      10.618ms         0.03%      20.892ms      11.454us     551.39 Kb      40.64 Kb          1824
                    aten::resize_         0.03%      20.582ms         0.03%      20.582ms       1.208us     235.06 Kb     235.06 Kb         17040
                         aten::gt         0.02%      17.243ms         0.03%      19.869ms      78.226us       2.29 Mb       2.29 Mb           254
                 aten::as_strided         0.03%      19.007ms         0.03%      19.007ms       1.037us           0 b           0 b         18327
                 aten::empty_like         0.01%       7.589ms         0.03%      18.765ms      14.614us       1.24 Gb          64 b          1284
                 torchvision::nms         0.01%       5.096ms         0.03%      18.184ms     144.318us      85.48 Kb    -438.11 Kb           126
                      aten::stack         0.00%       2.477ms         0.02%      16.699ms      50.602us       5.22 Mb           0 b           330
                        aten::sub         0.01%       8.678ms         0.02%      14.276ms      13.097us      15.85 Mb      15.85 Mb          1090
                       aten::sort         0.01%       4.991ms         0.01%       8.926ms      35.422us     256.45 Kb      85.48 Kb           252
              aten::empty_strided         0.01%       7.711ms         0.01%       7.711ms       2.194us     444.63 Mb     444.63 Mb          3514
                        aten::pow         0.01%       7.094ms         0.01%       7.484ms      19.490us      18.23 Mb      18.23 Mb           384
                    aten::nonzero         0.01%       6.592ms         0.01%       7.462ms      29.379us     175.69 Kb     175.69 Kb           254
                   aten::meshgrid         0.01%       3.883ms         0.01%       7.298ms      22.115us           0 b           0 b           330
                    aten::argsort         0.00%     386.118us         0.01%       5.734ms      45.512us      86.47 Kb     -40.77 Kb           126
                     aten::expand         0.01%       3.927ms         0.01%       5.086ms       3.853us           0 b           0 b          1320
           aten::split_with_sizes         0.00%       3.395ms         0.01%       4.484ms      11.676us           0 b           0 b           384
                 aten::index_put_         0.00%     851.939us         0.01%       3.967ms      15.497us           0 b           0 b           256
                       aten::view         0.01%       3.905ms         0.01%       3.905ms       1.332us           0 b           0 b          2932
                      aten::zeros         0.00%       1.850ms         0.01%       3.597ms       9.467us      96.17 Kb           0 b           380
                       aten::sub_         0.00%       1.245ms         0.00%       3.249ms      12.692us          68 b        -888 b           256
                    aten::permute         0.00%       1.893ms         0.00%       3.231ms       8.415us           0 b           0 b           384
           aten::_index_put_impl_         0.00%       2.320ms         0.00%       3.115ms      12.170us           0 b           0 b           256
                       aten::ones         0.00%       1.825ms         0.00%       2.910ms      12.024us      66.34 Kb           0 b           242
                    aten::reshape         0.00%       1.880ms         0.00%       2.606ms       2.560us           0 b           0 b          1018
                        aten::max         0.00%       2.427ms         0.00%       2.427ms      19.259us     135.30 Kb     135.30 Kb           126
                  aten::unsqueeze         0.00%       1.648ms         0.00%       2.133ms       3.232us           0 b           0 b           660
                     aten::clamp_         0.00%       1.787ms         0.00%       1.872ms       3.656us           0 b           0 b           512
                       aten::div_         0.00%     694.593us         0.00%       1.546ms      12.077us          24 b        -464 b           128
                      aten::fill_         0.00%       1.227ms         0.00%       1.227ms       3.317us           0 b           0 b           370
                      aten::zero_         0.00%     597.202us         0.00%       1.126ms       2.964us           0 b           0 b           380
                     aten::unbind         0.00%     356.305us         0.00%       1.084ms       8.472us           0 b           0 b           128
                       aten::mul_         0.00%     958.276us         0.00%     958.276us       7.605us           0 b           0 b           126
                aten::result_type         0.00%     282.361us         0.00%     282.361us       0.735us           0 b           0 b           384
                 aten::lift_fresh         0.00%     266.608us         0.00%     266.608us       0.347us           0 b           0 b           768
                      aten::alias         0.00%     198.784us         0.00%     198.784us       1.553us           0 b           0 b           128
                    aten::detach_         0.00%      75.336us         0.00%      75.336us       0.589us           0 b           0 b           128
                         [memory]         0.00%       0.000us         0.00%       0.000us       0.000us     -60.12 Gb     -60.12 Gb         37564
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 70.014s

`line_profiler` Results:

### `YOLOv5n`

Timer unit: 1e-09 s

Total time: 5.64134 s

File: /tmp/ipykernel_20890/1432606016.py

Function: batch_inference at line 1

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                       |
|--------|------|-----------|--------------|--------|-------------------------------------|
| 1      |      |           |              |        | def batch_inference(model, dataset): |
| 2      | 1    | 23,979    | 23,979       | 0.0    | results = []                        |
| 3      | 129  | 106,979   | 829.3        | 0.0    | for image in dataset:               |
| 4      | 128  | 5,641,032,681 | 44,070,566 | 100.0  | result = model(image)               |
| 5      | 128  | 177,343   | 1,385.5      | 0.0    | results.append(result)              |
| 6      | 1    | 111       | 111.0        | 0.0    | return results                      |

### `YOLOv5s`

Timer unit: 1e-09 s

Total time: 10.5091 s

File: /tmp/ipykernel_20890/1432606016.py

Function: batch_inference at line 1

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                       |
|--------|------|-----------|--------------|--------|-------------------------------------|
| 1      |      |           |              |        | def batch_inference(model, dataset): |
| 2      | 1    | 557       | 557.0        | 0.0    | results = []                        |
| 3      | 129  | 126,381   | 979.7        | 0.0    | for image in dataset:               |
| 4      | 128  | 10,509,000,000 | 82,070,313 | 100.0  | result = model(image)               |
| 5      | 128  | 189,695   | 1,482.0      | 0.0    | results.append(result)              |
| 6      | 1    | 167       | 167.0        | 0.0    | return results                      |

### `YOLOv5m`

Timer unit: 1e-09 s

Total time: 21.0146 s

File: /tmp/ipykernel_20890/1432606016.py

Function: batch_inference at line 1

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                       |
|--------|------|-----------|--------------|--------|-------------------------------------|
| 1      |      |           |              |        | def batch_inference(model, dataset): |
| 2      | 1    | 1,138     | 1,138.0      | 0.0    | results = []                        |
| 3      | 129  | 88,548    | 686.4        | 0.0    | for image in dataset:               |
| 4      | 128  | 21,014,000,000 | 164,171,875 | 100.0  | result = model(image)               |
| 5      | 128  | 183,560   | 1,434.1      | 0.0    | results.append(result)              |
| 6      | 1    | 142       | 142.0        | 0.0    | return results                      |

### `YOLOv5l`

Timer unit: 1e-09 s

Total time: 39.5307 s

File: /tmp/ipykernel_20890/1432606016.py

Function: batch_inference at line 1

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                       |
|--------|------|-----------|--------------|--------|-------------------------------------|
| 1      |      |           |              |        | def batch_inference(model, dataset): |
| 2      | 1    | 593       | 593.0        | 0.0    | results = []                        |
| 3      | 129  | 108,482   | 840.9        | 0.0    | for image in dataset:               |
| 4      | 128  | 39,530,000,000 | 308,046,875 | 100.0  | result = model(image)               |
| 5      | 128  | 173,890   | 1,358.5      | 0.0    | results.append(result)              |
| 6      | 1    | 240       | 240.0        | 0.0    | return results                      |

### `YOLOv5x`

Timer unit: 1e-09 s

Total time: 66.9742 s

File: /tmp/ipykernel_20890/1432606016.py

Function: batch_inference at line 1

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                       |
|--------|------|-----------|--------------|--------|-------------------------------------|
| 1      |      |           |              |        | def batch_inference(model, dataset): |
| 2      | 1    | 704       | 704.0        | 0.0    | results = []                        |
| 3      | 129  | 94,068    | 729.2        | 0.0    | for image in dataset:               |
| 4      | 128  | 66,974,000,000 | 523,234,375 | 100.0  | result = model(image)               |
| 5      | 128  | 186,582   | 1,457.7      | 0.0    | results.append(result)              |
| 6      | 1    | 176       | 176.0        | 0.0    | return results                      |

### Task 4: Optimization

#### Method 1: PyTorch Quantization

Line Profiler Results

`YOLOv5n`

Timer unit: 1e-09 s

Total time: 8.62739 s

File: /tmp/ipykernel_1162/1538104890.py

Function: inference at line 1

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                     |
|--------|------|-----------|--------------|--------|-----------------------------------|
| 1      |      |           |              |        | `def inference(model, dataset):` |
| 2      | 1    | 5,943     | 5,943        | 0.0    | `results = []`                   |
| 3      | 129  | 134,673   | 1,044        | 0.0    | `for image in dataset:`          |
| 4      | 128  | 8,626,993,911 | 67,421,828 | 100.0  | `result = model(image)`          |
| 5      | 128  | 252,216   | 1,970        | 0.0    | `results.append(result)`         |
| 6      | 1    | 198       | 198          | 0.0    | `return results`                 |

`YOLOv5s`

Timer unit: 1e-09 s

Total time: 13.7571 s

File: /tmp/ipykernel_1162/1538104890.py

Function: inference at line 1

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                     |
|--------|------|-----------|--------------|--------|-----------------------------------|
| 1      |      |           |              |        | `def inference(model, dataset):` |
| 2      | 1    | 2,855     | 2,855        | 0.0    | `results = []`                   |
| 3      | 129  | 134,712   | 1,044.3      | 0.0    | `for image in dataset:`          |
| 4      | 128  | 10,000,000,000 | 78,125,000 | 100.0  | `result = model(image)`          |
| 5      | 128  | 225,032   | 1,758.1      | 0.0    | `results.append(result)`         |
| 6      | 1    | 174       | 174          | 0.0    | `return results`                 |

`YOLOv5m`

Timer unit: 1e-09 s

Total time: 26.8231 s

File: /tmp/ipykernel_1162/1538104890.py

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                     |
|--------|------|-----------|--------------|--------|-----------------------------------|
| 1      | 1    | 1,413     | 1,413        | 0.0    | `def inference(model, dataset):` |
| 2      | 1    | 1413.0    | 1413.0       | 0.0    | `results = []`                   |
| 3      | 129  | 150,916   | 1,169.9      | 0.0    | `for image in dataset:`          |
| 4      | 128  | 30,000,000,000 | 234,375,000 | 100.0  | `result = model(image)`          |
| 5      | 128  | 256,954   | 2,007.5      | 0.0    | `results.append(result)`         |
| 6      | 1    | 228       | 228.0        | 0.0    | `return results`                 |

`YOLOv5l`

Timer unit: 1e-09 s

Total time: 52.5227 s

File: /tmp/ipykernel_1201/1538104890.py

| Line # | Hits | Time (ns) | Per Hit (ns) | % Time | Line Contents                     |
|--------|------|-----------|--------------|--------|-----------------------------------|
| 1      | 1    | 1,286.0   | 1,286.0      | 0.0    | def inference(model, dataset):   |
| 2      | 1    | 1,286.0   | 1,286.0      | 0.0    | results = []                     |
| 3      | 129  | 108,864.0 | 843.9        | 0.0    | for image in dataset:            |
| 4      | 128  | 50,000,000,000 | 390,625,000 | 100.0  | result = model(image)            |
| 5      | 128  | 257,988.0 | 2,015.5      | 0.0    | results.append(result)           |
| 6      | 1    | 288.0     | 288.0        | 0.0    | return results                   |

`YOLOv5x`

Timer unit: 1e-09 s

Total time: 77.5109 s

File: /tmp/ipykernel_1201/1538104890.py

| Line # | Hits | Time       | Per Hit   | % Time | Line Contents                  |
|--------|------|------------|-----------|--------|--------------------------------|
| 1      | 1    | 814.0      | 814.0     | 0.0    | def inference(model, dataset): |
| 2      | 1    | 814.0      | 814.0     | 0.0    | results = []                   |
| 3      | 129  | 117,801.0  | 913.2     | 0.0    | for image in dataset:          |
| 4      | 128  | 80,000,000,000      | 600,000,000     | 100.0  | result = model(image)          |
| 5      | 128  | 207,478.0  | 1,620.9   | 0.0    | results.append(result)         |
| 6      | 1    | 240.0      | 240.0     | 0.0    | return results                 |

#### Method 2: Using ONNX

`YOLOv5n`

Timer unit: 1e-09 s

Total time: 6.76007 s

| Line # | Hits | Time (ns)      | Per Hit (ns) | % Time | Line Contents                          |
|--------|------|----------------|--------------|--------|----------------------------------------|
| 9      |      |                |              |        | def inference_onnx(model_session, dataset): |
| 10     | 1    | 1,047.0         | 1047.0       | 0.0    | results = []                           |
| 11     | 129  | 260,044.0       | 2015.8       | 0.0    | for image_path in dataset:             |
| 12     | 128  | 1,447,650,156.0   | 11309173.1   | 21.4   | input_data = preprocess_image(image_path) |
| 13     | 128  | 2,635,143.0      | 20587.1      | 0.0    | input_name = model_session.get_inputs()[0].name |
| 14     | 128  | 5,308,867,294.0   | 41475835.1   | 78.5   | result = model_session.run(None, {input_name: input_data}) |
| 15     | 128  | 657,842.0       | 5139.4       | 0.0    | results.append(result)                 |
| 16     | 1    | 318.0          | 318.0        | 0.0    | return results                         |

---

`YOLOv5s`

Timer unit: 1e-09 s

Total time: 15.0752 s

| Line # | Hits | Time (ns)      | Per Hit (ns) | % Time | Line Contents                          |
|--------|------|----------------|--------------|--------|----------------------------------------|
| 9      |      |                |              |        | def inference_onnx(model_session, dataset): |
| 10     | 1    | 1,115.0         | 1115.0       | 0.0    | results = []                           |
| 11     | 129  | 258,071.0       | 2000.6       | 0.0    | for image_path in dataset:             |
| 12     | 128  | 1,542,414,593.0   | 12050113.2   | 10.2   | input_data = preprocess_image(image_path) |
| 13     | 128  | 2,864,807.0      | 22381.3      | 0.0    | input_name = model_session.get_inputs()[0].name |
| 14     | 128  | 13,500,000,000.0  | 105468750.0  | 89.7   | result = model_session.run(None, {input_name: input_data}) |
| 15     | 128  | 718,826.0       | 5615.8       | 0.0    | results.append(result)                 |
| 16     | 1    | 254.0          | 254.0        | 0.0    | return results                         |

---

`YOLOv5m`

Timer unit: 1e-09 s

Total time: 30.9797 s

| Line # | Hits | Time (ns)      | Per Hit (ns) | % Time | Line Contents                          |
|--------|------|----------------|--------------|--------|----------------------------------------|
| 9      |      |                |              |        | def inference_onnx(model_session, dataset): |
| 10     | 1    | 1,122.0         | 1122.0       | 0.0    | results = []                           |
| 11     | 129  | 192,094.0       | 1489.1       | 0.0    | for image_path in dataset:             |
| 12     | 128  | 1,420,368,423.0   | 11096534.5   | 4.6    | input_data = preprocess_image(image_path) |
| 13     | 128  | 2,679,434.0      | 20933.1      | 0.0    | input_name = model_session.get_inputs()[0].name |
| 14     | 128  | 29,500,000,000.0  | 230468750.0  | 95.4   | result = model_session.run(None, {input_name: input_data}) |
| 15     | 128  | 713,389.0       | 5573.4       | 0.0    | results.append(result)                 |
| 16     | 1    | 339.0          | 339.0        | 0.0    | return results                         |

---

`YOLOv5l`

Timer unit: 1e-09 s

Total time: 56.4818 s

| Line # | Hits | Time (ns)      | Per Hit (ns) | % Time | Line Contents                          |
|--------|------|----------------|--------------|--------|----------------------------------------|
| 9      |      |                |              |        | def inference_onnx(model_session, dataset): |
| 10     | 1    | 1,189.0         | 1189.0       | 0.0    | results = []                           |
| 11     | 129  | 195,777.0       | 1517.7       | 0.0    | for image_path in dataset:             |
| 12     | 128  | 1,407,053,642.0   | 10914247.2   | 2.5    | input_data = preprocess_image(image_path) |
| 13     | 128  | 2,478,226.0      | 19361.1      | 0.0    | input_name = model_session.get_inputs()[0].name |
| 14     | 128  | 55,000,000,000.0  | 429687500.0  | 97.5   | result = model_session.run(None, {input_name: input_data}) |
| 15     | 128  | 665,050.0       | 5195.7       | 0.0    | results.append(result)                 |
| 16     | 1    | 338.0          | 338.0        | 0.0    | return results                         |

---

`YOLOv5x`

Timer unit: 1e-09 s

Total time: 96.4681 s

| Line # | Hits | Time (ns)      | Per Hit (ns) | % Time | Line Contents                          |
|--------|------|----------------|--------------|--------|----------------------------------------|
| 9      |      |                |              |        | def inference_onnx(model_session, dataset): |
| 10     | 1    | 1,413.0         | 1413.0       | 0.0    | results = []                           |
| 11     | 129  | 218,030.0       | 1690.2       | 0.0    | for image_path in dataset:             |
| 12     | 128  | 2,789,174,571.0   | 21790267.7   | 2.9    | input_data = preprocess_image(image_path) |
| 13     | 128  | 2,503,744.0      | 19560.5      | 0.0    | input_name = model_session.get_inputs()[0].name |
| 14     | 128  | 93,500,000,000.0  | 730468750.0  | 97.1   | result = model_session.run(None, {input_name: input_data}) |
| 15     | 128  | 679,960.0       | 5312.2       | 0.0    | results.append(result)                 |
| 16     | 1    | 298.0          | 298.0        | 0.0    | return results                         |

### Method 3: Multi threading

### Time breakdown of various stages

### Optimization suggestions and results

### Summary
