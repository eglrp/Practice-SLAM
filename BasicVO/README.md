# 简单视觉里程计

参考fengbing的博客，以及fengbing博客中的引用博客，在KITTI其中的一个odometry数据集进行测试。

单目SLAM，其中没有使用初始化，而是直接读取GroundTruth文件，计算scale。

具体步骤如下：

1. 初始化第一帧，计算FAST特征点；
2. 初始化第二帧，根据第一帧的FAST特征点，使用OpenCV中的光流，计算跟踪这些关键点。并计算本质矩阵，用本质矩阵分解获取旋转矩阵和平移向量；
3. 循环处理后续的帧，首先进行跟踪，然后计算本质矩阵，分解恢复到旋转矩阵和平移向量，然后利用前一帧的旋转矩阵进行局部约束，获取相对的旋转矩阵和平移向量；
4. 如果跟踪的点数低于一定的阈值，那么重新检测一下（这里原始的代码有点错误）
5. 循环，直到所有的帧都被处理完成。


下面是原始博客代码运行的结果:
![原始结果](https://github.com/zhangxiaoya/Practice-SLAM/blob/master/BasicVO/images/Trajectory_2.png)

修改代码错误后的结果如下：
![修改后的结果](https://github.com/zhangxiaoya/Practice-SLAM/blob/master/BasicVO/images/Trajectory_1.png)