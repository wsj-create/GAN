# 生成对抗网络GAN源码实战篇

> 1、准备数据集
>
> 2、加载数据集
>
> 3、搭建神经网络
>
> 4、创建网络模型
>
> 5、设置损失函数、优化器等参数
>
> 6、训练网络
>
> 7、获取模型结果



----

## MNIST数据集



![image-20230129175658915](https://tutouxiaosu.oss-cn-beijing.aliyuncs.com/img/img/image-20230129175658915.png)

<center>格式：单通道28×28的灰度图</center>







## BCELOSS解析【秃头小苏-->DCGAN】

​	这部分我在上一篇[GAN网络](https://juejin.cn/post/7120443914854613029)讲解中已经介绍过，但是我没有细讲，这里我想重点讲一下BCELOSS损失函数。【就是二值交叉熵损失函数啦】我们先来看一下pytorch官网对这个函数的解释，如下图所示：

![image-20220723142323032](https://tutouxiaosu.oss-cn-beijing.aliyuncs.com/img/img/image-20220723142323032.png)

​		其中N表示batch_size，$w_n$表示一个权重系数，$y_n$表示标签值，$x_n$表示数据。我们会对每个batch_size的数据都计算一个$l_n$ ，最后求平均或求和。【默认求均值】

​		看到这里大家可能还是一知半解，不用担心，我举一个小例子大家就明白了。首先我们初始化一些输入数据和标签：

```python
import torch
import math
input = torch.randn(3,3)
target = torch.FloatTensor([[0, 1, 1], [1, 1, 0], [0, 0, 0]])
```

​		来看看输入数据和标签的结果：

![image-20220723144544905](https://tutouxiaosu.oss-cn-beijing.aliyuncs.com/img/img/image-20220723144544905.png)

​		接着我们要让输入数据经过Sigmoid函数将其归一化到[0,1]之间【BCELOSS函数要求】：

```python 
m = torch.nn.Sigmoid()
m(input)
```

​		输出的结果如下：

![image-20220723145022493](https://tutouxiaosu.oss-cn-beijing.aliyuncs.com/img/img/image-20220723145022493.png)

​		最后我们就可以使用BCELOSS函数计算输入数据和标签的损失了：

```python
loss =torch.nn.BCELoss()
loss(m(input), target)
```

​		输出结果如下：

![](https://tutouxiaosu.oss-cn-beijing.aliyuncs.com/img/img/image-20220723145932793.png)

​		==大家记住这个值喔！！！==

​		上文似乎只是介绍了BCELOSS怎么用，具体怎么算的好像并不清楚，下面我们就根据官方给的公式来一步一步手动计算这个损失，看看结果和调用函数是否一致，如下：

```python
r11 = 0 * math.log(0.8172) + (1-0) * math.log(1-0.8172)
r12 = 1 * math.log(0.8648) + (1-1) * math.log(1-0.8648)
r13 = 1 * math.log(0.4122) + (1-1) * math.log(1-0.4122)

r21 = 1 * math.log(0.3266) + (1-1) * math.log(1-0.3266)
r22 = 1 * math.log(0.6902) + (1-1) * math.log(1-0.6902)
r23 = 0 * math.log(0.5620) + (1-0) * math.log(1-0.5620)

r31 = 0 * math.log(0.2024) + (1-0) * math.log(1-0.2024)
r32 = 0 * math.log(0.2884) + (1-0) * math.log(1-0.2884)
r33 = 0 * math.log(0.5554) + (1-0) * math.log(1-0.5554)

BCELOSS = -(1/9) * (r11 + r12+ r13 + r21 + r22 + r23 + r31 + r32 + r33)
```

​		来看看结果叭：

![image-20220723145941661](https://tutouxiaosu.oss-cn-beijing.aliyuncs.com/img/img/image-20220723145941661.png)

​		你会发现调用`BCELOSS`函数和手动计算的结果是一致的，只是精度上有差别，这说明我们前面所说的理论公式是正确的。【注：官方还提供了一种函数——[`BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)，其和`BCELOSS`大致一样，只是对输入的数据不需要再调用Sigmoid函数将其归一化到[0,1]之间，感兴趣的可以阅读看看】

​		这个损失函数讲完训练部分就真没什么可讲的了，哦，这里得提一下，在计算生成器的损失时，我们不是最小化$log(1-D(G(Z)))$ ，而是最大化$logD(G(z))$ 。这个在GAN网络论文中也有提及，我[上一篇](https://juejin.cn/post/7120443914854613029)没有说明这点，这里说声抱歉，论文中说是这样会更好的收敛，这里大家注意一下就好。







![image-20230129182716473](https://tutouxiaosu.oss-cn-beijing.aliyuncs.com/img/img/image-20230129182716473.png)

![image-20230129182659601](https://tutouxiaosu.oss-cn-beijing.aliyuncs.com/img/img/image-20230129182659601.png)





```python 
real_loss = loss_fn(discriminator(gt_images), labels_one)       #log(D(X))
fake_loss = loss_fn(discriminator(fake_images.detach()), labels_zero)    #log(1-D(G(Z)))
d_loss = (real_loss + fake_loss)
```

> gt_images --> x       discriminator(gt_images) --> D(x)        
>
> fake_images --> G(z)    discriminator(fake_images.detach()) --> D(G(z))



```python
g_loss = loss_fn(discriminator(fake_images), labels_one)  #log(D(G(Z)))
```

> fake_images --> G(z)    discriminator(fake_images.detach()) --> D(G(z))