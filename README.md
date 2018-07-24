## 一.算法模型
### 1. 无监督的图像翻译算法
域迁移网络学习跨域图像的共享特征空间实现无监督模型。该模型的损失函数比较复杂：公式如下
>```math
>L_G=L_{GAN}+\lambda L_{const}+\beta L_{TID}+\gamma L_{TV}
>```
- 第一项：对抗损失项，促使模型的输出拟合目标域图像的分布
- 第二项：促使模型的输入和输出共享相同的特征空间
- 第三项：正则化项
- 第四项：各向异性总变分损失函数-->目的：平滑输出图像

### 2. 基于GAN的半监督图像翻译算法
标准的GAN在判别模型输出或真实标签时，总会有局部区域不理想，采取的办法是--**多个判别器，取平均损失**

>使用成对数据进行训练模型，非成对数据提升性能。

#### 模型损失
模型根据监督和半监督分为两个部分：
>监督阶段：L1 + 判别损失

>半监督阶段：生成对抗网络的对抗损失
- 利用成对数据集训练模型
>```math
>  D\_{loss}:L_{D_Y}(G,D_Y,x,y)=-[logD_Y(y)+log(1-D_Y(G(x)))]
>
>  G\_{loss}:L_G(G,D_Y,x,y)=-log_{D_Y}(G(x))+\|G(x)-y\|_1
>```
- 利用非成对数据提升性能
>```math
>   D\_loss:L_{D_Y}(G,D_Y,x,y)=-[logD_Y(y)+log(1-D_Y(G(x)))]
>
>   G\_loss:L_G(G,D_Y,x,y)=-log_{D_Y}(G(x))
>```
### 3. 权重共享的多尺度判别生成对抗网络的半监督图像翻译算法
- 利用成对数据集训练模型
>```math
>  D\_{loss}:L_D=\sum_i{L_{D_i}(G,D_i,x,y)}=-\sum_i{[logD_i(y)+log(1-D_i(G(x)))]}
>
>  G\_{loss}:L_G=\sum_i{L_G(G,D_i,x,y)}=-\sum_i{log_{D_i}(G(x))}+\|G(x)-y\|_1
>```
- 利用非成对数据提升性能
>```math
>   D\_loss:L_D=\sum_i{L_{D_i}(G,D_i,x,y)}=-\sum_i{[logD_i(y)+log(1-D_i(G(x)))]}
>
>   G\_loss:L_G=\sum_i{L_G(G,D_i,x,y)}=-\sum_i{log_{D_i}(G(x))}
>```
### 4. 基于对抗和对偶的半监督图像翻译算法
一对映射函数`$G:X \to Y$`和`$F:Y \to X$`对应的判别器`$D_X$`和`$D_Y$`，对映射`$G:X \to Y$`,再通过逆变换`$F'$`得到`$F'(F(X))$`与原图像`$x$`保持一致。
>```math
>   x \to F(x) \Rightarrow F'(F(x)) \thickapprox x
>```
模型的损失函数包括两项：**对抗损失项：**可以促使生成数据的分布拟合目标分布；**循环一致损失项：**保证生成图像保持输入的某些特征及映射的唯一性。
- 监督：利用成对数据集训练模型
>```math
>  D1:D\_{loss}:L_{D_Y}(G,D_Y,x,y)=-[logD_Y(y)+log(1-D_Y(G(x)))]
>
>  D2:D\_{loss}:L_{D_X}(G,D_X,x,y)=-[logD_X(x)+log(1-D_X(G(y)))]
>
>  G1:G\_{loss}:L_G(G,D_Y,x,y)=-log_{D_Y}(G(x))+\|G(x)-y\|_1
>
>  G2:G\_{loss}:L_G(G,D_X,x,y)=-log_{D_X}(G(y))+\|F(y)-x\|_1
>```
- 半监督：利用非成对数据提升性能
>```math
>   D1:D\_{loss}:L_{D_Y}(G,D_Y,x,y)=-[logD_Y(y)+log(1-D_Y(G(x)))]
>
>   D2:D\_{loss}:L_{D_X}(G,D_X,x,y)=-[logD_X(x)+log(1-D_X(G(y)))]
>
>   G1:Consistency: L_{cons}(G,F,x,y)=\|F(G(x))-x\|_1+\|G(F(y))-y\|_1
>
>   G2:D\_loss:L_G(G,D_Y,x,y)=-log_{D_Y}(G(x))
>
>   G3:G\_loss:L_F(G,D_x,x,y)=-log_{D_X}(F(y))
>```

## 二. 网络架构
因为原始的网络架构采取**原始的编码-解码架构**，该结构编码得到高级特征，但体现原始图像细节的低级特征被丢失。

![image](https://raw.githubusercontent.com/liurio/deep_learning/master/img/%E7%94%9F%E6%88%90%E5%99%A8%E5%8E%9F%E7%90%86.png)

因此采用新的架构模式-**U-Net**, 其在第i层和第n-i层增加了跳跃连接，这样使低层特征得到充分利用。

![image](https://raw.githubusercontent.com/liurio/deep_learning/master/img/%E7%94%9F%E6%88%90%E5%99%A8U-Net.png)

**注意：判别器采用全卷积网络，没有全连接层和池化层，但是批一化，使用LeakyReLU激活函数，使用ADam优化器**
## 三. 实验
- 数据集：City scapes数据集包含50个不同城市，CMP facades数据集，对于每个数据集分为四个部分：20%用于监督训练，60%用于无监督训练，10%用于验证，10%用于测试；
- 不同算法：CGAN、L1/GAN、MSD-CGAN、L1/MSG-GAN、pix2pix(L1+CGAN)、Dual-MSD-CGAN、L1/Dual-MSD-GAN

![image](https://raw.githubusercontent.com/liurio/deep_learning/master/img/MSD-GAN-result.JPG)

![image](https://raw.githubusercontent.com/liurio/deep_learning/master/img/Dual-MSD-GAN-result.JPG)