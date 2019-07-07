# learning-records
2019/07/07
1.位置参数 默认参数 *可变参数 **关键字参数
2.NumPy 
数组 np.array() 列表 元组 
np.arange() np.linspace()
np.zeros() :生成元素全是0的数组 np.ones()：生成元素全是1的数组 np.zeros_like(a):生成形状和a一样且元素全是0的数组 np.ones_like(a):生成形状和a一样且元素全是1的数组
随机数组 np.random.rand(2,2)   b = np.random.randn(2,2)   c = np.random.randint(0,9,(2,2))
a.shape   a.dtype   a.ndim
存取：切片法和整数列表
数组形状变换reshape() -1代表自动生成
数组维度交换swapaxes()
多维变1维reshape(-1)、flatten()和ravel()
堆叠数组:hstack()和vstack()函数
3.matplotlib库pyplot模块绘图
figure()调出一个画布  
用plot()在画布上画图 
xlabel,ylabel：分别设置X，Y轴的标题文字
title：设置标题
xlim,ylim：分别设置X,Y轴的显示范围
legend:显示图例
4.matlibplot读取图像
imread和imshow()
5.Linux基础命令
cd / 表示切换到根目录
ls
cat 读取文件内容及拼接文件
rm <文件> 或 rm -r <文件夹>
mkdir
cp <文件> <目标文件> 或者cp -r<文件夹><目标文件夹>
kill PID码 ps au查看进程
pwd
tar 文件压缩
unzip 文件解压
6.shell脚本文件
$用来获取变量值
7.git

