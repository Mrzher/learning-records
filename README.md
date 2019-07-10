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
SSH配置 clone push  
8.vim  
基本命令模式和输入模式    i切换到输入模式    Esc返回基本命令模式  
底线命令模式（在基本命令模式下按":"键）  在最底一行输入命令  保存、查找或者替换一些内容  
:wq 保存离开  
https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649030809&idx=1&sn=512513678a99218392260d3d5763e09a&chksm=8712bee4b06537f2253b469fda709698f90e23bf91387ceea4af313766125ea4b9119c015c58&scene=21#wechat_redirect  
9.g++ makefile cmake  
2019/07/08  
10.  
softmax函数的功能是将各个类别的打分转化成合理的概率值 tf.nn.softmax(tf.matmul(x,W)+b)  
交叉熵损失 cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y))   
tf.reduce_mean计算张量tensor沿着指定的数轴上的的平均值   tf.reduce_sum ：计算tensor指定轴方向上的所有元素的累加和  
softmax和交叉熵结合：tf.nn.softmax_cross_entropy_with_logits(labels=,logits=)  
11.  
tf.app.flags.FLAGS是TensorFlow内部的一个全局变量存储器，同时用于命令行参数的处理  
12.  
tf.train.string_input_producer  
tf.train.start_queue_runners  
2019/07/10  
13._activation_summary()  
TensorBoard显示训练信息  
再另一个命令行窗口：tensorboard --logdir cifar10_train/ (events.out开头文件记录日志信息)  
tensorboard --logdir cifar10_eval/ --port 6007
14. 一个checkpoint文件和一些以model.ckpt开头的文件
15. Tensorflow中读入数据的三种方法：  
（1）用占位符placeholder读入  
（2）用队列的形式建立文件到Tensor的映射  
（3）用Dataset API读入  
16.TFRecords文件的生成  
(1)建立tfrecord存储器  
tf.python_io.TFRecordWriter(path)  
(2)构造每个样本的Example模块  
writer=tf.python_io.TFRecordWriter(filename)  
example=tf.train.Example(features=tf.train.Features(feature={'i':_int64_feature(i),'j':_int64_feature(j)}))  
writer.write(example.SerializeToString())#序列转换成字符串  
#如上读文件与如下写文件对应  
filename_queue=tf.train.string_input_producer(files,shuffle=False) #传入文件名list，系统将其转化为文件名queue  
reader=tf.TFRecordReader()  
_,serialized=reader.read(filename_queue)  
features=tf.parse_single_example(serialized,features={'i':tf.FixedLenFeature([],tf.int64),'j':tf.FixedLenFeature([],tf.int64)}) #tf.TFRecordReader()的parse_single_example()解析器，用于将Example协议内存块解析为张量  
i,j=features['i'],features['j']  
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/  
17.Python中的下划线  
单个下划线_作为名称使用：用作被丢弃的名称  
单下划线前缀的名称（例如_shahriar）：指定了这个名称是“私有的”  
双下划线前缀的名称（例如__shahriar）：Python会改写这些名称，以免与子类中定义的名称产生冲突。  
前后都带有双下划线的名称（例如__init__）：一种确保Python系统中的名称不会跟用户自定义的名称发生冲突的方式  

