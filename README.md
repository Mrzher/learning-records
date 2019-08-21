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
model.ckpt.meta文件保存了TensorFlow计算图的结构  
model.ckpt.index是对应模型的索引文件  
model.ckpt.data-00000-of-00001文件保存了TensorFlow程序中每一个变量的取值  
15. Tensorflow中读入数据的三种方法：  
（1）用占位符placeholder读入  
（2）用队列的形式建立文件到Tensor的映射  
（3）用Dataset API读入  
16.TFRecords文件的生成与读取  
TFRecords文件包含了tf.train.Example 协议内存块(protocol buffer)(协议内存块包含了字段 Features)。我们可以写一段代码获取你的数据，将数据填入到Example协议内存块(protocol buffer)，将协议内存块序列化为一个字符串，并且通过tf.python_io.TFRecordWriter 写入到TFRecords文件。  
从TFRecords文件中读取数据， 可以使用tf.TFRecordReader的tf.parse_single_example解析器。这个操作可以将Example协议内存块(protocol buffer)解析为张量。  
(1)建立tfrecord存储器  
writer=tf.python_io.TFRecordWriter(filename)  
(2)构造每个样本的Example模块  
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
2019/07/11  
18.tmux  
https://blog.csdn.net/lihao21/article/details/68958515  
https://segmentfault.com/a/1190000016283278  
19.tf.decode_raw()  
tf.decode_raw函数的意思是将原来编码为字符串类型的变量重新变回来  
20.  
协调器 tf.train.Coordinator  
入队线程启动器 tf.train.start_queue_runners  
2019/08/19  
ssh cuizhe@hass.top -p 7715        1234  
wget下载anaconda安装包  
安装anaconda  bash Anaconda3-5.2.0-Linux-x86_64.sh  
创建py36torch41环境  
conda activate py36torch41  
conda init  
2019/08/20  
虚拟环境建立好了之后使用如下语句激活虚拟环境source activate py36torch41,如果需要退出该虚拟环境，可以使用如下命令source deactivate  
cuda版本10.0.130  
pytorch官网获取安装指令  
py36torch41环境下conda install pytorch=0.4.1 torchvision cudatoolkit=10.0 -c pytorch  
CUDA看作是一个工作台 cuDNN是基于CUDA的深度学习GPU加速库 想要在CUDA上运行深度神经网络，就要安装cuDNN  
只要把cuDNN文件复制到CUDA的对应文件夹里就可以，即是所谓插入式设计，把cuDNN数据库添加CUDA里，cuDNN是CUDA的扩展计算库，不会对CUDA造成其他影响。  
运行poolnet  
conda install opencv  
2019/08/21  
运行poolnet时遇到第一个错误，dataset.py中line27 line28使用os.path.join第二个参数的首个字符如果是"/" , 拼接出来的路径会不包含第一个参数  
linux下如何删除整个文件夹  
运行basnet  
Type 给出是在GPU中使用的是计算（用C代表）还是图形图像处理（用G代表）  
