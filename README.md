## tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）

除去name参数用以指定该操作的name，与方法有关的一共四个参数：                  
 - value：指需要做卷积的输入图像，要求是一个4维Tensor，具有[batch, height, width, channels]这样的shape
具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数] 
 - filters：相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, channels, out_channels]这样的shape
具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，同理这里第三维channels，就是参数value的第四维
 - rate：要求是一个int型的正数，正常的卷积操作应该会有stride（即卷积核的滑动步长），但是空洞卷积是没有stride参数的，默认是1，这一点尤其要注意。
取而代之，它使用了新的rate参数，那么rate参数有什么用呢？它定义为我们在输入图像上卷积时的采样间隔，
你可以理解为卷积核当中穿插了（rate-1）数量的“0”，把原来的卷积核插出了很多“洞洞”，这样做卷积时就相当于对原图像的采样间隔变大了。
具体怎么插得，可以看后面更加详细的描述。此时我们很容易得出rate=1时，就没有0插入，此时这个函数就变成了普通卷积。  
 - padding：string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同边缘填充方式。

ok，完了，到这就没有参数了，或许有的小伙伴会问那“stride”参数呢。其实这个函数已经默认了stride=1，也就是滑动步长无法改变，固定为1。
结果返回一个Tensor，填充方式为“VALID”时，返回[batch,height-2*(filter_width-1),width-2*(filter_height-1),out_channels]的Tensor
填充方式为“SAME”时，返回[batch, height, width, out_channels]的Tensor  

atrous_conv2d详解：https://blog.csdn.net/mao_xiao_feng/article/details/77924003
在不使用池化层的情况下增大感受视野：https://blog.csdn.net/guvcolie/article/details/77884530?locationNum=10&fps=1