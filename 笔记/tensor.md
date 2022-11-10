##### 张量维度

1. tensor.size()
2. tensor.shape

##### 合并张量

###### 1. torch.cat(tensor, dim)

将tensor在dim维度上进行合并，除dim维之外，其他的维度必须相同

###### 2. tensor.squeeze()

去掉长度为1的维

##### 访问tensor中的元素

需要通过int(a\[i][j])将元素转化为int，否则为tensor类
