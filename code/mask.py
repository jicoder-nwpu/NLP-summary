# mask 机制的简单实现
from lib import *


# batch = next(iter(train_iter))
# input_seq = batch.English.transpose(0, 1)
# input_pad = EN_TEXT.vocab.stoi('<pad>')

# # 把pad位置 mask
# input_msk = (input_seq != input_pad).unsqueeze(1)

# target_seq = batch.French.transpose(0, 1)
# target_pad = FR_TEXT.vocab.stoi('<pad>')
# target_msk = (target_seq != target_pad).unsqueeze(1)

# size = target_seq.size(1)

size = 7
nopeak_mask = np.triu(np.ones((1, size, size)), k = 1).astype('uint8')   #triu 返回数组的上三角部分，其他部分设置为0， k表示上三角的边界
nopeak_mask = torch.tensor(nopeak_mask == 0)
print(nopeak_mask)
# target_msk = target_msk & nopeak_mask
