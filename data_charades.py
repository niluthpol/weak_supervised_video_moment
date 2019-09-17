import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import pandas
import scipy.io as sio
import skimage.measure as scikit



class Charades(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, dpath, vocab):
        self.vocab=vocab
        path=dpath+"/Caption/charades_"+ str(data_split) + ".csv"
        df=pandas.read_csv(path)
        df_temp = pandas.read_csv(path,dtype={'ID': object})
        self.inds = df_temp['video']
        self.desc=df['description']
        self.data_split=data_split
        self.data_path=dpath

    def __getitem__(self, index):

        img_id = index
        inds=self.inds
        desc=self.desc

        video_feat_file=self.data_path+"/c3d_features/"+str(inds[index])+".mat"
        video_feat_mat = sio.loadmat(video_feat_file)
        video_feat=video_feat_mat['feature']
        # 128 frame features
        video_feat1=scikit.block_reduce(video_feat, block_size=(8, 1), func=np.mean)
        # 256 frame features
        video_feat2=scikit.block_reduce(video_feat, block_size=(16, 1), func=np.mean)
        # concatenation of all 128 frame feature and 256 frame feature
        video_feat=np.concatenate((video_feat1,video_feat2),axis=0)  
            
        image = torch.Tensor(video_feat)
        caption = desc[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.desc)

				
def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.

    Returns:
        images: torch tensor of shape (batch_size, feature_size).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    lengths_img = [len(im) for im in images]
    
    target_images = torch.zeros(len(images), max(lengths_img), 4096)

    #images = torch.stack(images, 0)
    for i, im in enumerate(images):
        end = lengths_img[i]
        target_images[i, :end,] = im[:end,]

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return target_images, targets, lengths, lengths_img, ids



def get_charades_loader(data_split, dpath, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = Charades(data_split, dpath, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader



def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_charades_loader('train', dpath, vocab, opt,
                                          batch_size, True, workers)
    val_loader = get_charades_loader('val', dpath, vocab, opt,
                                        batch_size, False, workers)


    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_charades_loader(split_name, dpath, vocab, opt,
                                         batch_size, False, workers)
										 

    return test_loader

	
