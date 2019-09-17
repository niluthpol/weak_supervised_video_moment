from __future__ import print_function
import os
import pickle

import numpy
from data_charades import get_test_loader
import time
import numpy as np
from vocab import Vocabulary 
import torch
from model_charades import VSE, order_sim
from collections import OrderedDict
import pandas


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    #attn_weights =
    for i, (images, captions, lengths, lengths_img, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, attn_weight_s = model.forward_emb(images, captions, lengths, lengths_img, volatile=True)
		
        if(attn_weight_s.size(1)<10):
            attn_weight=torch.zeros(attn_weight_s.size(0),10,attn_weight_s.size(2))
            attn_weight[:,0:attn_weight_s.size(1),:]=attn_weight_s
        else:
            attn_weight=attn_weight_s

        batch_length=attn_weight.size(0)
        attn_weight=torch.squeeze(attn_weight)
        
        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            attention_index = np.zeros((len(data_loader.dataset), 10))
            rank1_ind = np.zeros((len(data_loader.dataset)))
            lengths_all = np.zeros((len(data_loader.dataset)))
			
        
        attn_index= np.zeros((batch_length, 10)) # Rank 1 to 10
        rank_att1= np.zeros(batch_length)
        temp=attn_weight.data.cpu().numpy().copy()
        for k in range(batch_length):
            att_weight=temp[k,:]
            sc_ind=numpy.argsort(-att_weight)
            rank_att1[k]=sc_ind[0]
            attn_index[k,:]=sc_ind[0:10]
	
        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
        attention_index[ids] = attn_index
        lengths_all[ids] = lengths_img
        rank1_ind[ids] = rank_att1

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs, attention_index, lengths_all

	
#def cIoU_old(a,b,prec):
#    return np.around(1.0*(min(a[1], b[1])-max(a[0], b[0]))/(max(a[1], b[1])-min(a[0], b[0])),decimals=prec)
	
	
def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union

def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    if data_path is not None:
        opt.data_path = data_path
    opt.vocab_path = "./vocab/"
    # load vocabulary	   
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, 'vocab.pkl'), 'rb'))
        
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)
    
    # load model state
    model.load_state_dict(checkpoint['model'])
    print(opt)	
	
    ####### input video files
    path= os.path.join(opt.data_path, opt.data_name)+"/Caption/charades_"+ str(split) + ".csv"
    df=pandas.read_csv(open(path,'rb'))
    #columns=df.columns
    inds=df['video']
    desc=df['description']

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs, attn_index, lengths_img = encode_data(model, data_loader)

    print(img_embs.shape)
    print(cap_embs.shape)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0], cap_embs.shape[0]))

	# retrieve moments
    r13, r15, r17 = t2i(img_embs, cap_embs, df, attn_index, lengths_img, measure=opt.measure, return_ranks=True)
		
def t2i(images, captions, df, attn_index, lengths_img, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, K) matrix of images
    Captions: (N, K) matrix of captions
    """
    inds=df['video']
    desc=df['description']
    start_segment=df['start_segment']
    end_segment=df['end_segment']
	
    if npts is None:
        npts = images.shape[0]
    ims = numpy.array([images[i] for i in range(0, len(images), 1)])

    ranks = numpy.zeros(int(npts))
    top1 = numpy.zeros(int(npts))
    average_ranks = []
    average_iou = []
    correct_num05=0
    correct_num07=0
    correct_num03=0

    R5IOU5=0
    R5IOU7=0
    R5IOU3=0
    R10IOU3=0
    R10IOU5=0
    R10IOU7=0
	
    for index in range(int(npts)):
        att_inds=attn_index[index,:]
        len_img=lengths_img[index]
        gt_start=start_segment[index]
        gt_end=end_segment[index]
        break_128=np.floor(len_img*2/3)-1
        rank1_start=att_inds[0]
        if (rank1_start<break_128):
           rank1_start_seg =rank1_start*128
           rank1_end_seg = rank1_start_seg+128
        else:
           rank1_start_seg =(rank1_start-break_128)*256
           rank1_end_seg = rank1_start_seg+256
			
        iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
        if iou>=0.5:
           correct_num05+=1
        if iou>=0.7:
           correct_num07+=1
        if iou>=0.3:
           correct_num03+=1
		   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.5:
               R5IOU5+=1
               break
			   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.7:
               R5IOU7+=1
               break
			   
        for j1 in range(5):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.3:
               R5IOU3+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.5:
               R10IOU5+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.7:
               R10IOU7+=1
               break
			   
        for j1 in range(10):
            if (att_inds[j1]<break_128):
               rank1_start_seg =att_inds[j1]*128
               rank1_end_seg = rank1_start_seg+128
            else:
               rank1_start_seg =(att_inds[j1]-break_128)*256
               rank1_end_seg = rank1_start_seg+256
			   
            iou = cIoU((gt_start, gt_end),(rank1_start_seg, rank1_end_seg))
            if iou>=0.3:
               R10IOU3+=1
               break    
			    	
			   
	
	
	############################

    # Compute metrics
    R1IoU05=correct_num05
    R1IoU07=correct_num07
    R1IoU03=correct_num03
    total_length=images.shape[0]
    #print('total length',total_length)
    print("R@1 IoU0.3: %f" %(R1IoU03/float(total_length)))
    print("R@5 IoU0.3: %f" %(R5IOU3/float(total_length)))
    print("R@10 IoU0.3: %f" %(R10IOU3/float(total_length)))
	
    print("R@1 IoU0.5: %f" %(R1IoU05/float(total_length)))
    print("R@5 IoU0.5: %f" %(R5IOU5/float(total_length)))
    print("R@10 IoU0.5: %f" %(R10IOU5/float(total_length)))
	
    print("R@1 IoU0.7: %f" %(R1IoU07/float(total_length)))
    print("R@5 IoU0.7: %f" %(R5IOU7/float(total_length)))
    print("R@10 IoU0.7: %f" %(R10IOU7/float(total_length)))
	
	
    return R1IoU03, R1IoU05, R1IoU07
		
		