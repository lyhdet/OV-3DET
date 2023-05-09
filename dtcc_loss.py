from models.DETR.matcher import build_matcher
import torch.nn.functional as F
import torch
from torch import logit, nn
from criterion import Matcher
from utils.dist import all_reduce_average
from utils.box_util import generalized_box3d_iou
import random

class_prob_statistic = {}
class_prob_num = {}


def cal_sim(z_i,z_j,temperature):
    z_i = z_i / z_i.norm(dim=len(z_i.shape)-1, keepdim=True)
    z_j = z_j / z_j.norm(dim=len(z_j.shape)-1, keepdim=True)
    return z_i @ z_j.t() / temperature

def DTCC_loss(objs, temperature=0.1):
    device = objs.device
    dtcc_loss = torch.tensor(0,device=device,dtype=torch.float)
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device=device)
    
    valid_obj_cnt = 1
    for obj_ind in range(objs.shape[0]):
    	obj = objs[obj_ind,:]
    	#logits = []
    	obj_feature = obj[:-2]
    	obj_cls = obj[-2]
    	obj_score = obj[-1]
    	neg_objs_inds = torch.where(objs[:,-2]!=obj_cls)[0]
    	
    	if len(neg_objs_inds) > 0:
    		neg_objs = objs[neg_objs_inds,:]
    		neg_loss = cal_sim(obj_feature,neg_objs[:,:-2],temperature).unsqueeze(0)
    	else:
    		continue
    	
    	pos_objs_inds = torch.where(objs[:,-2]==obj_cls)[0]
    	pos_objs_inds = [i for i in pos_objs_inds if i!= obj_ind]		# remove itself
    	
    	if len(pos_objs_inds) > 0:
    		pos_loss = cal_sim(obj_feature,objs[pos_objs_inds,:-2],temperature).unsqueeze(0).t()
    	else:
    		pos_loss = torch.tensor([1/temperature],device=device,dtype=torch.float).unsqueeze(0)
    		valid_obj_cnt -= 1

    	'''
    	print(neg_loss)
    	print(torch.mean(neg_loss))
    	print("\n")
    	print(pos_loss)
    	valid_pos = torch.where(pos_loss<5)[0]
    	print(pos_loss[valid_pos])
    	print(torch.mean(pos_loss[valid_pos]))

    	print(objs[img_obj_num:, -2])
    	input()
        '''
    	
    	
    	
    	logits = torch.cat([pos_loss,neg_loss.repeat(pos_loss.shape[0],1)],dim=1)
    	labels = torch.zeros(logits.shape[0], device=device,dtype=torch.long)
    	
    	cur_loss = criterion(logits,labels)
    	#print(cur_loss)
    	dtcc_loss += cur_loss
    	valid_obj_cnt += 1
    
    dtcc_loss /= valid_obj_cnt
    
    #print(dtcc_loss)
    #input()
    return dtcc_loss
    
def dtcc_pc_img_text(pair_img_output, pc_output, text_output):
    assert torch.sum(pair_img_output["pair_img_label"] - pc_output["pc_label"]) == 0
    assert torch.sum(pair_img_output["pair_img_prob"] - pc_output["pc_prob"]) == 0
    
    
    # # Stastic confidence of each class
    # class_label = pair_img_output["pair_img_label"].detach().cpu().numpy()
    # class_prob = pair_img_output["pair_img_prob"].detach().cpu().numpy()

    # for ind, (cur_label, cur_prob) in enumerate(zip(class_label, class_prob)):
    #     #print(cur_label, " ,", cur_prob)
    #     if cur_label not in class_prob_statistic.keys():
    #         class_prob_statistic[cur_label] = cur_prob

    #     if cur_label not in class_prob_num.keys():
    #         class_prob_num[cur_label] = 0

    #     class_prob_num[cur_label] += 1
    #     class_prob_statistic[cur_label] = ( min(class_prob_num[cur_label], 99)*class_prob_statistic[cur_label] + cur_prob ) / ( min(class_prob_num[cur_label], 99) + 1 )
        

    # min_valid_prob = torch.zeros_like(pair_img_output["pair_img_prob"])
    # for ind in range(min_valid_prob.shape[0]):
    #     min_valid_prob[ind] = class_prob_statistic[class_label[ind]]


    #prob_threshold = 0.1
    class_num = 365
    #min_obj_num = 2
    # print(pair_img_output["pair_img_label"].detach().cpu().numpy())

    #prepare pair image branch
    pair_img_feat = pair_img_output["pair_img_feat"]
    pair_img_label = pair_img_output["pair_img_label"]
    pair_img_prob = pair_img_output["pair_img_prob"]
    pair_img_label = torch.unsqueeze(pair_img_label, dim=1)
    pair_img_prob = torch.unsqueeze(pair_img_prob, dim=1)
    pair_img_objs = torch.cat([pair_img_feat, pair_img_label, pair_img_prob], dim=1)

    # only keep valid class and prob > average
    #valid_ind = torch.where( (pair_img_objs[:, -1] > min_valid_prob) & (pair_img_objs[:, -2] < class_num))[0]
    valid_ind = torch.where(pair_img_objs[:, -2] < class_num)[0]
    pair_img_objs = pair_img_objs[valid_ind]

    
    '''
    if pair_img_objs.shape[0] > min_obj_num:
        sorted_prob, prob_sort_ind = torch.sort(pair_img_objs[:, -1], descending=True)
        valid_cnt = torch.where(sorted_prob>prob_threshold)[0].shape[0]
        if valid_cnt <= min_obj_num:
            valid_cnt = min_obj_num

        pair_img_objs = pair_img_objs[prob_sort_ind[:valid_cnt], :]
    '''

    #prepare pc branch
    pc_feat = pc_output["pc_feat"]
    pc_label = pc_output["pc_label"]
    pc_prob = pc_output["pc_prob"]
    pc_label = torch.unsqueeze(pc_label, dim=1)
    pc_prob = torch.unsqueeze(pc_prob, dim=1)
    pc_objs = torch.cat([pc_feat, pc_label, pc_prob], dim=1)

    # valid_ind = torch.where( (pc_objs[:, -1] > min_valid_prob) & (pc_objs[:, -2] < class_num))[0]
    valid_ind = torch.where(pc_objs[:, -2] < class_num)[0]
    pc_objs = pc_objs[valid_ind]


    #prepare text branch
    text_feat = text_output["text_feat"]
    text_label = text_output["text_label"]
    text_label = torch.unsqueeze(text_label, dim=1)
    text_prob = torch.ones_like(text_label)
    text_objs = torch.cat([text_feat, text_label, text_prob], dim=1)

    #print(text_objs)
    #print(text_objs.shape)
    #input()

    #print(text_objs.shape)
    #print(pair_img_objs.shape)
    print(pc_objs.shape)

    unique_text_cls = torch.unique(pc_objs[:,-2].detach()).long()
    unique_text_objs = text_objs[unique_text_cls,:]

    rand_text_cls = random.sample(range(0,class_num), 20)
    rand_text_objs = text_objs[rand_text_cls,:]
    text_objs = torch.cat([unique_text_objs, rand_text_objs], dim=0)

    # print(unique_text_objs.shape)
    # print(rand_text_objs.shape)
    # print(text_objs.shape)
    # exit()
    
    dtcc_group_1 = torch.cat([text_objs, pc_objs], dim=0)
    dtcc_group_2 = torch.cat([pc_objs, pair_img_objs], dim=0)
    

    #print(dtcc_group_1.shape)
    #print(dtcc_group_2.shape)
    #print(dtcc_group_3.shape)
    
    dtcc_loss_1 = DTCC_loss(dtcc_group_1)
    dtcc_loss_2 = DTCC_loss(dtcc_group_2)
    # dtcc_loss_3 = DTCC_loss(dtcc_group_3)

    return dtcc_loss_1, dtcc_loss_2
    