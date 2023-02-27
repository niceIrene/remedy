import copy
import time

names = ["Output of get_temp function, all of the attributes for given group"]
compas_y = "column name of y_label"
#####################################

# Optimized Identification Functions

#####################################
# helper function for optimized, counts the number of negative and positive neighbors
def compute_neighbors_opt(group_lst,lst_of_counts, pos, neg):
    times = len(group_lst)
    pos_cnt = 0
    neg_cnt = 0
    for i in range(times):
        df_groupby = lst_of_counts[i]
        temp_group_lst_pos = copy.copy(group_lst)
        temp_group_lst_neg = copy.copy(group_lst)
        del temp_group_lst_pos[i]
        del temp_group_lst_neg[i]
        # count positive
        temp_group_lst_pos.append(1)
        group_tuple_pos = tuple(temp_group_lst_pos)
        if group_tuple_pos in df_groupby.keys():
            pos_cnt += df_groupby[group_tuple_pos]
        else:
            pos_cnt += 0
        # count negative
        temp_group_lst_neg.append(0)
        group_tuple_neg = tuple(temp_group_lst_neg)
        if group_tuple_neg in df_groupby.keys():
            neg_cnt += df_groupby[group_tuple_neg]
        else:
            neg_cnt += 0
    pos_val = pos_cnt - times* pos
    neg_val = neg_cnt - times* neg

    if neg_val == -1 or (neg_val == 0 and pos_val == 0):
        return (pos_val, neg_val, -1)
    if pos_val == 0 or neg_val == 0:
        return (pos_val, neg_val, 0)


    return (pos_val, neg_val, pos_val/neg_val)


#Function to determine based on the neighbors if the group is positive or negative
def determine_problematic_opt(group_lst, names, temp2, lst_of_counts, label, threshold= 0.3):
    #0: ok group, 1: need negative records, 2: need positive records
    d = copy.copy(temp2)
    for i in range(len(group_lst)):
        d = d[d[names[i]] == group_lst[i]]
    total =  d['cnt'].sum()
    d = d[d[label] == 1]
    pos = d['cnt'].sum()
    neg = total - pos
    neighbors = compute_neighbors_opt(group_lst,lst_of_counts, pos, neg)
    if(neighbors[2] == -1):
        # there is no neighbors
        return 0
    if(total > 30):
        # need to be large enough, need to adjust with different datasets.
        if neg == 0:
            if (pos > neighbors[2]):
                return 1
            if(pos <= neighbors[2]):
                return 0
        if (pos/(neg) - neighbors[2] > threshold):
            # too many positive records
            return 1
        if (neighbors[2] - pos/(neg) > threshold):
            return 2
    return 0


#Function to designate if a group is positive or negative
def compute_problematic_opt(temp2, temp_g, names, label, lst_of_counts):
    need_pos = []
    need_neg = []
    for index, row in temp_g.iterrows():
        group_lst = []
        for n in names:
            group_lst.append(row[n])
        problematic = determine_problematic_opt(group_lst, names, temp2, lst_of_counts,label)
        if(problematic == 1):
            if group_lst not in need_neg:
                need_neg.append(group_lst)
        if(problematic == 2):
            if group_lst not in need_pos:
                need_pos.append(group_lst)
    return need_pos, need_neg


#####################################

# Naive Identification Functions

#####################################

# compute the pos/neg ration of this neighbor
def compute_neighbors(group_lst, result):
    # compute the ratio of positive and negative records
    start2 = time.time()
    pos = 0
    neg = 0 
    for r in result:
        total  = r['cnt'].sum()
        r = r[r[compas_y] == 1]
        pos += r['cnt'].sum()
        neg += total - r['cnt'].sum()
    if(neg == 0):
        return (pos, neg, -1)
    end2 = time.time()
    
    return(pos, neg, pos/neg)


# get the list of numbers of the given group
def get_one_degree_neighbors(temp2, names, group_lst):
    result = []
    for i in range(len(group_lst)):
        d = copy.copy(temp2)
        for k in range(len(group_lst)):
            if k != i:
                d = d[d[names[k]] == group_lst[k]]
            else:
                d = d[d[names[k]] != group_lst[k]]
        result.append(d)
    return result

def determine_problematic(group_lst, temp2, result, label, threshold= 0.3):
    # return a value for a given group about whether it is a problematic group
    #0: ok group, 1: need negative records, 2: need positive records
    d = copy.copy(temp2)
    for i in range(len(group_lst)):
        d = d[d[names[i]] == group_lst[i]]
    total =  d['cnt'].sum()
    d = d[d[label] == 1]
    pos = d['cnt'].sum()
    neg = total - pos
    neighbors = compute_neighbors(group_lst, result)
    if(neighbors[2] == -1):
        # there is no neighbors
        return 0
    if(total > 10):
        # need to be large enough
        if (pos/(neg+1) - neighbors[2] > threshold):
            # too many positive records
            return 1
        if (neighbors[2] - pos/(neg+1) > threshold):
            # too many negative records
            return 2
    return 0