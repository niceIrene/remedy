import pandas as pd
import copy
from sklearn.naive_bayes import MultinomialNB
from sympy import Symbol
from sympy.solvers import solve
import time


columns_all = ["List of all of the columns in the dataset except y_label"]
compas_y = "column name of y_label"
columns_compas = ["List of all of the protected attributes in the dataset"]
names = ["Output of get_temp function, all of the attributes for given group"]
temp2 = ["Output of get_temp function, sum count by group"]

#####################################

#  Compute_diff functions
# Helper functions for sampling algorithms

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

def compute_diff_add_and_remove(group_lst, temp2, need_positive_or_negative, label, names):
    d = copy.copy(temp2)
    for i in range(len(group_lst)):
        d = d[d[names[i]] == group_lst[i]]
    total =  d['cnt'].sum()
    # Total here was 0: here, errors when this is commented out
    if total == 0:
      return -1
    d = d[d[label] == 1]
    pos = d['cnt'].sum()
    neg = total - pos
    result = get_one_degree_neighbors(temp2,names, group_lst)
    neighbors = compute_neighbors(group_lst, result)
    if(need_positive_or_negative == 1):
        # need pos
        x = Symbol('x')
        try:
          diff = solve((pos + x)/ (neg - x) - neighbors[2])[0]
        except:
          return -1     
    else:
        #need negative
        x = Symbol('x')
        try:
          diff = solve((pos - x)/ (neg + x) - neighbors[2])[0]
        except:
          return -1
    return diff

def compute_diff_add(group_lst, temp2, names, label_y, need_positive_or_negative):

    d = copy.copy(temp2)

    for i in range(len(group_lst)):

        d = d[d[names[i]] == group_lst[i]]
    total =  d['cnt'].sum()
    d = d[d[label_y] == 1]
    pos = d['cnt'].sum()
    neg = total - pos
    result = get_one_degree_neighbors(temp2, names, group_lst)
    neighbors = compute_neighbors(group_lst, result)
    if(need_positive_or_negative == 1):
        # need pos

        x = Symbol('x')
        try:
          diff = solve((pos + x)/ neg -  neighbors[2])[0]
        except:
          return -1

        print(neighbors[2], pos, neg, diff)
    else:
        #need negative
        x = Symbol('x')
        try:
          diff = solve(pos/ (neg + x) -  neighbors[2])[0]
        except:
          return -1
    print(neighbors[2], pos, neg, diff)
    return diff

def compute_diff_remove(group_lst, temp2, names, label_y, need_positive_or_negative):
    d = copy.copy(temp2)
    for i in range(len(group_lst)):

        d = d[d[names[i]] == group_lst[i]]
    total =  d['cnt'].sum()
    d = d[d[label_y] == 1]
    pos = d['cnt'].sum()
    neg = total - pos
    result = get_one_degree_neighbors(temp2, names, group_lst)
    neighbors = compute_neighbors(group_lst, result)
    if(need_positive_or_negative == 1):
        # need pos, remove some neg
        x = Symbol('x')
        try:
          diff = solve( pos/ (neg - x) -  neighbors[2])[0]
        except:
          return -1

        print(neighbors[2], pos, neg, diff)
    else:
        #need negative
        x = Symbol('x')
        try:
          diff = solve((pos -x )/ neg -  neighbors[2])[0]
        except:
          return -1
        print(neighbors[2], pos, neg, diff)
    return diff


#####################################

#  Preferential Sampling Algorithm

#####################################
def pref_sampling_opt(train_set, cols_given, label, need_pos, need_neg):
    if len(need_pos)+ len(need_neg) > 0:
        temp_train_x = pd.DataFrame(train_set, columns = columns_all)
        temp_train_label = pd.DataFrame(train_set, columns = [label])
        temp_train_label = temp_train_label[label]
        temp_train_label = temp_train_label.astype('int')
        mnb = MultinomialNB()
        mnb = mnb.fit(temp_train_x, temp_train_label)
        probs = mnb.predict_proba(temp_train_x)[:,0]
        train_set["prob"] = abs(probs - 0.5)
        # get the set of 
    new_train_set = pd.DataFrame(columns = list(train_set.columns))
    updated_pos = 0
    for i in need_pos:
        # needs to updated more positive records
        temp_df = copy.deepcopy(train_set)
        for n in range(len(i)):
          temp_df = temp_df[temp_df[cols_given[n]] == i[n]]
        # update the skew and diff
        idx = list(temp_df.index)
        train_set.loc[idx, 'skewed'] = 1
        idx_pos = list(temp_df[(getattr(temp_df, label) == 1)].index)
        if(len(idx_pos) == 0):
          # if there is no positive
          idx_neg = list(temp_df[(getattr(temp_df, label) == 0)].index)
          neg_ranked = train_set.loc[idx_neg].sort_values(by="prob", ascending=True)
          new_train_set = pd.concat([new_train_set, neg_ranked], ignore_index=True)
          continue
        idx_neg = list(temp_df[(getattr(temp_df, label) == 0)].index)
        pos_ranked = train_set.loc[idx_pos].sort_values(by="prob", ascending=True)
        neg_ranked = train_set.loc[idx_neg].sort_values(by="prob", ascending=True)
        diff = compute_diff_add_and_remove(i, temp2,  1, compas_y, names)
        if diff == -1:
          new_train_set = pd.concat([new_train_set, pos_ranked], ignore_index=True)
          new_train_set = pd.concat([new_train_set, neg_ranked], ignore_index=True)
          continue
        train_set.loc[idx, 'diff'] = int(diff)
        cnt = int(train_set.loc[idx_pos[0]]["diff"])
        updated_pos += cnt * 2 
        # add more records when there are not enough available records
        new_train_set = pd.concat([new_train_set, pos_ranked], ignore_index=True)
        temp_cnt = cnt
        if len(pos_ranked) >= temp_cnt:
            new_train_set = pd.concat([new_train_set,pos_ranked[0:cnt]], ignore_index=True)
        else:
            while(temp_cnt > 0 ):
                new_train_set = pd.concat([new_train_set,pos_ranked[0:temp_cnt]], ignore_index=True) 
            # duplicate the dataframe
                temp_cnt = temp_cnt - len(pos_ranked)
        # duplicate the top cnt records from the pos
        # remove the top cnt records from the neg
        if cnt == 0:
          new_train_set = pd.concat([new_train_set, neg_ranked], ignore_index=True)
        else:
          new_train_set = pd.concat([new_train_set, neg_ranked[cnt-1:-1]], ignore_index=True)
    print("updated {} positive records".format(str(updated_pos)))

    updated_neg = 0
    # adding more records to the need_neg set
    for i in need_neg:
        # list of idx belongs to this group
        temp_df = copy.deepcopy(train_set)
        for n in range(len(i)):
          temp_df = temp_df[temp_df[cols_given[n]] == i[n]]
        # update the skew and diff
        idx = list(temp_df.index)
        train_set.loc[idx, 'skewed'] = 1
        idx_pos = list(temp_df[(getattr(temp_df, label) == 1)].index)
        idx_neg = list(temp_df[(getattr(temp_df, label) == 0)].index)
        if(len(idx_neg) == 0):
          pos_ranked = train_set.loc[idx_pos].sort_values(by="prob", ascending=True)
          new_train_set = pd.concat([new_train_set, pos_ranked], ignore_index=True)
          continue
        pos_ranked = train_set.loc[idx_pos].sort_values(by="prob", ascending=True)
        neg_ranked = train_set.loc[idx_neg].sort_values(by="prob", ascending=True)
        diff = compute_diff_add_and_remove(i, temp2, 0, compas_y, names)
        if diff == -1:
          new_train_set = pd.concat([new_train_set, neg_ranked], ignore_index=True)
          new_train_set = pd.concat([new_train_set, pos_ranked], ignore_index=True)
          continue
        train_set.loc[idx, 'diff'] = int(diff)
        cnt = int(train_set.loc[idx_pos[0]]["diff"])
        updated_neg += cnt * 2 
        # add more records when there are not enough available records
        new_train_set = pd.concat([new_train_set, neg_ranked], ignore_index=True)
        temp_cnt = cnt
        if len(neg_ranked) >= temp_cnt:
            new_train_set = pd.concat([new_train_set,neg_ranked[0:cnt]], ignore_index=True)
        else:
            while(temp_cnt > 0 ):
                new_train_set = pd.concat([new_train_set,neg_ranked[0:temp_cnt]], ignore_index=True) 
            # duplicate the dataframe
                temp_cnt = temp_cnt - len(neg_ranked)
        # duplicate the top cnt records from the pos
        # remove the top cnt records from the neg
        if cnt ==0:
          new_train_set = pd.concat([new_train_set, pos_ranked], ignore_index=True)       
        else:
          new_train_set = pd.concat([new_train_set, pos_ranked[cnt-1:-1]], ignore_index=True)
            
    print("updated {} negative records".format(str(updated_neg)))
    # add the other irrelavant items:
    idx_irr = list(train_set[train_set['skewed'] == 0].index)
    irr_df = train_set.loc[idx_irr]
    new_train_set = pd.concat([new_train_set, irr_df], ignore_index=True)
    print("The new dataset contains {} rows.".format(str(len(new_train_set))))
    new_train_set.reset_index()
    return new_train_set


#####################################

#  Duplication/Oversampling Algorithm

#####################################
def round_int(x):
    if x in [float("-inf"),float("inf")]: return 0
    return int(round(x))
    

def make_duplicate(d, group_lst, diff, label_y, names, need_positive_or_negative):

    selected = copy.deepcopy(d)
    for i in range(len(group_lst)):
        att_name = names[i]
        selected = selected[(selected[att_name] == group_lst[i])]
    selected = selected[(selected[label_y] == need_positive_or_negative)]

    if len(selected) == 0:
        return pd.DataFrame()

    while(len(selected) < diff):
        # duplicate the dataframe
        select_copy = selected.copy(deep=True)
        selected = pd.concat([selected, select_copy])
        # the number needed is more than the not needed numbers.
  
    generated = selected.sample(n = diff, replace = False, axis = 0)
    return generated 


def naive_duplicate(d, temp2, names, need_pos, need_neg, label_y):
    # add more records for all groups
    # The smote algorithm to boost the coverage
    for r in need_pos:
    # add more positive records
        # determine how many points to add
        diff = compute_diff_add(r, temp2, names, label_y, 1)
        if diff == -1:
          continue
        diff = round_int(diff)
        # add more records
        print("Adding " + str(diff) +" positive records")
        samples_to_add = make_duplicate(d, r, diff, label_y, names, need_positive_or_negative = 1)
        d = pd.concat([d, samples_to_add], ignore_index=True) 
    for k in need_neg:
        diff = compute_diff_add(k, temp2, names, label_y, need_positive_or_negative = 0)
        if diff == -1:
          continue
        diff = round_int(diff)
        print("Adding " + str(diff) +" negative records")
        samples_to_add = make_duplicate(d, k, diff, label_y, names, need_positive_or_negative = 0)
        d = pd.concat([d, samples_to_add], ignore_index=True)
    return d

########################################

#  Downsampling/Undersampling Algorithm

########################################

def make_remove(d, group_lst, diff, names, label_y, need_positive_or_negative):

    temp = copy.deepcopy(d)
    for i in range(len(group_lst)):
        att_name = names[i]
        temp = temp[(temp[att_name] == group_lst[i])]
    temp = temp[(temp[label_y] == need_positive_or_negative)]
    # randomly generated diff samples
        # the number needed is more than the not needed numbers.

    if(diff>len(temp)):
        diff = len(temp)
    generated = temp.sample(n = diff, replace = False, axis = 0)
    return generated.index


def naive_downsampling(d, temp2, names, need_pos, need_neg, label_y):
    # add more records for all groups
    # The smote algorithm to boost the coverage
    for r in need_pos:
    # add more positive records
        # determine how many points to add
        diff = compute_diff_remove(r, temp2, names, label_y, need_positive_or_negative = 1)
        if diff == -1:
          continue
        diff = round_int(diff)
        # add more records
        print("Removed " + str(diff) +" negative records")
        samples_to_remove = make_remove(d, r, diff, names, label_y, need_positive_or_negative = 0)
        d.drop(index  = samples_to_remove, inplace = True)

    for k in need_neg:
        diff = compute_diff_remove(k, temp2, names, label_y, need_positive_or_negative = 0)
        if diff == -1:
          continue
        diff = round_int(diff)
        print("Removed " + str(diff) +" positive records")
        samples_to_remove = make_remove(d, k, diff, names, label_y, need_positive_or_negative = 1)
        d.drop(index  = samples_to_remove, inplace = True)
    return d

#####################################

#  Massaging Algorithm

#####################################
def get_depromotion(d, diff, group_lst, names, label_y, flag_depro):
    input_test = pd.DataFrame(d, columns = columns_compas)
    clf = MultinomialNB()

    temp_train_label = pd.DataFrame(d, columns = [label_y])
    temp_train_label = temp_train_label[label_y]
    temp_train_label = temp_train_label.astype('int')
    clf = clf.fit(input_test, temp_train_label)
    prob  = clf.predict_proba(input_test)[:,0]
    select = copy.deepcopy(d)
    select['prob'] = prob # the higher the probablity is, the more likely for it to be 0
    # filter out those belongs to this group
    for i in range(len(group_lst)):
        att_name = names[i]
        select = select[(select[att_name] == group_lst[i])]
    select = select[(select[label_y] == flag_depro)]
    # rank them according to the probability
    # filp the records and remove the records from d
    if (flag_depro == 0):
        select.sort_values(by="prob", ascending=True, inplace=True)
        select[label_y] = 1
    else:
        select.sort_values(by="prob", ascending=False, inplace=True)
        select[label_y] = 0
    head = select.head(diff)
    index_list = []
    index_list = list(head.index)
    d.drop(index_list,inplace = True)
    head.drop(columns = ['prob'],inplace = True)
    return head



def naive_massaging(d, temp2, names, need_pos, need_neg,label_y):
    # add more records for all groups
    # The smote algorithm to boost the coverage
    for r in need_pos:
    # add more positive records
        # determine how many points to add
        diff = compute_diff_add_and_remove(r, temp2, 1, label_y, names)
        diff =  round_int(diff)
        # add more records
        #0 for promotion
        samples_to_add = get_depromotion(d, diff, r, names, label_y, flag_depro = 0)
        print("Changed " + str(len(samples_to_add)) +" records")
        d = pd.concat([d, samples_to_add])

    for k in need_neg:
        diff = compute_diff_add_and_remove(k, temp2, 0, label_y, names)
        diff =  round_int(diff)
        #1 for demotion
        samples_to_add = get_depromotion(d, diff, k, names, label_y, flag_depro = 1)
        print("Changed " + str(len(samples_to_add)) +" records")
        d = pd.concat([d, samples_to_add])

    return d