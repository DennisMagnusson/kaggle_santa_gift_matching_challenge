from __future__ import division

from math import ceil, floor, log

import pandas as pd
import numpy as np
from evaluate import avg_normalized_happiness

def fill_triplets(pred, c_table, item, greedy=True):
  for i in range(0, 5001, 3):
    if pred[i] != -1:
      continue
    if c_table[item] > 997:
      continue
    if not greedy and (c_table[item] == 996 or c_table[item] == 994):
      continue
  
    pred[i] = item
    pred[i+1] = item
    pred[i+2] = item
    c_table[item] += 3

def fill_twins(pred, c_table, wishlists, top, gift):
  if c_table[gift] > 998:
    return
  for t in range(top):
    done = False
    for i in range(5000, 40000, 2):
      if pred[i] != -1:
        continue
      if gift == wishlists[i][t] or gift == wishlists[i+1][t]:
        pred[i] = gift
        pred[i+1] = gift
        c_table[gift] += 2
        done = True
        break
    if done:
      break

def fill_twins_santa_style(pred, c_table, good_stuff, top, g):
  good_bois = good_stuff[g]
  for boi in good_bois[:top]:
    if c_table[g] > 998:
      break
    if boi > 40000 or boi < 5000:
      continue
    if pred[boi] != -1:
      continue
    
    if boi&1:
      boi -= 1

    pred[boi] = g
    pred[boi+1] = g
    c_table[g] += 2

def fill_twins_greedy(pred, c_table, gift):
  for i in range(5000, 40000, 2):
    if c_table[gift] > 998:
      return
    if pred[i] != -1:
      continue

    pred[i] = gift
    pred[i+1] = gift
    c_table[gift] += 2

def fill_rest(pred, c_table, scores):
  for i in range(40000, len(pred)):
    if pred[i] != -1:
      continue

    for item in range(len(c_table)):
      if c_table[item] < 1000:
        pred[i] = item
        c_table[item] += 1
        scores[i] = -2
        break

def radix_sort(pop):
  maxval = -1
  for i in pop:
    if i[1] > maxval:
      maxval = i[1]
  k = int(ceil(log(maxval)/log(2)))

  for i in range(k):
    c = 2**i
    t = 0

    offset = 0
    for t in range(len(pop)):
      if not pop[t-offset][1]&c:
        pop.append(pop.pop(t-offset))
        offset += 1
    

def get_popularity(wishlists):
  m = [0]*1000
  t = []
  for i in range(1000):
    t.append([])

  for w in range(len(wishlists)):
    wishlist = wishlists[w]
    for item in wishlist:
      m[item] += 1
      t[item].append(w)

  return m, t

def obvious_choices_item(pred, item, pop, wishlists, c_table, good_list, table, scores, improve, t1, t2):
  kids = set(table[item]).intersection(good_list[item])
  for kid in kids:
    if c_table[item] >= 1000:
      return
    if kid < 40000:
      continue
    if pred[kid] != -1:
      if improve and 2*(t1+t2) - scores[kid] > 0:#TODO FIXME Should this be 0?
        if item in wishlists[kid][:(11-t1)] and kid in good_list[item][:(11-t2)*100]:#Awesome
          c_table[pred[kid]] -= 1
          pred[kid] = item
          c_table[item] += 1
          scores[kid] = 2*(t1+t2)
   
      continue

    if item in wishlists[kid][:(11-t1)] and kid in good_list[item][:(11-t2)*100]:
      pred[kid] = item
      scores[kid] = 2*(t1+t2)
      c_table[item] += 1


def obvious_choices(pred, pop, wishlists, c_table, good_list, table, t1, t2):
  n_ops = 0
  wishl = wishlists.tolist()
  for item in reversed(pop):
    item = item[0]
    for w in table[item]:#Obvious choices
      if c_table[item] >= 1000:
        break
      if pred[w] != -1:
        continue
      if w < 40000:
        continue
      
      if wishl[w].index(item) > t1:
        continue

      if w in good_list[item][:t2]:
        pred[w] = item
        c_table[item] += 1
        n_ops += 1
        continue

  return n_ops

"""
Change the loop to loop the expected score. Calculate that properly.
Also save the score in an array and compare and see if changing is worth it.
This can be done in the "if pred[i] != -1:" block
"""
#TODO Add prioritization to obvious choices.
def no_sadness(wishlists, good_list, table, pop):
  c_table = [0]*1000
  pred = [-1]*1000000
  scores = [-2]*1000000
  improve = False
  
  for b in range(1):
    if b > 0:
      improve = True
    if b == 9:
      improve = False
    print(sum(scores))

    for item in reversed(pop):
      item = item[0]
      if c_table[item] >= 1000:
        continue

      obvious_choices_item(pred, item, pop, wishlists, c_table, good_list, table, scores, improve, 500, 50)
      """
      for i in range(1, 11):
       obvious_choices_item(pred, item, pop, wishlists, c_table, good_list, table, scores, improve, i, (10-i))
      """

    for item in reversed(pop):
      item = item[0]


      for i in range(100):#Expected score = 2*i-1

        expected_score = 2*(10-i)-1
        s = int(floor(expected_score/2)) + 1
        
        for k in range(0, 101, 10):
          if s <= k:
            continue
          obvious_choices_item(pred, item, pop, wishlists, c_table, good_list, table, scores, improve, k, s-k)
          obvious_choices_item(pred, item, pop, wishlists, c_table, good_list, table, scores, improve, s-k, k)
        
        fill_twins_santa_style(pred, c_table, good_list, (i+1)*10, item)
        fill_twins(pred, c_table, wishlists, i+1, item)

        for kid in good_list[item][:10*i]:#Good_list
          if c_table[item] >= 1000:
            break
          if kid < 40000:
            continue
          if pred[kid] != -1:
            if improve and expected_score - scores[kid] > 0:
              c_table[pred[kid]] -= 1
              pred[kid] = item
              c_table[item] += 1
              scores[kid] = expected_score
             
            continue

          pred[kid] = item
          scores[kid] = expected_score
          c_table[item] += 1
        
        for w in table[item]:#Wishlists
          if c_table[item] >= 1000:
            break
          if w < 40000:
            continue
          if pred[w] != -1:
            if improve and wishlists[w][i] == item and expected_score - scores[w] > 0:
              c_table[pred[w]] -= 1
              pred[w] = item
              c_table[item] += 1
              scores[w] = expected_score
            continue

          wishlist = wishlists[w]
          if item == wishlist[i]:
            pred[w] = item
            scores[w] = expected_score
            c_table[item] += 1

        if i == 75:
          fill_triplets(pred, c_table, item, False)#I am aware that this is stupid.
          fill_twins_greedy(pred, c_table, item)



  for item in reversed(pop):
    item = item[0]

    for i in range(101):
      fill_twins_santa_style(pred, c_table, good_list, i*10, item)
      fill_twins(pred, c_table, wishlists, i, item)
    
    fill_twins_greedy(pred, c_table, item)
    fill_triplets(pred, c_table, item)


  c = 0
  for i in pred:
    if i == -1:
      c += 1
  
  print(str(c) + " sad bois")

  fill_rest(pred, c_table, scores)
  
  for k in range(len(c_table)):
    if c_table[k] != 1000:
      print(k, c_table[k])
  
  """
  for i in range(0, 40000, 2):
    if pred[i] != pred[i+1]:
      print(i, pred[i], pred[i+1])
  """
  return pred


def popularity_prio(wishlists, good_list):
  pop, table = get_popularity(wishlists)
  pop = [[i, pop[i]] for i in range(len(pop))]
  c_table = [0]*1000
  radix_sort(pop)
  pred = no_sadness(wishlists, good_list, table, pop)

  ###TESTS
  c = 0
  for i in range(len(pred)):
    if pred[i] == -1:
      print(i)
      c += 1
  
  print(c)
  
  return pred

if __name__ == "__main__":
  wishlists  = pd.read_csv("data/child_wishlist_v2.csv", header=None).drop(0, 1).values
  good_lists = pd.read_csv("data/gift_goodkids_v2.csv",  header=None).drop(0, 1).values

  pred = popularity_prio(wishlists, good_lists)
  """
  pred = [-1]*1000000
  for i in range(1000):
    for u in range(1000):
      pred[i*1000+u] = u
  """

  for i in range(len(pred)):
    pred[i] = [i, pred[i]]
  #print(avg_normalized_happiness(np.array(pred, dtype=np.int32), wishlists, good_lists))
  for i in pred:
    print(str(i[0])+","+str(i[1]))
