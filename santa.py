from __future__ import division

import pandas as pd
import numpy as np
from evaluate import avg_normalized_happiness

"""
Okay, some planning here.

ANSH can prob be done manually. But what is the optimal way?

PCA for the rest. Maybe down the n_gifts to a hundred?

After testing random, ANCH is more important

"""

#Average normalized child happiness.
#Maximize this.
def anh(wishlists, good_list, gifts):
  nch = anch(wishlists, gifts)
  print(nch)
  nsh = ansh(good_list, gifts)
  print(nsh)
  return nch + nsh

#TODO This to numpy
def anch(wishlists, gifts):
  s = 0
  for i in range(len(wishlists)):
    wishlist = wishlists[i]
    l = 10
    gift = gifts[i]
    index = -1
    try:
      index = wishlist.index(gift)
    except ValueError:
      pass
    s += 2*(l-index) if index != -1 else -1

  return s / (len(wishlists)*20)


def ansh(good_list, gifts):
  s = 0
  n_gifts = 1000
  for i in range(len(gifts)):
    gift = gifts[i]
    index = -1
    try:
      index = good_list[gift].index(i)
    except ValueError:
      pass

    val = 2*(1000-index) if index != -1 else -1
    s += val
  return s / (2000*1000*1000)

def combo_kids(pred, c_table, wishlists, top):
  n_ops = 0
  for i in range(4000, len(wishlists[4000:])):
    if pred[i] != -1:#Ignore done kids
      continue

    wishlist = wishlists[i]
    for w in range(top):
      wish = wishlist[w]
      if c_table[wish] < 1000:
        pred[i] = wish
        c_table[wish] += 1
        n_ops += 1
        break

  return n_ops

#TODO May be something wrong here.
def combo_santa(pred, c_table, good_stuff, top):
  n_ops = 0
  for i in range(len(good_stuff)):
    if c_table[i] >= 1000:#Ignore finished gifts
      continue

    l = good_stuff[i]
    for index in range(top):
      kid = l[index]
      if kid < 4000:
        continue
      if pred[kid] != -1:
        continue
      if c_table[i] >= 1000:
        continue
      
      pred[kid] = i
      c_table[i] += 1
      n_ops += 1

  return n_ops

def obvious_choices(pred, c_table, good_stuff, wishlists, top1, top2):
  n_ops = 0
  for i in range(4000, len(wishlists)):
    if pred[i] != -1:
      continue

    wishlist = wishlists[i]
    for w in range(top1):
      wish = wishlist[w]
      very_good_bois = good_stuff[wish][:top2]
      if i in very_good_bois:
        if c_table[wish] >= 1000:
          continue

        pred[i] = wish
        n_ops += 1
        c_table[wish] += 1
        break

  return n_ops

def fill_twins(pred, c_table, wishlists, top):
  for gift in range(len(c_table)):
    if c_table[gift] > 998:
      continue
    for t in range(top):
      done = False
      for i in range(0, 4000, 2):
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

#TODO Incredibly slow
def fill_twins_santa_style(pred, c_table, good_stuff, top):
  for g in range(len(good_stuff)):
    good_bois = good_stuff[g]
    for boi in good_bois[:top]:
      if c_table[g] > 998:
        break
      if boi > 4000:
        continue
      if pred[boi] != -1:
        continue
      
      if boi&1:
        boi -= 1

      pred[boi] = g
      pred[boi+1] = g
      c_table[g] += 2

def fill_twins_greedy(pred, c_table):
  for i in range(0, 4000, 2):
    if pred[i] != -1:
      continue

    for gift in range(len(c_table)):
      if c_table[gift] > 998:
        continue
      pred[i] = gift
      pred[i+1] = gift
      c_table[gift] += 2
      break

def fill_rest(pred, c_table):
  for i in range(4000, len(pred)):
    if pred[i] != -1:
      continue

    for item in range(len(c_table)):
      if c_table[item] < 1000:
        pred[i] = item
        c_table[item] += 1
        break

#Combo of both greedy solutions.
"""
Idea:
what if we do obvi on high top1 and low top2
and then low top1 and high top2
"""
def greedy_combo(wishlists, good_list):
  #c_table = [0]*1000
  c_table = np.zeros(1000, dtype=int)
  pred = np.repeat(-1, 1000000)

  print(obvious_choices(pred, c_table, good_list, wishlists, 4, 900))
  print(obvious_choices(pred, c_table, good_list, wishlists, 9, 400))

  for i in range(1, 10):
    i1=0
    i3 = combo_santa(pred, c_table, good_list, i*100)
    i2 = combo_kids(pred, c_table, wishlists, i)
    fill_twins_santa_style(pred, c_table, good_list, i*100)
    fill_twins(pred, c_table, wishlists, i)
    print(i1, i2, i3)
 
  obvious_choices(pred, c_table, good_list, wishlists, 10, 1000)
  combo_kids(pred, c_table, wishlists, 10)
  combo_santa(pred, c_table, good_list, 1000)

  fill_twins(pred, c_table, wishlists, 10)
  fill_twins_santa_style(pred, c_table, good_list, 1000)
  fill_twins_greedy(pred, c_table)#This is wrong

  for i in range(0, 4000, 2):
    if pred[i] != pred[i+1]:
      print(i, pred[i], pred[i+1])

  fill_rest(pred, c_table)

  #TEST CASES
  for i in c_table:
    if i != 1000:
      print("failed first", i)

  k = 0
  for i in range(len(pred)):
    if pred[i] == -1:
      #print("Failed second", i)
      k += 1
  #print(k)


  for i in range(0, 4000, 2):
    if pred[i] != pred[i+1]:
      print(i, pred[i], pred[i+1])

  return pred
  

def test():
  pred = []
  c = 0
  for i in range(1000):
    for _ in range(1000):
      pred.append(i)

      c += 1

  return pred

if __name__ == "__main__":
  wishlists  = pd.read_csv("data/child_wishlist.csv",header=None).drop(0, 1).values
  good_lists = pd.read_csv("data/gift_goodkids.csv", header=None).drop(0, 1).values

  #pred = greedy_kids(wishlists)
  pred = greedy_combo(wishlists, good_lists).tolist()
  #print(anh(wishlists.tolist(), good_lists.tolist(), pred))
  for i in range(len(pred)):
    pred[i] = [i, pred[i]]
  print(avg_normalized_happiness(pred))
  #for i in pred:
    #print(str(i[0])+","+str(i[1]))
