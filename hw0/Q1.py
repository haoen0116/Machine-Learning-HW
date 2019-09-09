#!/usr/bin/env
import os,sys
text_file = open(sys.argv[1],'r')
text = text_file.read().split(' ')
text_file.close()
ans_list = []
for i in range(len(text)): #512 word
 if i == 0:
      ans_list.append(text[i])
 elif ans_list.count(text[i]) == 0:
      ans_list.append(text[i])
s = ""
for i in range(len(ans_list)-1):
 s += ans_list[i] + " " + str(i) + " " + str(text.count(ans_list[i])) + "\n"
s += ans_list[299][0:5] + " " + str(i+1) + " " + str(text.count(ans_list[i+1]))
text_save = open("Q1.txt", "w")
text_save.write(s)
text_save.close()