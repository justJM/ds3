
#In[1]

import tensorflow
from tensorflow.python.client import device_lib
def get_available_device():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU' or 'GPU']

print(get_available_device())


#%%
import tensorflow as tf
def gpu_device():
    with tf.device('/cpu:0'):
        a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape = [2,3],name='a')
        b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],shape = [3,2],name='b')
    c= tf.matmul(a,b)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run([c]))
    sess.close
gpu_device()

#%%
import tensorflow as tf
import numpy as np

#%%
import requests
url = 'https://cse.snu.ac.kr'
result = requests.get(url)

mydesk = 'C:\\Python\\'
fp = open(mydesk +'snu_cse.txt','w')
fp.write(result.text)
fp.close()


#%%
from bs4 import BeautifulSoup

url = 'https://cse.snu.ac.kr'
result = requests.get(url)

soup = BeautifulSoup(result.text , 'html.parser')
print(soup.prettify())

mydesk = 'C:\\Python\\'
fp = open(mydesk +'snu_cse2.txt','w')
fp.write(result.text)
fp.close()


#%%
import re 
mylist = ['105-52-33033-C', '010-2343-2498', '205-334-232-12', '010-2322-9765','534233234-93','24C-Z8fm-9892','070-1584-3857','15-12-12311','010-9134-2095']

phone_list = []
for item in mylist:
    matchobj = re.match(r'010-\d{3,4}-\d{4}',item)
    if matchobj:
        phone_list.append(matchobj.group())
print(phone_list)

#%%
import re 
mystr = 'My phone number : 010-3028-2971'
m_matchobj = re.match(r'010-\d{3,4}-\d{4}', mystr)
s_matchobj = re.search(r'010-\d{3,4}-\d{4}', mystr)

print('1',m_matchobj)
print('2',s_matchobj)

print('3',s_matchobj.group())

#%%
import re 
mystr = 'My phone number : 010-3028-2971'

num = re.sub(r'\D',"",mystr)
print(num)

#%%
import re
p = re.compile('[a-z]+')
m = p.match('string goes here')
if m:
    print('Match found :' , m.group())
else:
    print('No match')

#%%
import re
data = '''
park 821818-1000000
kim  919292-2000000
'''
data2=re.sub(('(\d{6})[-]\d{7}'),'\g<1>-*******',data)
print(data)
print(data2)

#%%
line = 'Python Orientation course helps professionals fish best opportunities'
print(re.findall(r'\b\w{13,}\b',line))


#%%
string= 'Python java c++ perl shell ruby tcl c c#'
print(re.findall(r'\bc[\w+]*',string,re.I))

#%%
import csv 
f = open('data.csv','w',encoding='utf-8',newline='')
wr = csv.writer(f)
wr.writerow([1,'김정수',False])
wr.writerow([2,'박상미',True])
f.close()

#%%
import csv 
f = open('eggs.csv','r',encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    print(line)
f.close()

#%%
import csv
spamReader = csv.reader(open('eggs.csv',newline=''),delimiter=' ', quotechar='|')
for row in spamReader:
    print(', '.join(row))

#%%
import csv
with open('names.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['first_name'],row['last_name'])
print(row)

#%%
