import os 
from tqdm import tqdm

path = '/data/jLog/huqiangqiang/RMB/image3008'
content = os.listdir(path)

txt =  './test3008.txt'
f = open(txt,'w')
for i in tqdm(content):
    f.write(path+'/'+i+'\n')

f.close()
