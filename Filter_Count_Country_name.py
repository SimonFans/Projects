from translate import Translator
import csv
import pycountry
import re
from collections import defaultdict

# define a translation function

def translate_fun(sentence):
    translator= Translator(from_lang="Korean",to_lang="english")
    translation = translator.translate(sentence)
    return translation

# res: save translted result | countries_name_list: save country nanme later used as a pattern list
res=[]
countries_name_list=[]

# add more country names
for i in range(len(pycountry.countries)):
    countries_name_list.append(list(pycountry.countries)[i].name)
countries_name_list.append('Korea')
countries_name_list.append('United States')
countries_name_list.append('Vietnam')

# open csv file and translate, save result to res
with open('/Users/xzhao/Downloads/practice_sample.csv') as csvfile:
    reader=csv.reader(csvfile,delimiter=',')
    for row in reader:
        translate_result=translate_fun(row[2])
        res.append(translate_result)

# parse country name from each string, save into dictionary, which key is country name and value is counter
dict=defaultdict(int)
for s in res:
    s_split=s.split()
    for word in s_split:
        for pattern in countries_name_list:
            if re.search(pattern,word):
                dict[pattern]+=1
        if word=='States':
            dict['United_States']+=1
        if word[-5:]=='China':
            dict['China']+=1


# write translated result to a csv column
with open('/Users/xzhao/Downloads/practice_sample_translate.csv','w') as csvfile:
    for row in res:
        csvfile.write(''.join(row)+'\n')
        
