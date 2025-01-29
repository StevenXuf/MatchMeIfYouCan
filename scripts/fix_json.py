import json

with open('./finished_combination.json','r',encoding='utf-8') as f:
    content=f.read()

content+=']'
data=eval(content)
print(type(data))

with open('./finished_combinations.json','w') as f:
    json.dump(data,f)

with open('./finished_combinations.json','r') as f:
    lst=json.load(f)

print(len(lst))

