from graphviz import Digraph

dot = Digraph('Pipeline', format='pdf')
dot.graph_attr['rankdir'] = 'LR' 
dot.graph_attr['size']='8,8'

with dot.subgraph(name='cluster1') as branch1:
    branch1.attr(style='dashed', color='lightblue', label='Poster Processing',fontcolor='blue',fontsize='15')
    branch1.node('X', 'Scraping Poster Data', style='filled', fillcolor='lightblue')
    branch1.node('Z','Poster Sifting, Labeling\n and Normalizing',style='filled',fillcolor='lightgreen')

with dot.subgraph(name='cluster2') as branch2:
    branch2.attr(style='dashed', color='lightcoral', label='Topic Defining',fontcolor='blue',fontsize='15')
    branch2.node('Y','Selecting Country\n and Defining Topics',style='filled', fillcolor='lightseagreen')

with dot.subgraph(name='cluster3') as branch3:
    branch3.attr(style='dashed', color='lightgreen', label='Text Processing',fontcolor='blue',fontsize='15')
    branch3.node('A','Gathering Articles\n via Impresso Platform', style='filled',fillcolor='lightblue')
    branch3.node('B', 'Transforming Text\n via Qwen2.5-7B-Instruct', style='filled', fillcolor='lightgreen')

with dot.subgraph(name='cluster4') as branch4: 
    branch4.attr(style='dashed', color='gold', label='Vectorization',fontcolor='blue',fontsize='15')
    branch4.node('D', 'Extracting Features\n via CLIP/BLIP', style='filled', fillcolor='lightgoldenrod')

with dot.subgraph(name='cluster5') as branch5:
    branch5.attr(style='dashed', color='purple', label='Evaluation',fontcolor='blue',fontsize='15')
    branch5.node('E', 'Computing Metrics\n based on Ranked Similarity', style='filled', fillcolor='lightpink')
    branch5.node('F', 'Retrieving Cross-modal Data', style='filled', fillcolor='lightcoral')

# Add edges
dot.edges(['YX','YA','AB','BD','DE', 'EF','XZ','ZD'])

# Render to file
dot.render('general_pipeline', cleanup=True)

