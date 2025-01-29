from graphviz import Digraph

dot = Digraph('Pipeline', format='pdf')
dot.graph_attr['rankdir'] = 'LR' 
dot.graph_attr['size']='8,8'

with dot.subgraph(name='cluster1') as branch1:
    branch1.attr(style='dashed', color='lightblue', label='Poster Processing',fontcolor='blue',fontsize='15')
    branch1.node('X', 'Scrape Poster Data', style='filled', fillcolor='lightblue')

with dot.subgraph(name='cluster2') as branch2:
    branch2.attr(style='dashed', color='lightgreen', label='Text Processing',fontcolor='blue',fontsize='15')
    branch2.node('A','Gather Articles \n via Impresso', style='filled',fillcolor='lightblue')
    branch2.node('B', 'Transform text \n via Llama-3.1-8B-Instruct', style='filled', fillcolor='lightgreen')

with dot.subgraph(name='cluster3') as branch3:    
    branch3.attr(style='dashed', color='gold', label='Vectorization',fontcolor='blue',fontsize='15')
    branch3.node('D', 'Feature Extraction \n via CLIP/BLIP', style='filled', fillcolor='lightgoldenrod')
    branch3.node('C', 'BoW/TF-IDF', style='filled', fillcolor='lightgoldenrod')

with dot.subgraph(name='cluster4') as branch4:
    branch4.attr(style='dashed', color='purple', label='Evaluation',fontcolor='blue',fontsize='15')
    branch4.node('E', 'Compute Similarity', style='filled', fillcolor='lightpink')
    branch4.node('F', 'Cross-modal Retrieval \n Based on Ranked Distance', style='filled', fillcolor='lightcoral')

# Add edges
dot.edges(['AB','BC', 'CE', 'DE', 'EF','XD','BD'])

# Render to file
dot.render('general_pipeline', cleanup=True)

