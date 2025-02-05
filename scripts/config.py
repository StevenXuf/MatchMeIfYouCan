in_file='../data/swiss_poster_links.csv'
out_file='../data/swiss_poster_data.pt'
anno_file='../data/poster_anno.xlsx'

english_topics=[
    'nuclear accidents',
    'nuclear waste',
    'protest',
    'environmentalism'
    ]

form_en='An article'
english_prompts=[f'{form_en} concerning {query.lower()}' for query in english_topics]

german_queries=[
    'Störfall',
    'Atomabfall',
    'Protest',
    'Umweltschutz'
    ]
form_de='Ein Artikel'
german_prompts=[f"{form_de} über {query}" for query in german_queries]

extra_stopwords=['Störfall','Umweltschutz','Waldsterben','Unfall','Atomabfall','Protest']

system_role_translator="You are a professional translator in German language.\
            I will give you some articles, and you need to translate them from German to English.\
            Note the German article might contain some typos or errors, you must do as a professional translator in such cases.\
            Plus, please remove unnecessary symbols such as line breaks, paragraph breaks, spaces, and tabs, etc.\
            Only output translated articles."

system_role_editor="You are a helpful assistant that cleans metadata.\
            I will give you some articles, and you need to clean the meta data through methods like correcting the errors or typos according to the context,\
            or removing unnecessary symbols such as line breaks, paragraph breaks, spaces, and tabs, etc.\
            Only output the article after correction. No verbose explainations."

system_role_de_summarizer="You are a prefessional German litterateur with good comprehesion.\
            I will show you some articles, and you need to summarize them using concise sentences in German.\
            Note the German articles might contain some typos, special symbols, or errors, so you must distinguish them yourself when reading.\
            Output only the sammarization of given articles, and it should not exceed 76 words."

system_role_en_summarizer="You are a prefessional German translator with good comprehesion.\
            I will show you some articles, and you need to summarize them using concise sentences in English.\
            Note the German articles might contain some typos, special symbols, or errors, so you must distinguish them yourself when reading.\
            Output only the English sammarization of given articles, and it should not exceed 76 words."

llama="meta-llama/Meta-Llama-3.1-8B-Instruct"
deepseek='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
model_translate="Helsinki-NLP/opus-mt-de-en"

top_k=10

if __name__=='__main__':
    print(extra_stopwrods)
