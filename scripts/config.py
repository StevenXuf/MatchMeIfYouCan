in_file='../data/swiss_poster_links.csv'
out_file='../data/swiss_poster_data_n45.pt'
anno_file='../data/poster_anno.xlsx'

english_topics=[
    'nuclear accidents',
    'nuclear waste',
    'environmentalism'
    ]

english_queries=[
    'Nuclear Accidents',
    'Nuclear Waste',
    'Environmentalism, Deforestation'
    ]
form_en='An article'
english_prompts=[f'{form_en} concerning {query.lower()}' for query in english_queries]

german_queries=[
    'Störfall',
    'Atomabfall',
    'Umweltschutz, Waldsterben'
    ]
form_de='Ein Artikel'
german_prompts=[f"{form_de} über {query}" for query in german_queries]

system_role_translator="You are a professional translator in German language.\
            I will give you some articles, and you need to translate them from German to English.\
            Note the German article might contain some typos or errors, you must do as a professional translator in such cases.\
            Plus, please remove unnecessary symbols such as line breaks, paragraph breaks, spaces, and tabs, etc.\
            Only output translated articles."

system_role_editor="You are a professional editor in German language.\
            I will give you some articles, and you need to correct the errors or typos according to the context in the article.\
            Plus, please remove unnecessary symbols such as line breaks, paragraph breaks, spaces, and tabs, etc.\
            Only output the article after correction."

system_role_de_summarizer="You are a prefessional German litterateur with good comprehesion.\
            I will show you some articles, and you need to summarize them using concise sentences in German.\
            Note the German articles might contain some typos, special symbols, or errors, so you must distinguish them yourself when reading.\
            Output only the sammarization of given articles, and it should not exceed 76 words."

system_role_en_summarizer="You are a prefessional German translator with good comprehesion.\
            I will show you some articles, and you need to summarize them using concise sentences in English.\
            Note the German articles might contain some typos, special symbols, or errors, so you must distinguish them yourself when reading.\
            Output only the English sammarization of given articles, and it should not exceed 76 words."

llama_3_1_8b_instruct = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_translate="Helsinki-NLP/opus-mt-de-en"

top_k=10
