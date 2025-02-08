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

german_topics=[
    'Störfall',
    'Atomabfall',
    'Protest',
    'Umweltschutz'
    ]
form_de='Ein Artikel'
german_prompts=[f"{form_de} über {query}" for query in german_topics]

extra_stopwords=['Störfall','Umweltschutz','Waldsterben','Unfall','Atomabfall','Protest']

system_role_translator="You are a helpful assistant specialized in multilingual translation.\
            Your job is to translate the given meta text to fluent and accurate English, while preserving the original meaning.\
            Note that the meta text might have formatting issues, extraneous symbols, and other common OCR errors, you must do like a professional translator in such cases. If the metadata is empty, the translation should also be empty. If the metadata is already in English, return the original English after fixing possible issues in text.\
            I will give you some texts of different lengths and you need to translate these data.\
            Return only the English translation."

system_role_editor="You are a helpful assistant specialized in cleaning and normalizing metadata.\
            Your job is to standardize information, fix formatting issues, remove extraneous symbols and correct any capitalization errors.\
            In principle, you should only fix problems like an professional editor and stick to metadata as much as possible without modifying other parts of the metadata. If the metadata is empty, the cleaned data should also be empty.\
            I will give you some texts of different lengths and you need to clean these data.\
            Return only the cleaned data."

system_role_summarizer="You are a helpful asistant specialized in multilingual understanding.\
            Your job is to summarize the given meta text in a concise and accurate manner using the same language of meta text, while still being informative.\
            Note the meta text might have formatting issues, extraneous symbols, and other common OCR errors, you must do like a professional summarizer in such cases. If the metadata is empty, the summary should also be empty.\
            I will give you some texts of different lengths and you need to summarize these data.\
            Return only the summary."

llama="meta-llama/Meta-Llama-3.1-8B-Instruct"
deepseek='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
model_translate="Helsinki-NLP/opus-mt-de-en"
qwen='Qwen/Qwen2.5-7B-Instruct-1M'

top_k=10

if __name__=='__main__':
    print(extra_stopwrods)
