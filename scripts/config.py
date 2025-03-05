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
            Note that the meta text might have formatting issues, extraneous symbols, and other common OCR errors, thus you must fix these problems like a professional translator in such cases. If the metadata is empty, the translation should also be empty. If the metadata is already in English, return the original English after fixing errors in text.\
            I will give you some texts of different lengths and you need to translate these data to English.\
            Return only the translated text in English in a clear and structured form."

system_role_editor="You are a helpful assistant specialized in cleaning and normalizing metadata.\
            Your job is to standardize information, fix formatting issues, remove extraneous symbols and correct any capitalization errors.\
            In principle, you should only fix problems like an professional editor and stick to metadata as much as possible without modifying other parts of the metadata. If the metadata is empty, the cleaned data should also be empty.\
            I will give you some texts of different lengths and you need to clean these data.\
            Return only the cleaned text in a clear and structured form."

system_role_summarizer="You are a helpful assistant specialized in multilingual understanding.\
            Your job is to summarize the given meta text in a concise and accurate manner, while still being informative.\
            Note the meta text might have formatting issues, extraneous symbols, and other common OCR errors, thus you must fix these problems like a professional summarizer in such cases. If the metadata is empty, the summary should also be empty.\
            I will give you some texts of different lengths and you need to summarize these data.\
            Return only the summary within 76 tokens in English in a clear and structured form."

llama="meta-llama/Meta-Llama-3.1-8B-Instruct"
deepseek='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
model_translate="Helsinki-NLP/opus-mt-de-en"
qwen='Qwen/Qwen2.5-7B-Instruct-1M'

api_key='REDACTED_OPENAI_KEY'

top_k=10


img2txt_top10=[
    [0, 1, 3, 4, 5, 6, 8, 9, 10, 11],
    [0, 1, 3, 2, 4, 6, 8, 9, 10, 11],
    [0, 1, 3, 4, 8, 9, 10, 11, 12, 13],
    [0, 3, 4, 1, 2, 8, 9, 10, 11, 12],
    [0, 8, 3, 4, 1, 2, 6, 5, 7, 9],
    [0, 8, 2, 3, 4, 1, 5, 6, 7, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 8, 1, 3, 4, 5, 6, 7, 9, 10],
    [0, 3, 1, 2, 4, 5, 6, 8, 7, 9],
    [0, 1, 3, 4, 5, 6, 8, 9, 10, 11],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 3, 4, 5, 8, 9, 10, 11, 12],
    [0, 1, 3, 4, 5, 8, 9, 10, 11, 12],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 3, 4, 5, 8, 9, 10, 11, 12],
    [83, 80, 79, 78, 77, 76, 75, 74, 73, 72],
    [0, 8, 1, 3, 4, 5, 6, 7, 2, 9],
    [0, 8, 1, 3, 4, 6, 2, 5, 7, 9],
    [0, 6, 1, 3, 4, 2, 8, 10, 11, 12],
    [0, 1, 3, 4, 5, 8, 10, 11, 12, 13],
    [59, 60, 61, 62, 63, 64, 65, 66, 67, 68],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
]

txt2img_top10=[[0,1,2,3,4,5,6,7,8,9],
    [0,1,3,4,6,8,10,12,14,16],
    [0,1,8,9,10,11,12,13,14,15],
    [0,1,8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,3,4,5,6,8,10,12,14],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [1,0,3,4,5,6,9,10,12,13],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,8,9,10,11,12,13,14,15],
    [0,1,8,9,10,12,13,14,15,16],
    [0,1,8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,9,10,12,13,14,15,16,17],
    [0,1,8,9,10,11,12,13,14,15],
    [0,1,8,9,10,11,12,13,14,15],
    [18,19,17,16,15,14,13,12,11,10],
    [0,1,3,4,5,6,8,9,10,11],
    [8,9,10,11,12,13,14,15,16,17],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [21,19,18,17,16,15,14,13,12,11],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [8,9,10,11,12,13,14,15,16,17],
    [8,9,10,11,12,13,14,15,16,17],
    [0,1,2,3,4,5,6,7,8,9],
    [9,10,11,12,13,14,15,16,17,18],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [12,13,14,15,16,17,18,19,20,21],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [18,19,17,16,15,14,13,12,11,10],
    [1,2,3,4,5,6,7,8,9,10],
    [8,9,10,11,12,13,14,15,16,17],
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],
    [8,9,10,11,12,13,14,15,16,17],
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,3,4,5,6,8,9,10,11],
    [1,2,3,4,5,6,7,8,9,10],
    [0,1,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,8,9,10,11,12,13,14,15],
    [1,2,3,4,5,6,7,8,9,10],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9,10],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [5,6,7,8,9,10,11,12,13,14],
    [1,0,8,9,10,12,14,15,16,17],
    [0,1,2,3,4,5,6,7,8,9],
    [8,9,10,11,12,13,14,15,16,17],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,3,4,5,8,9,10,12,13],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [1,0,3,4,5,6,7,8,9,10],
    [8,9,10,11,12,13,14,15,16,17],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,3,4,5,6,8,9,10,11],
    [0,1,2,3,4,5,6,7,8,9],
    [0,1,2,3,4,5,6,7,8,9],
    [12,11,10,9,8,7,6,5,4,3],
    [1,2,3,4,5,6,7,8,9,10],
    [0,1,2,3,4,5,6,7,8,9],
    [1,0,2,3,4,5,6,7,8,9],
    [0,1,3,4,6,8,9,10,11,12],
    [8,9,10,11,12,13,14,15,16,17],
    [12,11,10,9,8,7,6,5,4,3],
    [0,1,8,9,10,11,12,13,14,15],
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10]
]

if __name__=='__main__':
    print(extra_stopwrods)
