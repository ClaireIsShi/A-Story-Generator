'''
-- @Time    : 2025/7/10 01:46
-- @File    : SimilaityCalculate.py
-- @Project : StoryGenerator
-- @IDE     : PyCharm
'''

import os,sys,json
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, TypedDict, Optional
import warnings
warnings.filterwarnings("ignore")
from utils import get_content_between_a_b,set_env
set_env()
from settings import UTIL_LLM
# 将上一级目录添加到 sys.path 中
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

LENGTH = 300
IMPORTANT_PROMPT = """
Here is a outline of a story part. You need to extract the key parts of this outline from it.
Follow these steps:
1. Extract the beginning and the end of the story;
2. Summarize the important contents of the beginning and the end;
3. Your language should be {language}.
The operation outline you need to follow is as follows: {last_story}
Output your result in the following format, don't change the format English, such as "## abstraction:":
## abstraction:
<here put your abstraction of the outline in particular language>
## END
"""
ABSTRACT_PROMPT ="""
You are a knowledge graph extractor. Your task is to extract the relevant triples of the knowledge graph from the given text based on the following relationships :{abstract}. Output the knowledge graph triples in a specific ER(Entity-Relationship Model)format. Here's an example:
{{
  "entities": [
    {{"id": "e1", "name": "William Shakespeare", "type": "Person"}},
    {{"id": "e2", "name": "England", "type": "Country"}},
    {{"id": "e3", "name": "Hamlet", "type": "LiteraryWork"}}
  ],
  "relations": [
    {{"id": "r1", "subject": "e1", "predicate": "was born in", "object": "e2"}},
    {{"id": "r2", "subject": "e1", "predicate": "wrote", "object": "e3"}}
  ]
}}
Output your result in the following format, abstract particular words in {language}, don't change the format English, such as "## KG:"::
## KG:
<here put your knowledge graph in particular language>
## END
"""
GENERATE_TWIST_PRMPT = """
You're a story generator and a native speaker of {language}. Your task is to generate a extra node and its respect relations based on the following entities and relations of the short story, and generate corresponding outline at least write {length} words in {language}: {KG}.
Follow these steps:
1. find the most interesting part to generate a story obstruct node;
2. generate a story obstruct node, and its respect relations to existed nodes, let the story can be expend based on the original nodes;
3. based on the story obstruct node, generate the continuation of the story, your outline should reflect some growth of the related characters, better be around the story obstruct node;
4. Your output should be in {language}, and your story should be still across to this topic: {topic}.
Output your result in the following format,don't change the format English, such as "## KG after generated:":
## KG after generated:
<here add your generated node and relations to the original KG, in original ER(Entity-Relationship Model) format>
## outline:
<here put your story outline in particular language at least write {length} words>
## END
"""

def generate_twist(language: str, topic: str, KG:str, length = 500, llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.8)) -> str:
    prompt_generate = GENERATE_TWIST_PRMPT.format(language=language, KG=KG, topic=topic,length=length)
    story = llm.invoke(prompt_generate).content
    return story
def parser(story: str) -> (str, json):
    outline = get_content_between_a_b("## outline:", "## END",story)
    return outline
def process_twist(language: str, topic: str, KG:str, length = 500, llm = UTIL_LLM)-> (str, json):
    '''
    generate a twist of the story
    :param language: str, the language of the story
    :param topic: str, the topic of the story
    :param KG: str, the knowledge graph of outline
    :param length: int, minium length of generated outline
    :param llm: model
    :return: (generated outline: str,
            KG: json)
    '''
    button = True
    trying = 0
    while button and trying < 4:
        try:
            p = parser(generate_twist(language, topic,KG,length, llm))
            button = False
        except:
            trying += 1
    if trying == 4:
        print("Failed to generate twist.")
        sys.exit()
    return p

def get_abstract(last_story: str, language: str,llm=UTIL_LLM) -> Optional[str]:
    prompt = IMPORTANT_PROMPT.format(last_story=last_story, language=language)
    story_abstract = llm.invoke(prompt).content
    abstract = get_content_between_a_b("## abstraction:", "## END",story_abstract)
    story_KG = llm.invoke(ABSTRACT_PROMPT.format(abstract=abstract, language=language)).content
    KG = get_content_between_a_b("## KG:", "## END",story_KG)
    return KG