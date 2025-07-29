# Story Generator

Hi there guys! This is an auto Story Generator using OpenAI and Anthropic's LLMs (of course you can use different LLMs by changing settings in  `settings.py`).

You can use this to generate a story from a given topic and instruction. See if you can generate a funny story with your own prompts!

# Structure
```

StoryGenerator
├── End
│   ├── EndsGenerate.py
│   ├── __init__.py
│   └── build.py
├── Expender
│   ├── ExpenderWriterSimulator.py
│   ├── Interact.py
│   ├── ReaderSimulator.py
│   ├── __init__.py
│   └── build.py
├── Memory
│   ├── MemoryStore.py
│   └── __init__.py
├── PlainGenerator
│   ├── PlainGenerate.py
│   ├── PlainWritingAssistant.py
│   └── __init__.py
├── StoryStarter
│   ├── StoryStartGenerate.py
│   └── __init__.py
├── TwistGenerator
│   ├── TwistGenerate.py
│   └── __init__.py
├── MainGraph.py
├── StoryState.py
├── main.py
├── README.md
├── requirements.txt
├── settings.py
├── result.json
├── test.ipynb
└── utils.py
```
You don't need a strong prompt to generate a story———all prompts and arguments are already prepared for you. If you're not familiar with running code in the shell, you could use  `test.ipynb` notebook to execute the program. Alternatively, follow the steps below.


You'll notice a `result.json` file in this directory. This is just an example to show you where the final output will be saved. You can delete this file and generate your own story.

Note that the bash example below doesn't include all possible arguments. If you want to modify other parameters, please edit the `settings.py` file.
# How to run this project?
## 1 environment setup
Python 3.10 is a good choice for this project that is the version I use.

## 2 install packages
```
pip install -r requirements.txt
```
## 3 run in bash
After install all required packages, you can run 
```
python main.py --OPENAI_API_KEY your_key --ANTHROPIC_API_KEY your_key\
--MAIN_CHARACTOR your_own_character --TOPIC your_story_topic --MAIN_GOAL your_character_main_goal --LANGUAGE your_language
```
or just run
```
python main.py --OPENAI_API_KEY your_key --ANTHROPIC_API_KEY your_key
```
For a sample example, this line generates an English love-fiction.
If you don't have keys, you can visit <https://www.anthropic.com> and <https://openai.com> to get keys.