'''
-- @Time    : 2025/7/18 03:53
-- @File    : main.py
-- @Project : MainGraph.py
-- @IDE     : PyCharm
'''
import argparse


from utils import set_env
set_env()

from MainGraph import main_graph
parser = argparse.ArgumentParser(
        description='story writing')
parser.add_argument("--OPENAI_API_KEY", type=str, default="")
parser.add_argument("--ANTHROPIC_API_KEY", type=str, default="")
parser.add_argument("--MAIN_CHARACTOR", type=str, default="Ellen and Mika, two high school friends who grow up with each other")
parser.add_argument("--SIMILARITY_THRESHOLD", type=float, default=0.8)
parser.add_argument("--TOPIC", type=str, default="love-fiction in high school")
parser.add_argument("--MAIN_GOAL", type=str, default="Mika wants to find the meaning of love and get in love with Ellen forever")
parser.add_argument("--LANGUAGE", type=str, default="English")

args = parser.parse_args()


initial_state = {
    "Language": args.LANGUAGE,
    "Topic": args.TOPIC,
    "MainCharacter": args.MAIN_CHARACTOR,
    "MainGoal": args.MAIN_GOAL
}

result = main_graph.invoke(initial_state,config={"recursion_limit": 100})