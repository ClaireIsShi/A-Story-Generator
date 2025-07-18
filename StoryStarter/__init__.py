from StoryStarter.build import Starter_subgraph
# 外部用法
'''
from StoryStarter import Starter_subgraph

# compile graph
compiled_graph = Starter_subgraph.compile()
initial_state = {
            "Language": "English",
            "Topic": "love-fiction in high school",
            "MainCharacter": 'Eren and Mikasa',
            "MainGoal": "Eren and Mikasa get in love"
}
result = graph.invoke(initial_state)

Result example is: Dict[str, ]
MainCharacter:Eren and Mikasa
MainGoal:Eren and Mikasa get in love
StartSign:True
RecentStory:["**This is the beginning of a long story** Eren and Mi... it."]
Language:English
Topic:love-fiction in high school
similarity:0
TotalStoryLength:0


'''