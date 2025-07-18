from typing import List , TypedDict



class StoryState(TypedDict, total=False):
    MainCharacter: str
    MainGoal: str
    StartSign: bool
    RecentStory: List[str]
    Language: str
    Topic: str
    similarity: float
    TotalStoryLength: int