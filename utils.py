'''
-- @Project : StoryGenerator
This is a Python test script. Write some content you want to test:
This is some utils you may want to see?
'''
import re
# I tested the other method to calculate the similarity is that one better?see Expender/interact
# I think I put some nodes there.

def pass_INFO(state):
    return state
def set_env():
    import os , getpass
    def _set_env(var: str):
        if not os.environ.get ( var ):
            os.environ[var] = getpass.getpass ( f"{var}: " )
    # Set environment variables to ignore MKL warnings


    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    _set_env ( "OPENAI_API_KEY" )
    _set_env ( "ANTHROPIC_API_KEY" )

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import  util
import os
# Set environment variables to ignore MKL warnings
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
def get_content_between_a_b(a, b, text, none_delete_n = False):
    """
    Extract content between a and b from text using regular expressions.

    :param a: Start marker
    :param b: End marker
    :param text: Text to extract from
    :return: Extracted content with leading and trailing whitespace removed
    """
    if none_delete_n:
        return re.search(f"{a}(.*?)\n{b}", text, re.DOTALL).group(1).strip()

    else:
        return re.search(f"{a}(.*?)\n{b}", text, re.DOTALL).group(1).strip().strip("\n")
if __name__ == '__main__':
    test_text = """
    ## start
    Test the get_content_between_a_b function
    This is the content to be extracted.
    This is the second line of content to be extracted.
    ## end
    """
    print(get_content_between_a_b("## start", "## end", test_text))


"""
The following functions provide a series of functions for processing text paragraphs, calculating paragraph similarity, and segmenting paragraphs.
The main functions include converting text content into a list of paragraphs, calculating paragraph length, calculating similarity between paragraphs, 
identifying positions where the similarity score significantly decreases, segmenting paragraphs based on window size and similarity score, and visualizing the segmentation results.
"""

def content2list(content:str):
    """
    Convert input text content into a list of paragraphs, removing empty paragraphs.

    :param content (str): Input text content.

    return: list[str]: List containing non-empty paragraphs.
    """
    if "\n" in content:
        paralists = content.split("\n")
        # Remove empty strings from the list
        while '' in paralists:
            paralists.remove('')
    else:
        paralists = [content]
    return paralists

def para_length(paralists:List[str]):
    """
    Calculate the cumulative sum of string lengths of each paragraph in the paragraph list and return the cumulative length array.

    Parameters:
    paralists (list): List of paragraphs for which cumulative lengths need to be calculated.

    Returns:
    numpy.ndarray: Array containing the cumulative sum of string lengths of paragraphs in the input list.
    """
    len_para = []
    len_paras_sum = []
    tmp = 0
    try:
        for i in paralists:
            tmp += len(i)
            len_para.append(len(i))
            len_paras_sum.append(tmp)
        len_paras_sum = np.array(len_paras_sum)
        return len_paras_sum
    except:
        return np.array([0])


def get_similarity(paralists:List[str], embedder:SentenceTransformer = embedder):
    """
    Calculate similarity scores between adjacent paragraph string pairs using an embedding model.

    Parameters:
    paralists (list): List of paragraph strings for which similarity scores need to be calculated.
    embedder: Embedding model used to encode paragraph strings.

    Returns:
    numpy.ndarray: Array containing similarity scores between adjacent paragraph string pairs, with an additional score of 1 at the beginning.
    Input paragraph list: ['This is the content of the first paragraph.',
                'This is the content of the second paragraph, somewhat related to the first.',
                'This is the content of the third paragraph, not very related to the second.']
    Calculated similarity scores: [1.
                        0.87858981
                        0.88035393]
    """
    simi_score = []
    for i in range(len(paralists)-1):
        # Encode current paragraph
        target_embedding = embedder.encode(paralists[i], convert_to_tensor=True)
        # Encode next paragraph
        others_embedding = embedder.encode(paralists[i+1], convert_to_tensor=True)
        # Calculate cosine similarity
        memory_scores = util.cos_sim(target_embedding, others_embedding)
        numpy_scores = memory_scores.cpu().numpy()
        simi_score.append(numpy_scores)
    simi_score = np.array(simi_score).reshape(1,-1)[0]
    simi_score = np.concatenate(([1], simi_score))
    return simi_score


def max_drop(simi_score:np.ndarray, max_drop_threshold=0.1, score_threshold=0.7, mode=1):
    """
    Identify indices where similarity scores drop significantly and then fall below a specific threshold.

    Parameters:
    simi_score (numpy.ndarray): Array of similarity scores between adjacent string pairs.
    max_drop_threshold (float): Maximum allowed drop value for similarity scores.
    score_threshold (float): Threshold below which similarity scores are considered low.
    mode (int): Starting index for processing similarity scores.

    Returns:
    numpy.ndarray: Array containing indices where similarity scores drop significantly and fall below the threshold.
    """
    simi_score = simi_score[mode:]
    diff = [0]
    for i in range(len(simi_score)-1):
        diff.append(simi_score[i]-simi_score[i+1])
    diff = np.array(diff)
    # Find indices where the difference is greater than the maximum drop threshold
    idx = (np.argwhere(diff > max_drop_threshold)).flatten()
    del_id = np.where(simi_score[idx] >= score_threshold)
    idx = np.delete(idx, del_id)
    return idx+mode

def calculate_two_para_similarity(para1:str, para2:str, model:SentenceTransformer = embedder):
    str1_embedding = model.encode(para1,convert_to_tensor=True)
    str2_embedding = model.encode(para2, convert_to_tensor=True)
    similarity_score = util.cos_sim(str1_embedding, str2_embedding)
    return similarity_score



def Seperate_window(paralists:List, simi_score:List, min_length:int = 200,str_range:int=1000,threshold:float=None):
    """
    Move a window to find suitable paragraph indices for segmentation based on paragraph length and similarity scores.
    Parameters:
    paralists (list): Input paragraph list.
    simi_score (list): List of similarity scores between adjacent paragraphs.
    min_length (int): Minimum paragraph length requirement.
    str_range (int): Paragraph length range.
    threshold (float): Similarity score threshold.
    Returns:
    list: List of segmented paragraph indices.
    Input paragraph list: ['This is the content of the first paragraph.',
                'This is the content of the second paragraph, somewhat related to the first.',
                'This is the content of the third paragraph, not very related to the second.']
    Calculated similarity scores: [1.
                        0.87858981
                        0.88035393]
    Segmented paragraph index list: [1]
    """
    len_paras = para_length(paralists)
    if len_paras[-1] <= min_length:
        print(f"Paragraphs' length too short, paragraph list: [{paralists[0]}...(etc)] requiring total length larger than {min_length} up to the last paragraph: {len_paras[-1]}.\n"
              f"Paragraph length is too short. Paragraph list: [{paralists[0]}...(etc)] requires cumulative length greater than {min_length} for the last paragraph, but actual length is {len_paras[-1]}. Cutting directly")
        return [len(paralists)-1]
    left = -2
    right = 0
    min_threshold, max_threshold= 0,0
    # Dimension check
    assert len(len_paras)==len(simi_score), f"Dimension Not Match for {len(len_paras)} == {len(simi_score)}\nDimension mismatch {len(len_paras)} == {len(simi_score)}"
    chosen_idx = []
    while left < right and right <(len(paralists) -1):
        right = min ( right , len ( paralists ) - 1 )
        if min_threshold == 0:
            min_threshold = min ( min_length , len_paras[-1] )
        else:
            min_threshold = min ( max_threshold , len_paras[-1] )
        max_threshold = min ( min_threshold + str_range , len_paras[-1] )
        # print(min_threshold, max_threshold, len_paras - min_threshold)
        try:
            # Find the first index where cumulative length is greater than or equal to the minimum threshold
            left = int(np.argwhere((len_paras - min_threshold)>=0)[0][0])# type: ignore
            # Find the first index where cumulative length is greater than or equal to the maximum threshold
            print('left',left)
            right = int(np.argwhere((len_paras - max_threshold)>=0)[0][0]) # type: ignore
            print('right',right)
        except:
            print(f""" Unable to find suitable left (cumulative length <= minimum threshold) and right (cumulative length >= maximum threshold) indices!\n
            len_paras(number of paragraphs): {len_paras}, min_threshold(current minimum threshold): {min_threshold}
            len_paras - min_threshold(cumulative paragraph length - minimum threshold): {len_paras - min_threshold}
            """)
        try:
            idx = np.argmin(simi_score[left+1:right])+left+1
        except:
            print(f""" ValueERROR, you can reset your thresholds. Threshold setting is unreasonable, left idx (cumulative length <= minimum threshold) index= {left}, right idx (cumulative length >= maximum threshold) index= {right}""")
            right = right + 1
            continue
        if threshold:
            if simi_score[idx] < threshold:
                chosen_idx.append(idx)
        else:chosen_idx.append(idx)
        print(f"left(first index with cumulative length <= minimum threshold)={left}, right(first index with cumulative length >= maximum threshold)={right}, chosen idx={idx}, its score={simi_score[idx]}")


    if len(chosen_idx) < 1:
        print(f"WEIRD Content!! Please Check!!{'='*50}\nContentERROR:{paralists}\n{'='*100}")
        chosen_idx.append(len(paralists)-1)
    return [int(idx) for idx in chosen_idx]

def Seperate_similiraty(paralists:List, simi_score:List, threshold:float=None):
    """
    Find paragraph indices where similarity scores are below the threshold.
    Parameters:
    paralists (list): Input paragraph list.
    simi_score (list): List of similarity scores between adjacent paragraphs.
    threshold (float): Similarity score threshold.
    Returns:
    list: List of segmented paragraph indices.
    """
    idxes = np.argwhere(simi_score< threshold)
    return [int(idx) for idx in idxes]


def Seperate(paralists:List, simi_score:List, min_length:int = 200,str_range:int=1000,threshold_windows:float=None, threshold_smiliraty:float=None):
    idx_win = Seperate_window(paralists, simi_score, min_length, str_range, threshold_windows)
    idx_simi = Seperate_similiraty(paralists, simi_score, threshold_smiliraty)
    idx = idx_win+idx_simi
    return sorted(list(set(idx)))



def cut_paras(paralists:List[str], cut_idx:List[int]):
    """
    Segment the paragraph list according to the cutting indices. If the last segment is too short, merge it into the previous segment.

    Parameters:
    paralists (list): List of paragraphs.
    cut_idx (list): List of cutting indices.

    Returns:
    List, int: Tuple of the segmented paragraph list and a merge flag.
    """
    print(f"Cut idx = {cut_idx}")
    if cut_idx[-1]==0:
        paraparts = [paralists]
        return paraparts

    assert len(paralists) >= cut_idx[-1], f"outline_calculate.py：cut_paras function cutting index out of range, List out of RANGE for {len(paralists)} >= {cut_idx[-1]}"
    paraparts= []
    for i in range(len(cut_idx)+1):
        if i==0:
            slice=paralists[0:cut_idx[i]]
        elif i==len(cut_idx):
            slice=paralists[cut_idx[i-1]:]
        else:
            slice = paralists[cut_idx[i-1]:cut_idx[i]]
        paraparts.append(slice)
    last = paraparts[-1]
    merge = 0
    if len("\n".join(last)) < 100:
        paraparts[-2] = paraparts[-2]+paraparts[-1]
        paraparts = paraparts[:-1]
        merge = 1
    return paraparts, merge


def plot_cut(simi_score, cut_idx, title:str):
    """
    Visualize similarity scores and cutting points.

    Parameters:
    simi_score (numpy.ndarray): Array of similarity scores.
    cut_idx (list): List of cutting point indices.
    title (str): Chart title.
    """
    plt.title(title)
    # print(len(simi_score[cut_idx]), simi_score[cut_idx])
    # Draw scatter plot of all similarity scores
    plt.scatter(range(len(simi_score)), simi_score)
    # Draw line plot of all similarity scores
    plt.plot(range(len(simi_score)), simi_score)
    # Draw scatter plot of cutting points
    plt.scatter(cut_idx, simi_score[cut_idx], c="red")
    for k in cut_idx:
        plt.annotate(str(k), (k, simi_score[k]), c="red")
    plt.show()


import networkx as nx
import matplotlib.pyplot as plt
import json

def visualize_knowledge_graph(json_data, output_file, title="visual KG", figsize=(12, 10), font_family=None):
    """
    Draw a visualization graph based on JSON-formatted knowledge graph data and save it as an image

    Parameters:
    - json_data: JSON data containing entities and relations (dict format or JSON string)
    - output_file: Path to save the PNG image file
    - title: Image title
    - figsize: Image size, tuple (width, height)
    - font_family: Specify Chinese-supported fonts, such as ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    """
    # If input is a JSON string, parse it into a dict
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            raise ValueError("Input JSON string format is incorrect.")

    # Check if data structure contains entities and relations
    if "entities" not in json_data or "relations" not in json_data:
        raise ValueError("JSON data must contain the 'entities' and 'relations' fields")

    # Set Chinese font
    if font_family:
        plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False  # Solve the problem of negative sign display

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for entity in json_data["entities"]:
        G.add_node(entity["id"], label=entity["name"], type=entity["type"])

    # Add edges
    for relation in json_data["relations"]:
        G.add_edge(relation["subject"], relation["object"], label=relation["predicate"])

    # Create figure
    plt.figure(figsize=figsize)

    # Node layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)

    # Group nodes by type and set different colors
    node_types = list({entity["type"] for entity in json_data["entities"]})
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6']
    color_map = {node_type: colors[i % len(colors)] for i, node_type in enumerate(node_types)}

    # Draw nodes
    for node_type in node_types:
        nodes = [node for node, data in G.nodes(data=True) if data["type"] == node_type]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=color_map[node_type],
            node_size=2000,
            alpha=0.8,
            label=node_type
        )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=1.5,
        alpha=0.6,
        edge_color='gray',
        arrowsize=20
    )

    # Draw node labels
    node_labels = {node: data["label"] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

    # Draw edge labels
    edge_labels = {(u, v): data["label"] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=10,
        label_pos=0.3,
        bbox=dict(alpha=0)
    )

    # Add legend
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1)

    # Set title
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # Save image
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"KG png saved as {output_file}")