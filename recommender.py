# import numpy as np
# from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
# from datetime import datetime
from loguru import logger
from llm import get_llm
import re

def evaluate_paper_relevance(paper: ArxivPaper, keywords: str, repeated = False):
    llm = get_llm()
    # keywords = open('interest_summary.txt', 'r', encoding='utf-8').read()  # 请修改为你感兴趣的关键词
    prompt = """
Below is the title and abstract of a paper published in arXiv today:
Title: {__TITLE__}
Abstract: {__ABSTRACT__}

Given:
The user's existing Zotero library, containing titles and abstracts of their collected papers. 
\"\"\"
 {__KEYWORDS__}
\"\"\"
These papers cover a cohesive research trajectory encompassing precision measurements in quantum chromodynamics (QCD), extraction and refinement of the strong coupling constant (αₛ), tau-lepton physics, nonperturbative effects in hadronic decays, particle form factors, searches for dark photons and other exotic particles, as well as detector technologies, specifically with a focus on timing resolution, calorimetry, and advanced machine learning reconstruction methods. 
Recognize that the user's interests form a holistic and continuous research narrative rather than isolated fields.

You will:
1. Approach the comparison by understanding the user's thought process as macroscopic, continuous, and holistic, evaluating how the new arXiv submission integrates into or expands upon their overall research narrative,  explicitly grounded in conceptual alignment, methodological continuity, or thematic complementarity to the user's established research interests.
2. Clearly justify each relevance rating by explicitly referencing how the new work relates to the user's broader academic journey—highlighting common themes, overlapping methodologies, or innovative approaches that resonate with or expand their current research interests.

Your output should only in json format:
{
    "relevant": float [0-1], 1 for very relevant, 0 for not relevant
    "reason": "Concise and explicit justification tying the arXiv paper to the user's ongoing, integrated research interests, in {__LANG__}."
}
    """
    prompt = prompt.replace('__LANG__', llm.lang)
    prompt = prompt.replace('__TITLE__', paper.title)
    prompt = prompt.replace('__ABSTRACT__', paper.summary)
    # prompt = prompt.replace('__INTRODUCTION__', introduction)
    # prompt = prompt.replace('__CONCLUSION__', conclusion)
    prompt = prompt.replace('__KEYWORDS__', keywords)

    try:
        llm_output = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert high energy physicist, your are good at academic paper reading and understanding.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        logger.debug(llm_output)
    except Exception as e:
        logger.error(f"Error evaluating paper with SiliconFlow: {e}")
        return e, -1

    # Transform the LLM output to a machine-readable format.
    # 将 LLM 输出转换为可机器读取的格式。
    # If transformation fails, repeat the request.
    # 如果转换失败，请重复请求。
    # If the second request fails, return the error.
    # 如果第二次请求失败，则返回错误。
    try:
        remove_markdown = re.sub(r'```json|```', '', llm_output)
        # remove_markdown = re.sub(r'\$(.*?)\$', r'\\[\1\\]', remove_markdown)

        format_output = eval(remove_markdown)
        explanation = format_output.get("reason", "No Explaination.")
        score = float(format_output.get("relevant", "None"))
    except Exception as e:
        logger.error(f"Error when re-formatting output: {e}")
        logger.error(f"llm_output: {llm_output}")
        logger.error(f"remove_markdown: {remove_markdown}")
        logger.error(f"formate_output: {format_output}")
        if (repeated == False):
            logger.error(f"require response from LLM")
            explanation, score = evaluate_paper_relevance(paper,keywords, repeated=True)
        else:
            logger.error("This is the seconed time LLM gives wrong output, stop here.")
            return e, -1

    return explanation, score

def rerank_paper(candidate:list[ArxivPaper],interests:str,model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    logger.debug(f"papers number: {len(candidate)}")
    for idx, c in enumerate(candidate):
        explanation, score = evaluate_paper_relevance(c, interests)
        c.score = score
        c.explanation = explanation
        
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    return candidate