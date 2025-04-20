import arxiv
import argparse
import os, time
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm, get_llm
import feedparser

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus

def generate_markdown(zotero_items):
    """
    根据给定的 Zotero 条目列表，生成一个包含标题和摘要的 Markdown 文档字符串。
    
    参数:
        zotero_items (list of dict): 从 pyzotero 获取的条目列表，每个条目为一个 dict，
                                      其中标题在 entry['data']['title']，
                                      摘要在 entry['data']['abstractNote']。
    返回:
        str: 完整的 Markdown 文档字符串。
    """
    md_lines = []
    for entry in zotero_items:
        # 从 entry 中提取标题
        title = entry.get('data', {}).get('title', '').strip()
        # 从 entry 中提取摘要
        abstract = entry.get('data', {}).get('abstractNote', '').strip()
        add_date = entry.get('data', {}).get('dateAdded', '').strip()
        # 如果标题或摘要都为空，则跳过该条目
        if not title and not abstract:
            continue
        
        # 添加一级标题
        md_lines.append(f"# {title}")
        md_lines.append("")  # 空行，用于 Markdown 格式
        
        # 添加日期
        md_lines.append(f"**adding date** {add_date}")
        md_lines.append("")  # 空行，用于 Markdown 格式
        # 添加摘要内容
        md_lines.append(abstract)
        md_lines.append("")  # 空行
        
        # 添加分隔线，分割不同条目
        md_lines.append("---")
        md_lines.append("")  # 空行

    # 将所有行拼接为一个字符串并返回
    return "\n".join(md_lines)

def generate_interest_statement(corpus:list[dict]) -> str:
    llm = get_llm()
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    prompt = generate_markdown(corpus[:100])
    llm_output = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": """
Act as an Academic Literature Consultant.
Your goal is to precisely identify and summarize the academic interests of the user, 
based on the meta data of papers provided from their Zotero library.
You must be keen enough to the evolution of the user's scientific interest over time, and pay more attention to summarizing the user's recent research focus.
Now following is the meta data of the papers:

---

""",
                },
                {"role": "user", "content": prompt},
            ]
        )
    logger.debug(llm_output)
    return llm_output

def get_arxiv_paper(query:str, debug:bool=False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
        papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
        bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
        for i in range(0,len(all_paper_ids),50):
            search = arxiv.Search(id_list=all_paper_ids[i:i+50])
            batch = [ArxivPaper(p) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:hep-ex', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 2:
                break

    return papers



parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--interest_stat', type=str, help='Research Interest Statement', default = None)
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    # Set global LLM
    if args.use_llm_api:
        logger.info("Using OpenAI API as global LLM.")
        set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
    else:
        logger.info("Using Local LLM as global LLM.")
        set_global_llm(lang=args.language)

    # If the INTEREST_STAT is not provided, generate it through Zotero Library.
    if args.interest_stat is None:
        fresh_interest = os.path.exists('interest_stat.md') and time.time() - os.path.getmtime('interest_stat.md') < 86400 * 10
        if not fresh_interest:
            logger.info("Generate interest statement through Zotero Library.")
            logger.info("Retrieving Zotero corpus...")
            corpus = get_zotero_corpus(args.zotero_id, args.zotero_key) # TODO: add cache
            logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
            if args.zotero_ignore:
                logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
                corpus = filter_corpus(corpus, args.zotero_ignore)
                logger.info(f"Remaining {len(corpus)} papers after filtering.")
            interest_stat = generate_interest_statement(corpus)
            with open("interest_stat.md", "w", encoding="utf-8") as f:
                f.write(interest_stat)
        else:
            logger.info("Use cached interest statement.")
            interest_stat = open('interest_stat.md','r', encoding='utf-8').read()
    else:
        logger.info("Use interest statement specified by user.")
        interest_stat = args.interest_stat
        
    
    # Retrieve Arxiv papers
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)

    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        papers = rerank_paper(papers, interest_stat)

    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")

