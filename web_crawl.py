from firecrawl.firecrawl import FirecrawlApp
import json, os

app = FirecrawlApp(api_key="fc-3e919f1e284344bfa8bf1d8813c0f085")
INPUT = "url.json"
OUTPUT = "input.txt"
cnt = 0

def wrap(string, max_width):
    """ Limit the length of each line"""
    result1 = [string[i:i + max_width] for i in range(0, len(string), max_width)]
    result = '\n'.join(result1)
    return result

def crawl_web(url:str, content_sel:list):
    """ Crawl contents from the page and clean the data"""
    global cnt
    cnt += 1
    response = app.scrape_url(url=url, params={
        'formats': [ 'markdown'],
        'includeTags': content_sel,
	    'excludeTags': [ 'a', 'img'] # Tags of href & images
    })
    raw_input = response['markdown'].replace('\n', ' ').strip()
    title = response['metadata']['title']
    
    with open(OUTPUT, 'a', encoding='utf-8') as out_f:
        out_f.write(f"[ARTICLE {cnt}: {title}]\n\n") # Title of the News
        out_f.write(raw_input) # Contents
        out_f.write("\n\n")
        
with open(INPUT, "r", encoding="utf-8") as in_f:
    books = json.load(in_f)
    
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)
for book in books:
    crawl_web(book['url'], book['css'])
    