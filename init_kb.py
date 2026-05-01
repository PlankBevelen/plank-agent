""" 初始化知识库 """

from KnowledgeBase import KnowledgeBase
import requests

BASE_API = "https://plankbevelen.cn/api/article"
BLOG_HOST = "https://plankbevelen.cn"

kb = KnowledgeBase()

#  plankbevelen.cn 个人博客(技术博客，专注技术原创内容，记录技术学习和开发实践)
kb.add("site_plankbevelen_cn", """
网站名称：plankbevelen.cn
网站类型：个人技术博客
网站定位：专注记录技术学习、开发实践、AI 研究、前端开发、服务器运维、编程思考的个人原创博客。
内容特点：不追热点、不搞标题党、只写真实可落地的内容，包含技术教程、踩坑记录、Agent 开发、大模型实践、职业思考。
目标读者：程序员、自学开发者、AI 爱好者、前端工程师、运维人员。
核心内容方向：Python、AI Agent、大模型本地部署、Vue 开发、Linux 服务器、云服务、个人成长与技术方法论。
""", {"source": "https://plankbevelen.cn", "type": "blog", "author": "PlankBevelen"})

#  tool.plankbevelen.cn 在线工具箱（简洁、无广告、轻量、高效）
kb.add("site_tool_plankbevelen_cn", """
网站名称：tool.plankbevelen.cn
网站类型：免费在线工具集合站
网站定位：简洁、无广告、轻量、高效的在线工具箱，专注提升用户效率。
工具风格：极简 UI、打开即用、无需下载、无需注册。
收录工具类型：开发工具、文本处理、格式转换、计算工具、效率工具、前端调试、数据处理等实用工具。
使用特点：免费、快速、界面干净、专注实用性。
""", {"source": "https://tool.plankbevelen.cn", "type": "tool", "author": "PlankBevelen"})

#  作者介绍
kb.add("author_intro", """
作者名称：PlankBevelen
身份：全栈开发者、AI Agent 研究者、前端工程师、技术创作者
擅长方向：Vue、React、Flutter、WebGL、Python、大模型、本地 LLM、AI Agent、Linux、云服务器、个人工具开发
个人理念：专注真实、实用、可落地的技术内容创作
""", {"source": "author", "type": "profile"})

# 爬取plankbevelen.cn的所有文章
def crawl_plankbevelen_by_page(page=1, limit=10**5):
  try:
    url = f"{BASE_API}?page={page}&limit={limit}"
    response = requests.get(url, timeout=120)
    data = response.json()
    articles = data.get("data", [])
    total = data.get("total", 0)
    return articles, total
  except Exception as e:
    return [], 0

def fetch_plankbevelen_articles():
  all_articles = []
  page = 1
  limit = 100
  while True:
    articles, total = crawl_plankbevelen_by_page(page, limit)
    if not articles:  # 没有更多文章了
      break
    all_articles.extend(articles)
    page += 1
  return all_articles

def parse_article(art):
  """解析你的文章结构，提取标题、内容、链接"""
  try:
    title = art.get("title", "无标题")
    shortContent = art.get("shortContent", "")  # 正文  
    longContent = art.get("longContent", "")  # 正文
    slug = art.get("slug", f"{BLOG_HOST}/article/{art.get('id', '')}") # 文章链接
    url = slug
    tags = art.get("tags", []) # 标签
    category = art.get("category", "") # 分类标签
    create_time = art.get("create_time", "") # 创建时间
    update_time = art.get("update_time", "") # 更新时间

    return {
      "title": title,
      "shortContent": shortContent,
      "longContent": longContent,
      "url": url,
      "tags": tags,
      "category": category,
      "create_time": create_time,
      "update_time": update_time
    }
  except:
    return None

def crawl_my_blog():
  print("🚀 开始爬取 plankbevelen.cn 所有文章...")
  articles = fetch_plankbevelen_articles()
  result = []

  for art in articles:
    item = parse_article(art)
    if item:
      result.append(item)

  print(f"✅ 爬取完成！共 {len(result)} 篇文章")
  return result

articles = crawl_my_blog()
for idx, article in enumerate(articles):
  doc_id = f"blog_{idx+1:03d}"
  text = f"""
文章标题：{article['title']}
文章链接：{article['url']}
文章摘要：{article['shortContent']}
文章内容：{article['longContent']}
文章分类：{article['category']}
文章标签：{article['tags']}
文章创建时间：{article['create_time']}
文章更新时间：{article['update_time']}
"""
  kb.add(
    doc_id,
    text,
    {
      "source": article['url'],
      "type": "blog_article",
      "title": article['title'],
      "category": article['category'],
      "tags": article['tags'],
      "create_time": article['create_time'],
      "update_time": article['update_time'],
      "content": article['longContent'],
      "shortContent": article['shortContent'],
    }
  )

print("知识库初始化完成！")