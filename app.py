# Import necessary libraries for the app
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
import textwrap
import csv
from datetime import datetime
from flask import Flask, render_template, request, send_file
import os
import pandas as pd
from textblob import TextBlob
import uuid
import json
import logging
from dotenv import load_dotenv
from io import StringIO  # Added for CSV generation

# Initialize Flask app
app = Flask(__name__)

# Set up logging instead of print statements
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load API key from environment variable for security
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define platform-specific rules
PLATFORM_RULES = {
    "instagram": {"max_chars": 150, "style": "casual", "hashtags": 3, "default_tone": "casual"},
    "linkedin": {"max_chars": 3000, "style": "professional", "hashtags": 2, "default_tone": "professional"},
    "blog": {"max_words": 1000, "style": "informal", "hashtags": 0, "default_tone": "casual"},
    "article": {"max_words": 2000, "style": "formal", "hashtags": 0, "default_tone": "professional"}
}

# Define tone descriptions
TONE_OPTIONS = {
    "professional": "formal, concise, polished, business-like",
    "casual": "relaxed, friendly, conversational",
    "humorous": "witty, playful, lighthearted"
}

# Research Agent
research_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    tools=[DuckDuckGo(), lambda q: f"X posts (March 28, 2025): {app.config['grok'].search_x(q)}"],
    instructions=[
        "Search DuckDuckGo for trending data on the topic and use the provided X search tool for real-time X insights.",
        "Combine findings into a concise summary (100-150 words).",
        "Summarize X insights as a unique, topic-specific trend without quoting specific posts verbatim.",
        "Avoid overusing 'buzzing' or 'abuzz'—vary phrasing like 'users are raving,' 'trending online,' or 'hot on social feeds.'",
        "Focus on diverse, real-time insights and avoid repetition."
    ]
)

# Writer Agent
writer_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    instructions=[
        "Write varied content using the research and user inputs (platform, goal, topic).",
        "Optimize for the goal: engagement (questions, hooks), visibility (keywords), branding (consistent tone).",
        "For blogs and articles, include a short heading followed by 2-3 concise paragraphs. For Instagram/LinkedIn, use short lines with clear breaks.",
        "Incorporate the social media trend summary creatively, avoiding repetitive phrases like 'buzzing'—use alternatives like 'trending hot,' 'users are hooked,' or 'online hype is real.'",
        "Avoid repeating phrases or ideas across drafts."
    ]
)

# Formatter Agent
formatter_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    instructions=[
        "Polish the content to fit the platform’s rules (length, style, hashtags).",
        "Remove duplicates and enforce max length strictly.",
        "Add specified hashtags naturally."
    ]
)

# Analytics Agent
analytics_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    instructions=[
        "Predict engagement metrics (likes, comments, shares) for two content variants based on their text.",
        "Consider factors like: presence of a question (boosts comments), bold hooks (boosts likes), trending keywords (boosts shares), and length (shorter boosts visibility).",
        "Assign realistic scores (e.g., 10-50 likes, 5-20 comments, 0-10 shares) for a platform like LinkedIn.",
        "Provide a brief reasoning (e.g., 'Variant B wins due to question driving comments').",
        "Return metrics as a dict: {'variant_a': {'likes': X, 'comments': Y, 'shares': Z}, 'variant_b': {...}}"
    ]
)

# Analyze user style from uploaded file
def analyze_style(file):
    posts = []
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
            if "content" not in df.columns:
                raise ValueError("CSV must have a 'content' column.")
            posts = df["content"].tolist()
        else:
            posts = file.read().decode("utf-8").splitlines()
        if not posts:
            raise ValueError("File is empty.")
        avg_sentence_length = sum(len(TextBlob(post).sentences) for post in posts) / len(posts)
        sentiment_scores = [TextBlob(post).sentiment.polarity for post in posts]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return {
            "sentence_length": "short" if avg_sentence_length < 10 else "long",
            "sentiment": "positive" if avg_sentiment > 0 else "neutral" if avg_sentiment == 0 else "negative"
        }
    except Exception as e:
        logger.error(f"Style analysis failed: {e}")
        return None

# Input validation function
def validate_inputs(platform, goal, tone, topic, count):
    errors = []
    if platform not in PLATFORM_RULES:
        errors.append("Invalid platform.")
    if goal not in ["engagement", "visibility", "branding"]:
        errors.append("Goal must be 'engagement', 'visibility', or 'branding'.")
    if tone not in TONE_OPTIONS:
        errors.append("Invalid tone.")
    if not topic or len(topic.strip()) < 3:
        errors.append("Topic must be at least 3 characters long.")
    try:
        count = int(count)
        if count < 1 or count > 50:
            errors.append("Count must be between 1 and 50.")
    except ValueError:
        errors.append("Count must be a valid integer.")
    return errors

# Core content generation function
def generate_content(platform, goal, topic, tone=None, persona=None):
    rules = PLATFORM_RULES[platform]
    tone = tone if tone else rules["default_tone"]
    tone_description = TONE_OPTIONS.get(tone, TONE_OPTIONS["casual"])
    
    try:
        research = research_agent.run(f"Find trending insights on {topic}.").content
        logger.info(f"Research Output: {research}")
    except Exception as e:
        logger.error(f"Research Error: {e}")
        research = f"Failed to fetch trends: {e}"
    
    # Variant A (Bold Hook)
    write_prompt_a = f"Using this research: '{research}', write content for {platform} optimized for {goal} on {topic}. Use a {tone_description} tone with a bold, direct hook. Include a general summary of real-time social media trends."
    if persona:
        write_prompt_a += f" Write like someone who uses {persona['sentence_length']} sentences and has a {persona['sentiment']} tone."
    if platform in ["blog", "article"]:
        write_prompt_a += " Include a short heading and 2-3 concise paragraphs."
    else:
        write_prompt_a += " Use short lines with clear breaks."
    
    draft_a = writer_agent.run(write_prompt_a).content
    logger.info(f"Writer Draft A: {draft_a}")
    format_prompt_a = f"Format this draft: '{draft_a}' for {platform}. Max {rules.get('max_chars', rules.get('max_words'))} {'chars' if 'max_chars' in rules else 'words'}, {rules['style']} style, {'include ' + str(rules['hashtags']) + ' hashtags' if rules['hashtags'] else 'no hashtags'}."
    variant_a = formatter_agent.run(format_prompt_a).content
    logger.info(f"Variant A: {variant_a}")
    
    # Variant B (Question-Driven)
    write_prompt_b = f"Using this research: '{research}', write content for {platform} optimized for {goal} on {topic}. Use a {tone_description} tone with a thought-provoking question. Include a general summary of real-time social media trends."
    if persona:
        write_prompt_b += f" Write like someone who uses {persona['sentence_length']} sentences and has a {persona['sentiment']} tone."
    if platform in ["blog", "article"]:
        write_prompt_b += " Include a short heading and 2-3 concise paragraphs."
    else:
        write_prompt_b += " Use short lines with clear breaks."
    
    draft_b = writer_agent.run(write_prompt_b).content
    logger.info(f"Writer Draft B: {draft_b}")
    format_prompt_b = f"Format this draft: '{draft_b}' for {platform}. Max {rules.get('max_chars', rules.get('max_words'))} {'chars' if 'max_chars' in rules else 'words'}, {rules['style']} style, {'include ' + str(rules['hashtags']) + 'hashtags' if rules['hashtags'] else 'no hashtags'}."
    variant_b = formatter_agent.run(format_prompt_b).content
    logger.info(f"Variant B: {variant_b}")
    
    if "max_chars" in rules:
        variant_a = textwrap.shorten(variant_a, width=rules["max_chars"], placeholder="...")
        variant_b = textwrap.shorten(variant_b, width=rules["max_chars"], placeholder="...")
    
    analytics_prompt = f"Predict engagement metrics for these variants on {platform}:\nVariant A: '{variant_a}'\nVariant B: '{variant_b}'"
    analytics_result = analytics_agent.run(analytics_prompt).content
    logger.info(f"Analytics Prediction: {analytics_result}")
    
    try:
        analytics = json.loads(analytics_result) if isinstance(analytics_result, str) else analytics_result
        analytics_a = analytics.get("variant_a", {"likes": 0, "comments": 0, "shares": 0})
        analytics_b = analytics.get("variant_b", {"likes": 0, "comments": 0, "shares": 0})
    except Exception as e:
        logger.error(f"Analytics Parse Error: {e}, using fallback")
        analytics_a = {"likes": 20, "comments": 5, "shares": 2}
        analytics_b = {"likes": 25, "comments": 10, "shares": 3}
    
    post_id = str(uuid.uuid4())
    return {
        "post_id": post_id,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "analytics_a": analytics_a,
        "analytics_b": analytics_b
    }

# Generate and export multiple posts
def generate_and_export(platform, goal, topic, tone, persona, count, filename="content_schedule.csv"):
    outputs = []
    for i in range(count):
        variants = generate_content(platform, goal, topic, tone, persona)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        outputs.append({
            "platform": platform,
            "post_id": variants["post_id"],
            "variant_a": variants["variant_a"],
            "variant_b": variants["variant_b"],
            "timestamp": timestamp,
            "analytics_a_likes": variants["analytics_a"]["likes"],
            "analytics_a_comments": variants["analytics_a"]["comments"],
            "analytics_a_shares": variants["analytics_a"]["shares"],
            "analytics_b_likes": variants["analytics_b"]["likes"],
            "analytics_b_comments": variants["analytics_b"]["comments"],
            "analytics_b_shares": variants["analytics_b"]["shares"]
        })
    return outputs  # Return outputs instead of writing to file

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        platform = request.form.get("platform")
        goal = request.form.get("goal")
        tone = request.form.get("tone")
        topic = request.form.get("topic")
        count = request.form.get("count", 1)
        
        errors = validate_inputs(platform, goal, tone, topic, count)
        if errors:
            return render_template("index.html", error="; ".join(errors))
        
        count = int(count)
        file = request.files.get("file") if "file" in request.files else None
        persona = analyze_style(file) if file and file.filename else None
        
        logger.info(f"Form Data: platform={platform}, goal={goal}, tone={tone}, topic={topic}, count={count}")
        
        posts = generate_and_export(platform, goal, topic, tone, persona, count)
        formatted_posts = [
            {
                "post_id": post["post_id"],
                "variant_a": post["variant_a"],
                "variant_b": post["variant_b"],
                "analytics_a": {"likes": post["analytics_a_likes"], "comments": post["analytics_a_comments"], "shares": post["analytics_a_shares"]},
                "analytics_b": {"likes": post["analytics_b_likes"], "comments": post["analytics_b_comments"], "shares": post["analytics_b_shares"]}
            } for post in posts
        ]
        logger.info(f"Generated Posts: {formatted_posts}")
        return render_template("index.html", posts=formatted_posts, posts_json=json.dumps(posts))
    
    return render_template("index.html")

# Download route
@app.route("/download")
def download_csv():
    posts = request.args.get('posts', None)
    if not posts:
        return "No posts available to download."
    
    posts = json.loads(posts)
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["platform", "post_id", "variant_a", "variant_b", "timestamp", 
                                                "analytics_a_likes", "analytics_a_comments", "analytics_a_shares",
                                                "analytics_b_likes", "analytics_b_comments", "analytics_b_shares"])
    writer.writeheader()
    writer.writerows(posts)
    output.seek(0)
    return send_file(
        StringIO(output.getvalue()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='content_schedule.csv'
    )

# Inject Grok instance into Flask app config
def create_app():
    app.config['grok'] = globals().get('__self__', None)
    return app

if __name__ == "__main__":
    app = create_app()
