def truncate_text(text, max_chars=5000):
    """如果字符串过长则截断（优先在句子/段落分隔符）"""
    if len(text) <= max_chars:
        return text
    # 简单做法：按字符截断
    return text[:max_chars] + "... [TRUNCATED]"