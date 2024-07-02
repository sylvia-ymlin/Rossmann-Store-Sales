import os

def get_frontend_html() -> str:
    """Loads and returns the demo page HTML."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        template_path = "web/templates/index.html"
    if not os.path.exists(template_path):
        return "<h1>Frontend Template Not Found</h1>"

    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()
