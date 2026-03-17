"""
URL sanitization utilities for Jinja2 templates
"""

import re
import urllib.parse

def sanitize_image_url(image_path):
    """
    Sanitize image URL to remove control characters and properly encode it
    
    Args:
        image_path (str): The image path from database
        
    Returns:
        str: Sanitized URL-safe path
    """
    if not image_path:
        return ""
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str(image_path))
    
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
    
    # Normalize path separators
    sanitized = sanitized.replace('\\', '/')
    
    # URL encode the path properly
    sanitized = urllib.parse.quote(sanitized, safe='/')
    
    return sanitized

# Jinja2 filter functions
def register_filters(app):
    """Register custom filters with Jinja2 environment"""
    if hasattr(app, 'jinja_env'):
        app.jinja_env.filters['sanitize_url'] = sanitize_image_url
    elif hasattr(app, 'templating') and hasattr(app.templating, 'env'):
        app.templating.env.filters['sanitize_url'] = sanitize_image_url
